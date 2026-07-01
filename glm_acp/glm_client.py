"""Streaming client for the Z.ai GLM chat completions API.

Handles SSE parsing, reasoning_content / content separation, tool_call
assembly, and automatic continuation when the model hits finish_reason=length.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import (
    COMPACTION_SUMMARY_MAX_TOKENS,
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PREFIX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    MAX_AUTO_CONTINUATIONS,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRYABLE_STATUS_CODES,
    MAX_TOKENS_BY_MODEL,
    VISION_MODELS,
    get_api_key,
)


class GlmApiError(RuntimeError):
    """Raised when the Z.ai API returns a non-200 status code."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"GLM API error {status_code}: {message}")


logger = logging.getLogger("glm_acp")


@dataclass
class ToolCallAccumulator:
    """Assembles a single streamed tool call from incremental deltas."""

    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamResult:
    """Accumulated output from one or more API calls in a single model turn."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] | None = None  # input_tokens, output_tokens, total_tokens


class GlmClient:
    """Low-level streaming client for the Z.ai BigModel API."""

    def __init__(self, model: str = "glm-4.6", thought_level: str = "enabled", reasoning_effort: str | None = None, base_url: str | None = None):
        self.model = model
        self.thought_level = thought_level
        self.reasoning_effort = reasoning_effort
        self._cancelled = False
        self._api_key = get_api_key()
        self._client = httpx.AsyncClient(
            base_url=base_url or os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, read=DEFAULT_TIMEOUT),
        )

    def cancel(self) -> None:
        """Signal that the current operation should be aborted."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_reasoning: Any,
        on_content: Any,
        on_tool_call_started: Any,
    ) -> StreamResult:
        """Stream one model turn, calling back for each token delta.

        Auto-continues with a bare 'continue' message when finish_reason
        is 'length', keeping the turn alive until the model finishes
        naturally or the continuation cap is hit.
        """
        result = StreamResult()
        working_messages = list(messages)

        for _ in range(MAX_AUTO_CONTINUATIONS):
            if self._cancelled:
                break

            await self._do_stream_request(
                working_messages, tools, result,
                on_reasoning, on_content, on_tool_call_started,
            )

            if result.finish_reason == "length" and not result.tool_calls:
                working_messages = list(messages) + [
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": "Continue exactly where you left off. Do not repeat or summarize."},
                ]
                result.content += "\n"
                continue

            break

        return result

    async def summarize_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Summarize a conversation into a compact context-preserving block.

        Uses a dedicated summarization system prompt and a non-streaming
        request for simplicity (no tool calls, no thinking).
        """
        transcript = self._format_transcript(messages)
        max_tokens = min(
            COMPACTION_SUMMARY_MAX_TOKENS,
            MAX_TOKENS_BY_MODEL.get(self.model, DEFAULT_MAX_TOKENS),
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": COMPACTION_USER_PREFIX + transcript},
            ],
            "stream": False,
            "max_tokens": max_tokens,
        }
        # Vision models don't support the thinking parameter
        if self.model not in VISION_MODELS:
            body["thinking"] = {"type": "disabled"}

        # Retry with exponential backoff
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            resp = await self._client.post("/chat/completions", json=body)
            if resp.status_code == 200:
                break
            last_error = GlmApiError(resp.status_code, resp.text[:500])
            if resp.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise last_error
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "(compaction produced no summary)"
        content = choices[0].get("message", {}).get("content", "")
        return content.strip() or "(compaction produced no summary)"

    @staticmethod
    def _format_transcript(messages: list[dict[str, Any]]) -> str:
        """Render messages into a readable transcript for the summarizer."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            if role == "system":
                continue  # skip system prompt — summarizer has its own
            raw_content = msg.get("content")
            # Normalize content: None -> "", list -> extracted text
            if raw_content is None:
                content = ""
            elif isinstance(raw_content, list):
                content = " ".join(
                    b.get("text", "") for b in raw_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                content = str(raw_content)
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if tool_call_id:
                lines.append(f"[Tool Result] {content}")
            elif tool_calls:
                tc_strs = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tc_strs.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
                lines.append(f"[Assistant Tool Calls: {', '.join(tc_strs)}]")
                if content:
                    lines.append(f"[Assistant Content] {content}")
            elif role == "user":
                lines.append(f"[User] {content}")
            elif role == "assistant":
                lines.append(f"[Assistant] {content}")
            else:
                lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    async def _do_stream_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        result: StreamResult,
        on_reasoning: Any,
        on_content: Any,
        on_tool_call_started: Any,
    ) -> None:
        max_tokens = MAX_TOKENS_BY_MODEL.get(self.model, DEFAULT_MAX_TOKENS)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            # Request usage in the stream so we get real token counts
            "stream_options": {"include_usage": True},
        }
        # Thinking / reasoning_effort only applies to text reasoning models,
        # not vision models.
        is_vision = self.model in VISION_MODELS
        if not is_vision:
            body["thinking"] = {"type": self.thought_level}
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools

        # --- Retry with exponential backoff on transient errors ---
        for attempt in range(MAX_RETRIES + 1):
            if self._cancelled:
                return

            try:
                await self._execute_stream(body, result, on_reasoning, on_content, on_tool_call_started)
                return  # success
            except GlmApiError as e:
                if e.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "API error %d, retrying in %.1fs (attempt %d/%d)",
                    e.status_code, delay, attempt + 1, MAX_RETRIES,
                )
                # Clear partial results from the failed attempt so retry
                # doesn't produce duplicate content
                result.content = ""
                result.reasoning = ""
                result.tool_calls = []
                result.finish_reason = ""
                await asyncio.sleep(delay)
            except httpx.TransportError as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Network error after {MAX_RETRIES} retries: {e}")
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                result.content = ""
                result.reasoning = ""
                result.tool_calls = []
                result.finish_reason = ""
                await asyncio.sleep(delay)

    async def _execute_stream(
        self,
        body: dict[str, Any],
        result: StreamResult,
        on_reasoning: Any,
        on_content: Any,
        on_tool_call_started: Any,
    ) -> None:
        """Execute a single streaming request (no retry logic)."""
        tool_accs: dict[int, ToolCallAccumulator] = {}
        announced: set[int] = set()

        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                raise GlmApiError(resp.status_code, text.decode(errors="replace")[:500])

            async for line in resp.aiter_lines():
                if self._cancelled:
                    result.finish_reason = "cancelled"
                    return

                if not line or not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish = choices[0].get("finish_reason")

                reasoning = delta.get("reasoning_content") or ""
                content = delta.get("content") or ""

                if reasoning:
                    result.reasoning += reasoning
                    if on_reasoning:
                        await on_reasoning(reasoning)

                if content:
                    result.content += content
                    if on_content:
                        await on_content(content)

                tc_deltas = delta.get("tool_calls")
                if tc_deltas:
                    for tc in tc_deltas:
                        idx = tc.get("index", 0)
                        if idx not in tool_accs:
                            tool_accs[idx] = ToolCallAccumulator(
                                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            )
                        acc = tool_accs[idx]
                        if tc.get("function", {}).get("name"):
                            acc.name = tc["function"]["name"]
                        if tc.get("function", {}).get("arguments"):
                            acc.arguments += tc["function"]["arguments"]

                        if on_tool_call_started and acc.name and idx not in announced:
                            announced.add(idx)
                            await on_tool_call_started(acc.id, acc.name)

                if finish:
                    result.finish_reason = finish

                # Z.ai sends usage in the last chunk(s) — capture it whenever
                # it appears rather than only after the loop ends.
                usage = chunk.get("usage")
                if usage:
                    result.usage = {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

        for idx in sorted(tool_accs):
            acc = tool_accs[idx]
            if not acc.name:
                continue
            try:
                args = json.loads(acc.arguments) if acc.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": acc.arguments}
            result.tool_calls.append({
                "id": acc.id,
                "type": "function",
                "function": {"name": acc.name, "arguments": args},
            })

    async def aclose(self) -> None:
        await self._client.aclose()
