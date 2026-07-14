"""Streaming client for the Z.ai GLM chat completions API.

Handles SSE parsing, reasoning_content / content separation, tool_call
assembly, and automatic continuation when the model hits finish_reason=length.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
    MAX_TOKENS_BY_MODEL,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    RETRYABLE_STATUS_CODES,
    THINKING_UNSUPPORTED_MODELS,
    get_api_key,
)


class GlmApiError(RuntimeError):
    """Raised when the Z.ai API returns a non-200 status code."""

    def __init__(self, status_code: int, message: str, retry_after: str | None = None) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"GLM API error {status_code}: {message}")


class CompactionError(RuntimeError):
    """Raised when a compaction response cannot safely replace history."""


class IncompleteStreamError(RuntimeError):
    """Raised when an HTTP 200 SSE response ends without a terminal reason."""


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

    def __init__(
        self,
        model: str = "glm-4.6",
        thought_level: str = "enabled",
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        self.model = model
        self.thought_level = thought_level
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.top_p = top_p
        endpoint = base_url or os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
        self.preserve_thinking = thought_level == "enabled" and (
            reasoning_effort in {"high", "max"} or "/coding/" in endpoint
        )
        self._cancelled = False
        self._active_request_task: asyncio.Task[Any] | None = None
        self._api_key = get_api_key()
        self._client = httpx.AsyncClient(
            base_url=endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, read=DEFAULT_TIMEOUT),
        )

    def cancel(self) -> None:
        """Abort the active HTTP operation and mark the turn cancelled."""
        self._cancelled = True
        task = self._active_request_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def begin_turn(self) -> None:
        """Reset turn-local cancellation state before a serialized prompt."""
        self._cancelled = False

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

        for request_index in range(MAX_AUTO_CONTINUATIONS + 1):
            if self._cancelled:
                break

            await self._do_stream_request(
                working_messages,
                tools,
                result,
                on_reasoning,
                on_content,
                on_tool_call_started,
            )

            if result.finish_reason == "length" and not result.tool_calls:
                if request_index >= MAX_AUTO_CONTINUATIONS:
                    result.finish_reason = "continuation_limit"
                    break
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": result.content,
                }
                if self.preserve_thinking and result.reasoning:
                    assistant_message["reasoning_content"] = result.reasoning
                working_messages = list(messages) + [
                    assistant_message,
                    {
                        "role": "user",
                        "content": (
                            "Continue exactly where you left off. Do not repeat or summarize."
                        ),
                    },
                ]
                result.content += "\n"
                if on_content:
                    await on_content("\n")
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
        if self.model not in THINKING_UNSUPPORTED_MODELS:
            body["thinking"] = {"type": "disabled"}

        # Retry with exponential backoff
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            resp = await self._client.post("/chat/completions", json=body)
            if resp.status_code == 200:
                break
            # Decode error body safely — may contain non-UTF8 bytes
            try:
                err_text = resp.text[:500]
            except Exception:
                err_text = resp.content[:500].decode(errors="replace")
            last_error = GlmApiError(
                resp.status_code,
                err_text,
                getattr(resp, "headers", {}).get("Retry-After"),
            )
            if resp.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise last_error
            delay = self._retry_delay(attempt, last_error.retry_after)
            await asyncio.sleep(delay)

        try:
            data = resp.json()
        except Exception as exc:
            raise CompactionError("Compaction response did not contain a valid summary") from exc
        choices = data.get("choices", [])
        if not choices:
            raise CompactionError("Compaction response did not contain a summary")
        content = choices[0].get("message", {}).get("content", "")
        content = content.strip()
        if not content:
            raise CompactionError("Compaction response contained an empty summary")
        return content

    @staticmethod
    def _format_transcript(messages: list[dict[str, Any]]) -> str:
        """Render messages into a readable transcript for the summarizer."""
        lines: list[str] = []
        tool_labels: dict[str, str] = {}
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
                    b.get("text", "")
                    for b in raw_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                content = str(raw_content)
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if tool_call_id:
                label = tool_labels.get(str(tool_call_id), "unknown tool")
                lines.append(f"[Tool Result {tool_call_id} — {label}] {content}")
            elif tool_calls:
                tc_strs = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    call_id = str(tc.get("id", "?"))
                    label = f"{fn.get('name', '?')}({fn.get('arguments', '')})"
                    tool_labels[call_id] = label
                    tc_strs.append(f"{call_id}: {label}")
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
        if self.model not in THINKING_UNSUPPORTED_MODELS:
            body["thinking"] = {
                "type": self.thought_level,
                "clear_thinking": not self.preserve_thinking,
            }
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
            body["tool_stream"] = True
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p

        # --- Retry with exponential backoff on transient errors ---
        for attempt in range(MAX_RETRIES + 1):
            if self._cancelled:
                return

            attempt_result = StreamResult()
            try:
                await self._execute_stream(
                    body,
                    attempt_result,
                    on_reasoning,
                    on_content,
                    on_tool_call_started,
                )
                self._merge_result(result, attempt_result)
                return  # success
            except IncompleteStreamError:
                if attempt_result.content or attempt_result.reasoning:
                    self._merge_result(result, attempt_result)
                    result.finish_reason = "network_error"
                    return
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(self._retry_delay(attempt))
            except GlmApiError as e:
                if e.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                    raise
                delay = self._retry_delay(attempt, e.retry_after)
                logger.warning(
                    "API error %d, retrying in %.1fs (attempt %d/%d)",
                    e.status_code,
                    delay,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(delay)
            except httpx.TransportError as e:
                if attempt_result.content or attempt_result.reasoning:
                    self._merge_result(result, attempt_result)
                    result.finish_reason = "network_error"
                    return
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Network error after {MAX_RETRIES} retries: {e}")
                delay = self._retry_delay(attempt)
                await asyncio.sleep(delay)

    @staticmethod
    def _merge_result(result: StreamResult, attempt: StreamResult) -> None:
        """Merge one completed HTTP attempt into a logical model turn."""
        result.content += attempt.content
        result.reasoning += attempt.reasoning
        result.tool_calls.extend(attempt.tool_calls)
        result.finish_reason = attempt.finish_reason
        if attempt.usage:
            previous_input = (result.usage or {}).get("api_input_tokens", 0)
            previous_output = (result.usage or {}).get("output_tokens", 0)
            previous_cached = (result.usage or {}).get("cached_tokens", 0)
            result.usage = dict(attempt.usage)
            result.usage["api_input_tokens"] = previous_input + attempt.usage.get("input_tokens", 0)
            result.usage["output_tokens"] = previous_output + attempt.usage.get("output_tokens", 0)
            result.usage["cached_tokens"] = previous_cached + attempt.usage.get("cached_tokens", 0)
            result.usage["total_tokens"] = (
                result.usage.get("input_tokens", 0) + result.usage["output_tokens"]
            )

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
        """Return a capped, jittered delay while honoring Retry-After."""
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    delay = (when - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    delay = 0.0
            if delay > 0:
                return min(delay, RETRY_MAX_DELAY)
        ceiling = min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)
        return random.uniform(ceiling * 0.75, ceiling)

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
        tool_argument_parts: dict[int, list[str]] = {}
        announced: set[int] = set()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        pending_content: list[str] = []
        pending_reasoning: list[str] = []
        pending_count = 0
        emitted_any = False
        terminal_seen = False

        def sync_result() -> None:
            result.content = "".join(content_parts)
            result.reasoning = "".join(reasoning_parts)

        async def flush(force: bool = False) -> None:
            nonlocal pending_count, emitted_any
            if not force and emitted_any and pending_count < 16:
                return
            reasoning_batch = "".join(pending_reasoning)
            content_batch = "".join(pending_content)
            pending_reasoning.clear()
            pending_content.clear()
            pending_count = 0
            emitted_any = emitted_any or bool(reasoning_batch or content_batch)
            if reasoning_batch and on_reasoning:
                await on_reasoning(reasoning_batch)
            if content_batch and on_content:
                await on_content(content_batch)

        self._active_request_task = asyncio.current_task()
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise GlmApiError(
                        resp.status_code,
                        text.decode(errors="replace")[:500],
                        getattr(resp, "headers", {}).get("Retry-After"),
                    )

                async for line in resp.aiter_lines():
                    if self._cancelled:
                        result.finish_reason = "cancelled"
                        terminal_seen = True
                        return

                    if not line or not line.startswith("data:"):
                        continue

                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        terminal_seen = True
                        break
                    if not data:
                        continue

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        usage = chunk.get("usage")
                        if usage:
                            details = usage.get("prompt_tokens_details") or {}
                            result.usage = {
                                "input_tokens": usage.get("prompt_tokens", 0),
                                "output_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                                "cached_tokens": details.get("cached_tokens", 0),
                            }
                        continue
                    delta = choices[0].get("delta", {})
                    finish = choices[0].get("finish_reason")

                    reasoning = delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""

                    if reasoning:
                        reasoning_parts.append(reasoning)
                        pending_reasoning.append(reasoning)
                        pending_count += 1

                    if content:
                        content_parts.append(content)
                        pending_content.append(content)
                        pending_count += 1

                    if reasoning or content:
                        sync_result()
                        await flush()

                    tc_deltas = delta.get("tool_calls")
                    if tc_deltas:
                        await flush(force=True)
                        for tc in tc_deltas:
                            idx = tc.get("index", 0)
                            if idx not in tool_accs:
                                tool_accs[idx] = ToolCallAccumulator(
                                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                )
                                tool_argument_parts[idx] = []
                            acc = tool_accs[idx]
                            if tc.get("function", {}).get("name"):
                                acc.name = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                tool_argument_parts[idx].append(tc["function"]["arguments"])

                            if on_tool_call_started and acc.name and idx not in announced:
                                announced.add(idx)
                                await on_tool_call_started(acc.id, acc.name)

                    if finish:
                        result.finish_reason = finish
                        terminal_seen = True

                    usage = chunk.get("usage")
                    if usage:
                        details = usage.get("prompt_tokens_details") or {}
                        result.usage = {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "cached_tokens": details.get("cached_tokens", 0),
                        }
        finally:
            if self._active_request_task is asyncio.current_task():
                self._active_request_task = None
            sync_result()
            await flush(force=True)

        if not terminal_seen:
            raise IncompleteStreamError("GLM stream ended before a terminal event")

        for idx in sorted(tool_accs):
            acc = tool_accs[idx]
            if not acc.name:
                continue
            try:
                acc.arguments = "".join(tool_argument_parts.get(idx, []))
                args = json.loads(acc.arguments) if acc.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": acc.arguments}
            result.tool_calls.append(
                {
                    "id": acc.id,
                    "type": "function",
                    "function": {"name": acc.name, "arguments": args},
                }
            )

    async def aclose(self) -> None:
        await self._client.aclose()
