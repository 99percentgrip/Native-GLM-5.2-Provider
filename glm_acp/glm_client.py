"""Streaming client for the Z.ai GLM chat completions API.

Handles SSE parsing, reasoning_content / content separation, tool_call
assembly, and automatic continuation when the model hits finish_reason=length.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import (
    COMPACTION_SUMMARY_MAX_TOKENS,
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PREFIX,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    MAX_AUTO_CONTINUATIONS,
    get_api_key,
)


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

    def __init__(self, model: str = "glm-4.6", thought_level: str = "enabled"):
        self.model = model
        self.thought_level = thought_level
        self._api_key = get_api_key()
        self._client = httpx.AsyncClient(
            base_url=os.environ.get("ZAI_BASE_URL", DEFAULT_BASE_URL),
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, read=DEFAULT_TIMEOUT),
        )

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
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": COMPACTION_USER_PREFIX + transcript},
            ],
            "stream": False,
            "max_tokens": COMPACTION_SUMMARY_MAX_TOKENS,
            "thinking": {"type": "disabled"},
        }
        resp = await self._client.post("/chat/completions", json=body)
        if resp.status_code != 200:
            raise RuntimeError(
                f"GLM API error during summarization {resp.status_code}: {resp.text[:500]}"
            )
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
            content = msg.get("content", "")
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
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "thinking": {"type": self.thought_level},
        }
        if tools:
            body["tools"] = tools

        tool_accs: dict[int, ToolCallAccumulator] = {}
        announced: set[int] = set()

        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                raise RuntimeError(
                    f"GLM API error {resp.status_code}: {text.decode()[:500]}"
                )

            async for line in resp.aiter_lines():
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
