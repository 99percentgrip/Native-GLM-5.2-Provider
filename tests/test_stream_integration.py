"""Integration tests for GlmClient._execute_stream with mock SSE responses.

These tests feed fake SSE byte streams through the real parser,
exercising the full chunk-parsing, tool-call assembly, usage tracking,
and error-handling logic without hitting the real API.
"""

import json
from unittest.mock import AsyncMock

import pytest

os = __import__("os")
os.environ.setdefault("ZAI_API_KEY", "test-key")

from glm_acp.glm_client import GlmApiError, GlmClient, StreamResult

# ============================================================
# Helpers: build mock SSE responses
# ============================================================


def _sse_line(data: dict | str) -> bytes:
    """Build a single SSE line as bytes."""
    if isinstance(data, str):
        return f"data: {data}\n".encode()
    return f"data: {json.dumps(data)}\n".encode()


def _sse_chunk(
    content: str = "",
    reasoning: str = "",
    tool_calls: list[dict] | None = None,
    finish: str | None = None,
    usage: dict | None = None,
) -> dict:
    """Build a single SSE chunk dict (choices[0].delta format)."""
    chunk: dict = {"choices": [{"delta": {}, "index": 0}]}
    delta = chunk["choices"][0]["delta"]
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    if tool_calls:
        delta["tool_calls"] = tool_calls
    if finish:
        chunk["choices"][0]["finish_reason"] = finish
    if usage:
        chunk["usage"] = usage
    return chunk


class MockStreamResponse:
    """Mock httpx streaming response for _execute_stream."""

    def __init__(self, lines: list[bytes], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code
        self._body = b"".join(lines)
        self.headers = {}

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            # Each line may contain trailing \n
            text = line.decode()
            for sub in text.split("\n"):
                if sub:
                    yield sub

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockHttpClient:
    """Mock httpx.AsyncClient that returns canned SSE responses."""

    def __init__(self, lines: list[bytes], status_code: int = 200):
        self._lines = lines
        self._status = status_code
        self.requests: list[dict] = []

    def stream(self, method: str, url: str, **kwargs):
        self.requests.append(kwargs)
        return MockStreamResponse(self._lines, self._status)

    async def aclose(self):
        pass


def _make_client(lines: list[bytes], status_code: int = 200) -> GlmClient:
    """Create a GlmClient with a mock httpx client."""
    client = GlmClient(model="glm-5.2")
    client._client = MockHttpClient(lines, status_code)
    return client


# ============================================================
# Content streaming
# ============================================================


class TestStreamContent:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Multiple content chunks should accumulate into result.content."""
        lines = [
            _sse_line(_sse_chunk(content="Hello")),
            _sse_line(_sse_chunk(content=" world")),
            _sse_line(_sse_chunk(content="!", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.content == "Hello world!"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_empty_content_response(self):
        """Response with no content should not crash."""
        lines = [
            _sse_line(_sse_chunk(finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.content == ""
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_content_callback_fires(self):
        """on_content callback should fire for each content chunk."""
        lines = [
            _sse_line(_sse_chunk(content="AB")),
            _sse_line(_sse_chunk(content="CD", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        chunks = []
        await client._execute_stream(
            {},
            result,
            None,
            AsyncMock(side_effect=lambda c: chunks.append(c)),
            None,
        )
        assert "".join(chunks) == "ABCD"
        assert result.content == "ABCD"

    @pytest.mark.asyncio
    async def test_small_deltas_are_coalesced(self):
        """ACP updates should not be emitted once per tiny model delta."""
        lines = [
            *[_sse_line(_sse_chunk(content="x")) for _ in range(40)],
            _sse_line(_sse_chunk(finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        chunks = []

        await client._execute_stream(
            {},
            result,
            None,
            AsyncMock(side_effect=lambda chunk: chunks.append(chunk)),
            None,
        )

        assert result.content == "x" * 40
        assert "".join(chunks) == result.content
        assert len(chunks) < 10


# ============================================================
# Reasoning streaming
# ============================================================


class TestStreamReasoning:
    @pytest.mark.asyncio
    async def test_tools_enable_provider_tool_streaming(self):
        lines = [
            _sse_line(_sse_chunk(content="answer", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._do_stream_request([], [{"type": "function"}], result, None, None, None)
        assert client._client.requests[0]["json"]["tool_stream"] is True

    @pytest.mark.asyncio
    async def test_sampling_profile_adds_only_selected_parameter(self):
        lines = [
            _sse_line(_sse_chunk(content="answer", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        client.temperature = 0.7
        result = StreamResult()
        await client._do_stream_request([], [], result, None, None, None)
        body = client._client.requests[0]["json"]
        assert body["temperature"] == 0.7
        assert "top_p" not in body

    @pytest.mark.asyncio
    async def test_deep_reasoning_preserves_history_in_request(self):
        lines = [
            _sse_line(_sse_chunk(content="answer", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        client.reasoning_effort = "high"
        client.preserve_thinking = True
        result = StreamResult()

        await client._do_stream_request([], [], result, None, None, None)

        body = client._client.requests[0]["json"]
        assert body["thinking"] == {"type": "enabled", "clear_thinking": False}
        assert body["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_reasoning_accumulates(self):
        """Reasoning content should accumulate separately from content."""
        lines = [
            _sse_line(_sse_chunk(reasoning="Let me think...")),
            _sse_line(_sse_chunk(reasoning=" about this.")),
            _sse_line(_sse_chunk(content="The answer is 42.", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.reasoning == "Let me think... about this."
        assert result.content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_reasoning_callback_fires(self):
        """on_reasoning callback should fire for each reasoning chunk."""
        lines = [
            _sse_line(_sse_chunk(reasoning="thinking...")),
            _sse_line(_sse_chunk(content="answer", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        reasoning_chunks = []
        await client._execute_stream(
            {},
            result,
            AsyncMock(side_effect=lambda r: reasoning_chunks.append(r)),
            None,
            None,
        )
        assert reasoning_chunks == ["thinking..."]


# ============================================================
# Tool call assembly
# ============================================================


class TestStreamToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        """A complete tool call should be assembled from deltas."""
        lines = [
            _sse_line(
                _sse_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":'},
                        }
                    ]
                )
            ),
            _sse_line(
                _sse_chunk(tool_calls=[{"index": 0, "function": {"arguments": ' "main.py"}'}}])
            ),
            _sse_line(_sse_chunk(finish="tool_calls")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "read_file"
        assert tc["function"]["arguments"] == {"path": "main.py"}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Multiple tool calls in the same response."""
        lines = [
            _sse_line(
                _sse_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                        }
                    ]
                )
            ),
            _sse_line(
                _sse_chunk(
                    tool_calls=[
                        {
                            "index": 1,
                            "id": "call_2",
                            "function": {"name": "read_file", "arguments": '{"path": "b.py"}'},
                        }
                    ]
                )
            ),
            _sse_line(_sse_chunk(finish="tool_calls")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["function"]["arguments"]["path"] == "a.py"
        assert result.tool_calls[1]["function"]["arguments"]["path"] == "b.py"

    @pytest.mark.asyncio
    async def test_tool_call_started_callback(self):
        """on_tool_call_started should fire when name first appears."""
        lines = [
            _sse_line(
                _sse_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "write_file", "arguments": "{}"},
                        }
                    ]
                )
            ),
            _sse_line(_sse_chunk(finish="tool_calls")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        calls = []
        await client._execute_stream(
            {},
            result,
            None,
            None,
            AsyncMock(side_effect=lambda tc_id, name: calls.append((tc_id, name))),
        )
        assert len(calls) == 1
        assert calls[0] == ("call_1", "write_file")

    @pytest.mark.asyncio
    async def test_tool_call_no_name_skipped(self):
        """Tool call deltas without a name should be skipped in final list."""
        lines = [
            _sse_line(
                _sse_chunk(
                    tool_calls=[{"index": 0, "id": "call_1", "function": {"arguments": "{}"}}]
                )
            ),
            _sse_line(_sse_chunk(finish="tool_calls")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        # No name means it gets skipped in the final assembly
        assert len(result.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_tool_call_malformed_arguments(self):
        """Malformed JSON arguments should become {"_raw": ...}."""
        lines = [
            _sse_line(
                _sse_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": "{broken"},
                        }
                    ]
                )
            ),
            _sse_line(_sse_chunk(finish="tool_calls")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["arguments"] == {"_raw": "{broken"}

    @pytest.mark.asyncio
    async def test_tool_call_empty_arguments(self):
        """Empty arguments string should become empty dict."""
        lines = [
            _sse_line(
                _sse_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "update_plan", "arguments": ""},
                        }
                    ]
                )
            ),
            _sse_line(_sse_chunk(finish="tool_calls")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["arguments"] == {}

    @pytest.mark.asyncio
    async def test_tool_call_auto_id(self):
        """Tool call without an id should get an auto-generated one."""
        lines = [
            _sse_line(
                _sse_chunk(
                    tool_calls=[{"index": 0, "function": {"name": "read_file", "arguments": "{}"}}]
                )
            ),
            _sse_line(_sse_chunk(finish="tool_calls")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"].startswith("call_")  # auto-generated


# ============================================================
# Usage tracking
# ============================================================


class TestStreamUsage:
    @pytest.mark.asyncio
    async def test_usage_captured(self):
        """Usage data should be captured from the final chunk."""
        lines = [
            _sse_line(_sse_chunk(content="hello", finish="stop")),
            _sse_line(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                }
            ),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.usage is not None
        assert result.usage["input_tokens"] == 100
        assert result.usage["output_tokens"] == 50
        assert result.usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_no_usage(self):
        """Response without usage should leave result.usage as None."""
        lines = [
            _sse_line(_sse_chunk(content="hello", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.usage is None


# ============================================================
# Error handling
# ============================================================


class TestStreamErrors:
    @pytest.mark.asyncio
    async def test_non_200_raises_glm_error(self):
        """Non-200 status should raise GlmApiError."""
        lines = [b'{"error": "bad request"}']
        client = _make_client(lines, status_code=400)
        result = StreamResult()
        with pytest.raises(GlmApiError) as exc_info:
            await client._execute_stream({}, result, None, None, None)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_non_200_binary_body(self):
        """Non-200 with binary body should not crash on decode."""
        lines = [b"\xff\xfe not valid utf8"]
        client = _make_client(lines, status_code=500)
        result = StreamResult()
        with pytest.raises(GlmApiError) as exc_info:
            await client._execute_stream({}, result, None, None, None)
        assert exc_info.value.status_code == 500


# ============================================================
# Edge cases: malformed SSE, empty lines, garbage
# ============================================================


class TestStreamMalformed:
    @pytest.mark.asyncio
    async def test_skips_non_data_lines(self):
        """Lines not starting with 'data:' should be skipped."""
        lines = [
            b": ping\n",
            b"event: message\n",
            _sse_line(_sse_chunk(content="real content", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.content == "real content"

    @pytest.mark.asyncio
    async def test_skips_invalid_json(self):
        """Invalid JSON in data lines should be skipped."""
        lines = [
            b"data: {broken json\n",
            _sse_line(_sse_chunk(content="good", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.content == "good"

    @pytest.mark.asyncio
    async def test_empty_data_line_skipped(self):
        """data: with empty payload should be skipped."""
        lines = [
            b"data: \n",
            _sse_line(_sse_chunk(content="content", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.content == "content"

    @pytest.mark.asyncio
    async def test_empty_choices_skipped(self):
        """Chunks with empty choices array should be skipped (but usage captured)."""
        lines = [
            _sse_line(_sse_chunk(content="text", finish="stop")),
            _sse_line(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.content == "text"
        assert result.usage is not None
        assert result.usage["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_no_done_marker(self):
        """Stream without [DONE] should still process all chunks."""
        lines = [
            _sse_line(_sse_chunk(content="hello", finish="stop")),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_delta_missing_content_key(self):
        """Delta without content key should not crash."""
        lines = [
            _sse_line({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.content == ""
        assert result.finish_reason == "stop"


# ============================================================
# Cancellation
# ============================================================


class TestStreamCancel:
    @pytest.mark.asyncio
    async def test_cancel_sets_finish_reason(self):
        """Cancel flag should set finish_reason and return early."""
        lines = [
            _sse_line(_sse_chunk(content="partial")),
            _sse_line(_sse_chunk(content=" more", finish="stop")),
            _sse_line("[DONE]"),
        ]
        client = _make_client(lines)
        client._cancelled = True
        result = StreamResult()
        await client._execute_stream({}, result, None, None, None)
        assert result.finish_reason == "cancelled"
        # Content from first chunk may or may not be captured depending
        # on when the cancel check fires — but it should not crash


# ============================================================
# Retry logic (via _do_stream_request)
# ============================================================


class TestStreamRetry:
    @pytest.mark.asyncio
    async def test_retry_on_500_then_success(self):
        """Should retry on 500, then succeed on second attempt."""
        fail_lines = [b"server error"]
        success_lines = [
            _sse_line(_sse_chunk(content="recovered", finish="stop")),
            _sse_line("[DONE]"),
        ]

        call_count = 0

        class FlakyClient:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return MockStreamResponse(fail_lines, status_code=500)
                return MockStreamResponse(success_lines, status_code=200)

            async def aclose(self):
                pass

        client = GlmClient(model="glm-5.2")
        client._client = FlakyClient()

        result = StreamResult()
        await client._do_stream_request({}, [], result, None, None, None)
        assert result.content == "recovered"
        assert call_count == 2  # failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_retry_clears_partial_content(self):
        """Retry should clear partial content from the failed attempt."""
        # First attempt: partial content then 500
        fail_lines = [
            _sse_line(_sse_chunk(content="partial ")),
            b"then error",
        ]
        success_lines = [
            _sse_line(_sse_chunk(content="clean", finish="stop")),
            _sse_line("[DONE]"),
        ]

        call_count = 0

        class FlakyClient:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return MockStreamResponse(fail_lines, status_code=200)
                return MockStreamResponse(success_lines, status_code=200)

            async def aclose(self):
                pass

        client = GlmClient(model="glm-5.2")
        client._client = FlakyClient()

        emitted = []
        result = StreamResult()
        await client._do_stream_request(
            {},
            [],
            result,
            None,
            AsyncMock(side_effect=lambda chunk: emitted.append(chunk)),
            None,
        )

        assert call_count == 1
        assert result.content == "partial "
        assert "".join(emitted) == "partial "
        assert result.finish_reason == "network_error"

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        """400 error should raise immediately without retry."""
        lines = [b'{"error": "bad request"}']

        call_count = 0

        class FailClient:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return MockStreamResponse(lines, status_code=400)

            async def aclose(self):
                pass

        client = GlmClient(model="glm-5.2")
        client._client = FailClient()

        result = StreamResult()
        with pytest.raises(GlmApiError):
            await client._do_stream_request({}, [], result, None, None, None)
        assert call_count == 1  # no retry


# ============================================================
# Auto-continuation (via stream_completion)
# ============================================================


class TestAutoContinuation:
    @pytest.mark.asyncio
    async def test_continuation_cap_is_reported(self, monkeypatch):
        lines = [
            _sse_line(_sse_chunk(content="part", finish="length")),
            _sse_line("[DONE]"),
        ]
        call_count = 0

        class LengthClient:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return MockStreamResponse(lines, status_code=200)

            async def aclose(self):
                pass

        monkeypatch.setattr("glm_acp.glm_client.MAX_AUTO_CONTINUATIONS", 2)
        client = GlmClient(model="glm-5.2")
        client._client = LengthClient()
        result = await client.stream_completion(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            on_reasoning=None,
            on_content=None,
            on_tool_call_started=None,
        )

        assert call_count == 3  # initial request + two continuations
        assert result.finish_reason == "continuation_limit"

    @pytest.mark.asyncio
    async def test_length_triggers_continuation(self):
        """finish_reason=length without tool_calls should auto-continue."""
        # First stream: content + finish=length
        first_lines = [
            _sse_line(_sse_chunk(content="Part 1", finish="length")),
            _sse_line("[DONE]"),
        ]
        # Second stream: more content + finish=stop
        second_lines = [
            _sse_line(_sse_chunk(content=" Part 2", finish="stop")),
            _sse_line("[DONE]"),
        ]

        call_count = 0

        class ContinuationClient:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return MockStreamResponse(first_lines, status_code=200)
                return MockStreamResponse(second_lines, status_code=200)

            async def aclose(self):
                pass

        client = GlmClient(model="glm-5.2")
        client._client = ContinuationClient()

        result = await client.stream_completion(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            on_reasoning=None,
            on_content=None,
            on_tool_call_started=None,
        )
        assert "Part 1" in result.content
        assert "Part 2" in result.content
        assert result.finish_reason == "stop"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_continuation_on_tool_calls(self):
        """finish_reason=length WITH tool_calls should NOT auto-continue."""
        lines = [
            _sse_line(
                _sse_chunk(
                    content="",
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                    finish="length",
                )
            ),
            _sse_line("[DONE]"),
        ]

        call_count = 0

        class SingleClient:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return MockStreamResponse(lines, status_code=200)

            async def aclose(self):
                pass

        client = GlmClient(model="glm-5.2")
        client._client = SingleClient()

        result = await client.stream_completion(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            on_reasoning=None,
            on_content=None,
            on_tool_call_started=None,
        )
        assert call_count == 1  # no continuation
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_stop_does_not_continue(self):
        """finish_reason=stop should NOT trigger continuation."""
        lines = [
            _sse_line(_sse_chunk(content="done", finish="stop")),
            _sse_line("[DONE]"),
        ]

        call_count = 0

        class SingleClient:
            def stream(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1
                return MockStreamResponse(lines, status_code=200)

            async def aclose(self):
                pass

        client = GlmClient(model="glm-5.2")
        client._client = SingleClient()

        result = await client.stream_completion(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            on_reasoning=None,
            on_content=None,
            on_tool_call_started=None,
        )
        assert call_count == 1
        assert result.finish_reason == "stop"
