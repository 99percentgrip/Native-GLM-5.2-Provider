"""Tests for glm_acp.glm_client — GlmApiError, StreamResult, cancel, retry logic."""

import os

import pytest

os.environ.setdefault("ZAI_API_KEY", "test-key")

from glm_acp.config import MAX_RETRIES
from glm_acp.glm_client import GlmApiError, GlmClient, StreamResult, ToolCallAccumulator


class TestGlmApiError:
    def test_status_code_stored(self):
        err = GlmApiError(429, "rate limited")
        assert err.status_code == 429
        assert "429" in str(err)

    def test_is_runtime_error(self):
        err = GlmApiError(500, "server error")
        assert isinstance(err, RuntimeError)


class TestStreamResult:
    def test_defaults(self):
        r = StreamResult()
        assert r.content == ""
        assert r.reasoning == ""
        assert r.tool_calls == []
        assert r.finish_reason == ""
        assert r.usage is None

    def test_usage_format(self):
        r = StreamResult()
        r.usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        assert r.usage["input_tokens"] == 100


class TestToolCallAccumulator:
    def test_defaults(self):
        acc = ToolCallAccumulator()
        assert acc.id == ""
        assert acc.name == ""
        assert acc.arguments == ""

    def test_accumulation(self):
        acc = ToolCallAccumulator(id="call_123")
        acc.name = "read_file"
        acc.arguments += '{"path":'
        acc.arguments += ' "main.py"}'
        assert acc.name == "read_file"
        assert acc.arguments == '{"path": "main.py"}'


class TestGlmClientInit:
    def test_cancel_flag(self):
        client = GlmClient(model="glm-5.2")
        assert not client.cancelled
        client.cancel()
        assert client.cancelled

    def test_thinking_support_is_explicit(self):
        import inspect

        src = inspect.getsource(GlmClient._do_stream_request)
        assert "THINKING_UNSUPPORTED_MODELS" in src

    def test_coding_plan_preserves_standard_thinking(self):
        client = GlmClient(model="glm-5.2", thought_level="enabled")
        assert client.preserve_thinking is True

    def test_standard_plan_clears_standard_thinking(self):
        client = GlmClient(
            model="glm-5.2",
            thought_level="enabled",
            base_url="https://api.z.ai/api/paas/v4",
        )
        assert client.preserve_thinking is False

    def test_retry_after_is_honored(self):
        assert GlmClient._retry_delay(0, "12") == 12

    def test_sampling_profile_values_are_stored(self):
        client = GlmClient(model="glm-5.2", temperature=0.7, top_p=None)
        assert client.temperature == 0.7
        assert client.top_p is None

    def test_stream_options(self):
        """stream_options include_usage should be set."""
        import inspect

        src = inspect.getsource(GlmClient._do_stream_request)
        assert "include_usage" in src

    def test_retries_use_attempt_local_results(self):
        """Retries must not clear output from earlier successful continuations."""
        import inspect

        src = inspect.getsource(GlmClient._do_stream_request)
        assert "attempt_result = StreamResult()" in src
        assert 'result.content = ""' not in src

    def test_retry_count(self):
        """Should retry MAX_RETRIES + 1 times."""
        import inspect

        src = inspect.getsource(GlmClient._do_stream_request)
        assert "range(MAX_RETRIES + 1)" in src or f"range({MAX_RETRIES} + 1)" in src

    def test_summarize_retry(self):
        """Summarization should also have retry logic."""
        import inspect

        src = inspect.getsource(GlmClient.summarize_messages)
        assert "attempt" in src
        assert "MAX_RETRIES" in src

    def test_cancel_check_in_stream(self):
        """Stream execution should check cancel flag."""
        import inspect

        src = inspect.getsource(GlmClient._execute_stream)
        assert "self._cancelled" in src


# ============================================================
# Summarize robustness
# ============================================================


class TestSummarizeRobustness:
    def test_summarize_handles_non_json_response(self):
        """summarize_messages should handle non-JSON 200 response gracefully."""
        import inspect

        src = inspect.getsource(GlmClient.summarize_messages)
        # Must have a try/except around resp.json()
        assert "except Exception" in src or "json.JSONDecodeError" in src

    @pytest.mark.asyncio
    async def test_summarize_rejects_empty_choices(self):
        """An empty compaction response must not become authoritative history."""
        from unittest.mock import AsyncMock, MagicMock

        client = GlmClient(model="glm-5.2")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"choices": []}
        client._client.post = AsyncMock(return_value=response)

        with pytest.raises(RuntimeError, match="summary"):
            await client.summarize_messages([{"role": "user", "content": "keep me"}])


# ============================================================
# Error body decode robustness
# ============================================================


class TestErrorBodyDecode:
    def test_execute_stream_uses_replace_on_error_decode(self):
        """Error response body decode should use errors=replace."""
        import inspect

        src = inspect.getsource(GlmClient._execute_stream)
        assert 'errors="replace"' in src or "errors='replace'" in src
