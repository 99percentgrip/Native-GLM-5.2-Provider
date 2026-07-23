"""Tests for glm_acp.glm_client — GlmApiError, StreamResult, cancel, retry logic."""

import os

import pytest

os.environ.setdefault("ZAI_API_KEY", "test-key")

from glm_acp.config import MAX_RETRIES
from glm_acp.glm_client import (
    GlmApiError,
    GlmClient,
    PlanQuota,
    PlanUsage,
    StreamResult,
    ToolCallAccumulator,
)


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


class TestPlanUsage:
    def test_sync_quota_query_uses_the_same_credential_safe_parser(self, monkeypatch):
        from unittest.mock import MagicMock

        client = GlmClient(
            model="glm-5.2",
            base_url="https://api.z.ai/api/coding/paas/v4",
        )
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "data": {
                "limits": [
                    {
                        "type": "TOKENS_LIMIT",
                        "unit": 3,
                        "percentage": 12,
                    }
                ]
            }
        }
        request = MagicMock(return_value=response)
        monkeypatch.setattr("glm_acp.glm_client.httpx.get", request)

        usage = client.query_plan_usage_sync()

        assert usage.quotas[0].percentage == 12
        assert request.call_args.kwargs["headers"]["Authorization"] == "test-key"
        assert "Bearer" not in request.call_args.kwargs["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_official_quota_response_is_normalized_without_bearer_prefix(self):
        from unittest.mock import AsyncMock, MagicMock

        client = GlmClient(
            model="glm-5.2",
            base_url="https://api.z.ai/api/coding/paas/v4",
        )
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "data": {
                "limits": [
                    {
                        "type": "TOKENS_LIMIT",
                        "unit": 3,
                        "number": 5,
                        "usage": 800_000_000,
                        "currentValue": 120_000_000,
                        "remaining": 680_000_000,
                        "percentage": 15,
                        "nextResetTime": 1_770_648_402_389,
                    },
                    {
                        "type": "TOKENS_LIMIT",
                        "unit": 6,
                        "percentage": 9,
                    },
                    {
                        "type": "TIME_LIMIT",
                        "usage": 1000,
                        "currentValue": 25,
                        "remaining": 975,
                        "percentage": 2.5,
                        "usageDetails": [
                            {"modelCode": "search-prime", "usage": 20},
                            {"modelCode": "web-reader", "usage": 5},
                        ],
                    },
                ]
            }
        }
        client._client.get = AsyncMock(return_value=response)

        usage = await client.query_plan_usage()

        assert usage == PlanUsage(
            platform="Z.ai",
            quotas=(
                PlanQuota(
                    kind="TOKENS_LIMIT",
                    unit=3,
                    number=5,
                    limit=800_000_000,
                    used=120_000_000,
                    remaining=680_000_000,
                    percentage=15.0,
                    next_reset_ms=1_770_648_402_389,
                ),
                PlanQuota(
                    kind="TOKENS_LIMIT",
                    unit=6,
                    number=None,
                    limit=None,
                    used=None,
                    remaining=None,
                    percentage=9.0,
                    next_reset_ms=None,
                ),
                PlanQuota(
                    kind="TIME_LIMIT",
                    unit=None,
                    number=None,
                    limit=1000,
                    used=25,
                    remaining=975,
                    percentage=2.5,
                    next_reset_ms=None,
                    usage_details=(("search-prime", 20), ("web-reader", 5)),
                ),
            ),
        )
        call = client._client.get.call_args
        assert call.args[0] == "https://api.z.ai/api/monitor/usage/quota/limit"
        assert call.kwargs["headers"]["Authorization"] == "test-key"
        assert "Bearer" not in call.kwargs["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_custom_endpoint_cannot_receive_usage_credentials(self):
        from unittest.mock import AsyncMock

        client = GlmClient(model="glm-5.2", base_url="https://proxy.example/v4")
        client._client.get = AsyncMock()

        with pytest.raises(GlmApiError, match="custom API endpoint"):
            await client.query_plan_usage()
        with pytest.raises(GlmApiError, match="custom API endpoint"):
            client.query_plan_usage_sync()

        client._client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_malformed_or_empty_quota_response_fails_closed(self):
        from unittest.mock import AsyncMock, MagicMock

        client = GlmClient(model="glm-5.2")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": {"limits": [{"type": "UNKNOWN"}]}}
        client._client.get = AsyncMock(return_value=response)

        with pytest.raises(GlmApiError, match="no supported quota"):
            await client.query_plan_usage()


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

    @pytest.mark.asyncio
    async def test_auxiliary_completion_is_bounded_and_returns_usage(self):
        from unittest.mock import AsyncMock, MagicMock

        client = GlmClient(model="glm-5.2")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Useful title"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        }
        client._client.post = AsyncMock(return_value=response)

        result = await client.complete_auxiliary("system", "user", max_tokens=50_000)

        assert result.content == "Useful title"
        assert result.usage["input_tokens"] == 12
        assert client._client.post.call_args.kwargs["json"]["max_tokens"] == 4096


# ============================================================
# Error body decode robustness
# ============================================================


class TestErrorBodyDecode:
    def test_execute_stream_uses_replace_on_error_decode(self):
        """Error response body decode should use errors=replace."""
        import inspect

        src = inspect.getsource(GlmClient._execute_stream)
        assert 'errors="replace"' in src or "errors='replace'" in src
