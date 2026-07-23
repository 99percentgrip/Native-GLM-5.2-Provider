"""Tests for glm_acp context compaction — automatic threshold, manual /compact,
error handling, tool-call boundary adjustment, and persistence."""

from unittest.mock import AsyncMock, MagicMock

import pytest

os = __import__("os")
os.environ.setdefault("ZAI_API_KEY", "test-key")

from glm_acp.agent import (
    _COMPACTION_MARKER_CLOSE,
    _COMPACTION_MARKER_OPEN,
    GlmAcpAgent,
    Session,
)
from glm_acp.config import (
    COMPACTION_KEEP_RECENT,
    CONTEXT_WINDOW_TOKENS,
)
from glm_acp.glm_client import GlmApiError


@pytest.fixture
def agent():
    a = GlmAcpAgent()
    mock_conn = MagicMock()
    mock_conn.session_update = AsyncMock()
    mock_conn.request_permission = AsyncMock()
    a._conn = mock_conn
    return a


@pytest.fixture
def session():
    return Session("test-session-id", ".")


def _make_session_with_messages(n: int) -> Session:
    """Create a session with system + n user/assistant message pairs."""
    s = Session("test-session-id", ".")
    for i in range(n):
        s.messages.append({"role": "user", "content": f"User message {i}"})
        s.messages.append({"role": "assistant", "content": f"Assistant reply {i}"})
    return s


def _mock_client(summary: str = "This is a summary of the conversation."):
    """Create a mock GlmClient with a working summarize_messages."""
    client = MagicMock()
    client.summarize_messages = AsyncMock(return_value=summary)
    client.aclose = AsyncMock()
    client.cancelled = False
    return client


# ============================================================
# Threshold guard: doesn't compact when below threshold
# ============================================================


class TestCompactionThreshold:
    @pytest.mark.asyncio
    async def test_no_compact_below_threshold(self, agent, session):
        """Session well below threshold should NOT be compacted."""
        session.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "reply"},
        ]
        client = _mock_client()
        await agent._maybe_compact(session, client, force=False)
        client.summarize_messages.assert_not_called()
        # Messages unchanged
        assert len(session.messages) == 3

    @pytest.mark.asyncio
    async def test_no_compact_too_few_messages(self, agent, session):
        """Even at high token estimate, too few messages shouldn't compact."""
        session.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 100_000},  # huge but few msgs
        ]
        client = _mock_client()
        # Force to bypass threshold check, but too few messages to compact
        await agent._maybe_compact(session, client, force=True)
        client.summarize_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_when_above_threshold(self, agent, session):
        """Session above threshold with enough messages SHOULD be compacted."""
        session = _make_session_with_messages(20)
        # Make each message very large to exceed threshold
        for msg in session.messages:
            msg["content"] = msg["content"] + " " * 100_000
        client = _mock_client()
        await agent._maybe_compact(session, client, force=False)
        client.summarize_messages.assert_called_once()


# ============================================================
# Force flag (used by /compact command)
# ============================================================


class TestCompactionForce:
    @pytest.mark.asyncio
    async def test_force_compacts_regardless_of_size(self, agent, session):
        """force=True should compact even with tiny messages."""
        session = _make_session_with_messages(20)
        client = _mock_client()
        await agent._maybe_compact(session, client, force=True)
        client.summarize_messages.assert_called_once()

    @pytest.mark.asyncio
    async def test_focused_compaction_passes_precompression_evidence(self, agent, session):
        session = _make_session_with_messages(10)
        session.plan = [
            {"content": "Verify the cleanup fix", "status": "in_progress", "priority": "high"}
        ]
        session.messages.insert(
            -5,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "edit-1",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": '{"path": "client.py"}',
                        },
                    }
                ],
            },
        )
        client = _mock_client()

        await agent._maybe_compact(
            session,
            client,
            force=True,
            focus="Preserve the async cleanup contract",
        )

        kwargs = client.summarize_messages.call_args.kwargs
        assert kwargs["focus"] == "Preserve the async cleanup contract"
        assert "Verify the cleanup fix" in kwargs["preserved_context"]
        assert "client.py" in kwargs["preserved_context"]

    @pytest.mark.asyncio
    async def test_compaction_falls_back_when_auxiliary_context_is_too_small(
        self, agent, monkeypatch
    ):
        session = _make_session_with_messages(20)
        for message in session.messages:
            message["content"] += "x" * 2000
        auxiliary = _mock_client()
        auxiliary.model = "tiny-auxiliary"
        main = _mock_client()
        main.model = session.model
        session.client = main
        session.client_key = ("existing",)
        monkeypatch.setitem(CONTEXT_WINDOW_TOKENS, "tiny-auxiliary", 100)
        monkeypatch.setattr(agent, "_client_for_session", lambda _session: main)

        await agent._maybe_compact(session, auxiliary, force=True)

        auxiliary.summarize_messages.assert_not_called()
        main.summarize_messages.assert_called_once()

    def test_precompaction_extracts_decisions_fixes_pending_and_learning(self, session):
        session.plan = [{"content": "Publish release", "status": "pending"}]
        messages = [
            {
                "role": "assistant",
                "content": (
                    "Decision: keep the public API stable.\n"
                    "Fixed: close resources from finally.\n"
                    "Remaining: run the package test."
                ),
                "tool_calls": [
                    {
                        "id": "run-1",
                        "function": {
                            "name": "run_command",
                            "arguments": '{"command":"pytest -q"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "run-1",
                "content": "exit code: 0\n12 passed",
            },
        ]

        evidence = GlmAcpAgent._pre_compaction_evidence(session, messages)

        assert "Decisions:" in evidence
        assert "Fixes and root causes:" in evidence
        assert "Unresolved and pending work:" in evidence
        assert "Proposed durable learning" in evidence
        assert session.compaction_learning_proposals == [
            "Decision: keep the public API stable.",
            "Fixed: close resources from finally.",
        ]

    @pytest.mark.asyncio
    async def test_successive_compaction_detects_quality_decline(self, agent):
        session = _make_session_with_messages(20)
        session.plan = [{"content": "Finish verification", "status": "pending"}]
        strong = _mock_client(summary=("Plan pending and verification details preserved. " * 30))
        await agent._maybe_compact(session, strong, force=True)

        session.messages = _make_session_with_messages(20).messages
        weak = _mock_client(summary="Too short.")
        await agent._maybe_compact(session, weak, force=True)

        assert len(session.compaction_quality_history) == 2
        assert session.compaction_quality_history[-1]["declined"] is True
        assert "<retained_evidence>" in session.messages[1]["content"]


class TestContextPressure:
    @pytest.mark.asyncio
    async def test_pressure_tiers_are_reported_once(self, agent, session):
        session.context_size = 100
        for used in (60, 75, 85, 90):
            session.estimated_tokens = used
            await agent._report_context_pressure(session)

        assert session.context_pressure_level == 3
        assert agent._conn.session_update.await_count == 3


# ============================================================
# Message partitioning and reconstruction
# ============================================================


class TestCompactionPartition:
    @pytest.mark.asyncio
    async def test_system_prompt_preserved(self, agent, session):
        """The system prompt must always be the first message after compaction."""
        session = _make_session_with_messages(20)
        original_system = session.messages[0]
        client = _mock_client()
        await agent._maybe_compact(session, client, force=True)
        assert session.messages[0]["role"] == "system"
        assert session.messages[0]["content"] == original_system["content"]

    @pytest.mark.asyncio
    async def test_compaction_marker_wraps_summary(self, agent, session):
        """The summary must be wrapped in conversation_summary markers."""
        session = _make_session_with_messages(20)
        client = _mock_client(summary="My great summary.")
        await agent._maybe_compact(session, client, force=True)
        # Second message should be the summary block
        summary_msg = session.messages[1]
        assert summary_msg["role"] == "user"
        assert _COMPACTION_MARKER_OPEN in summary_msg["content"]
        assert "My great summary." in summary_msg["content"]
        assert _COMPACTION_MARKER_CLOSE in summary_msg["content"]

    @pytest.mark.asyncio
    async def test_keep_recent_messages_preserved(self, agent, session):
        """The last N messages must be preserved verbatim."""
        session = _make_session_with_messages(20)
        client = _mock_client()
        await agent._maybe_compact(session, client, force=True)
        # System (1) + summary (1) + kept recent (COMPACTION_KEEP_RECENT)
        expected_len = 1 + 1 + COMPACTION_KEEP_RECENT
        assert len(session.messages) == expected_len

        # Check that the kept messages are the actual last N non-system msgs
        kept = session.messages[2:]  # after system + summary
        assert len(kept) == COMPACTION_KEEP_RECENT

    @pytest.mark.asyncio
    async def test_message_count_reduced(self, agent, session):
        """Compaction must reduce total message count."""
        session = _make_session_with_messages(20)
        original_count = len(session.messages)
        client = _mock_client()
        await agent._maybe_compact(session, client, force=True)
        assert len(session.messages) < original_count


# ============================================================
# Tool-call boundary adjustment
# ============================================================


class TestCompactionToolBoundary:
    @pytest.mark.asyncio
    async def test_tool_result_moved_to_summarize(self, agent, session):
        """If first kept message is a tool result, move it to summarize
        so we don't orphan it from its tool_call."""
        session = _make_session_with_messages(10)
        # Insert a tool result as the first of the "keep_recent" slice
        # We need to craft messages so that the boundary falls on a tool result
        messages = [{"role": "system", "content": "sys"}]
        # Many messages to summarize
        for i in range(20):
            messages.append({"role": "user", "content": f"msg {i}"})
            messages.append({"role": "assistant", "content": f"reply {i}"})
        # Now add assistant tool_call + tool result pairs as the "recent" messages
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": "tc1", "content": "file contents"})
        messages.append({"role": "assistant", "content": "Done."})
        messages.append({"role": "user", "content": "next question"})
        session.messages = messages

        client = _mock_client()
        await agent._maybe_compact(session, client, force=True)

        # Find what was passed to summarize_messages
        # The first kept message should NOT be a tool result
        kept = session.messages[2:]  # after system + summary
        assert kept[0].get("role") != "tool", (
            "First kept message should not be an orphaned tool result"
        )

    @pytest.mark.asyncio
    async def test_no_tool_result_orphaned(self, agent, session):
        """Verify no tool result appears without a preceding tool_call."""
        session = _make_session_with_messages(10)
        # Add tool calls near the boundary
        messages = [{"role": "system", "content": "sys"}]
        for i in range(15):
            messages.append({"role": "user", "content": f"msg {i}"})
            messages.append({"role": "assistant", "content": f"reply {i}"})
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": "tc1", "content": "result"})
        messages.append({"role": "assistant", "content": "Based on that..."})
        messages.append({"role": "user", "content": "continue"})
        session.messages = messages

        client = _mock_client()
        await agent._maybe_compact(session, client, force=True)

        # Verify no orphaned tool results in kept messages
        kept = session.messages[2:]
        for i, msg in enumerate(kept):
            if msg.get("role") == "tool":
                # Must have a preceding assistant with tool_calls in kept messages
                has_tool_call_before = any(
                    m.get("role") == "assistant" and m.get("tool_calls") for m in kept[:i]
                )
                assert has_tool_call_before, (
                    "Tool result found without preceding tool_call in kept messages"
                )


# ============================================================
# Token estimate updates after compaction
# ============================================================


class TestCompactionTokens:
    @pytest.mark.asyncio
    async def test_token_estimate_reduced(self, agent, session):
        """After compaction, estimated_tokens should be recalculated."""
        session = _make_session_with_messages(20)
        for msg in session.messages:
            msg["content"] = msg["content"] + " " * 10_000
        client = _mock_client()
        await agent._maybe_compact(session, client, force=True)
        # Should be much smaller now
        assert session.estimated_tokens > 0  # has some tokens
        assert session.estimated_tokens < 50_000  # but not huge

    @pytest.mark.asyncio
    async def test_force_re_report(self, agent, session):
        """last_reported_tokens should be reset and re-reported after compaction.

        _maybe_compact sets it to -1 to force _report_usage to fire, which
        then updates it to the new (lower) estimated token count.
        """
        session = _make_session_with_messages(20)
        session.last_reported_tokens = 999
        client = _mock_client()
        await agent._maybe_compact(session, client, force=True)
        # Should have been updated from -1 to the new estimated value
        assert session.last_reported_tokens != 999
        assert session.last_reported_tokens == session.estimated_tokens


# ============================================================
# Error handling
# ============================================================


class TestCompactionErrors:
    @pytest.mark.asyncio
    async def test_empty_summary_leaves_history_unchanged(self, agent, session):
        """Compaction commits only a non-empty validated summary."""
        session = _make_session_with_messages(20)
        original = list(session.messages)
        client = _mock_client()
        client.summarize_messages = AsyncMock(return_value="")

        with pytest.raises(RuntimeError, match="summary"):
            await agent._maybe_compact(session, client, force=True)

        assert session.messages == original

    @pytest.mark.asyncio
    async def test_api_error_in_auto_compact_propagates(self, agent, session):
        """In the _run_turn path, summarize failure should raise (caught by prompt)."""
        session = _make_session_with_messages(20)
        client = _mock_client()
        client.summarize_messages = AsyncMock(side_effect=GlmApiError(429, "rate limited"))
        with pytest.raises(GlmApiError):
            await agent._maybe_compact(session, client, force=True)

    @pytest.mark.asyncio
    async def test_compact_command_handles_error(self, agent, session):
        """The /compact slash command should catch errors and return a message."""
        session = _make_session_with_messages(20)
        result = await agent._handle_command(session, "/compact")
        # summarize_messages will be called on the real API with a mock key.
        # It should either succeed (unlikely with test key) or fail gracefully.
        # Either way, result should not be None and should be a string.
        assert isinstance(result, str)
        # If it failed, the user should see an error message
        if "failed" in result.lower():
            assert "unchanged" in result.lower()
            # Messages should be unchanged
            # (at least system prompt is still there)
            assert session.messages[0]["role"] == "system"


# ============================================================
# Summarization prompt content
# ============================================================


class TestSummarizeTranscript:
    def test_tool_results_keep_call_identity(self):
        from glm_acp.glm_client import GlmClient

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "contents"},
        ]

        transcript = GlmClient._format_transcript(messages)
        assert "call_a" in transcript
        assert "read_file" in transcript

    def test_format_transcript_user_message(self):
        from glm_acp.glm_client import GlmClient

        messages = [
            {"role": "user", "content": "hello world"},
        ]
        transcript = GlmClient._format_transcript(messages)
        assert "[User] hello world" in transcript

    def test_format_transcript_assistant_message(self):
        from glm_acp.glm_client import GlmClient

        messages = [
            {"role": "assistant", "content": "hi there"},
        ]
        transcript = GlmClient._format_transcript(messages)
        assert "[Assistant] hi there" in transcript

    def test_format_transcript_tool_call(self):
        from glm_acp.glm_client import GlmClient

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }
                ],
            },
        ]
        transcript = GlmClient._format_transcript(messages)
        assert "read_file" in transcript

    def test_format_transcript_tool_result(self):
        from glm_acp.glm_client import GlmClient

        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "file contents here"},
        ]
        transcript = GlmClient._format_transcript(messages)
        assert "[Tool Result tc1 — unknown tool] file contents here" in transcript

    def test_format_transcript_skips_system(self):
        from glm_acp.glm_client import GlmClient

        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ]
        transcript = GlmClient._format_transcript(messages)
        assert "system prompt" not in transcript
        assert "hello" in transcript

    def test_format_transcript_empty(self):
        from glm_acp.glm_client import GlmClient

        transcript = GlmClient._format_transcript([])
        assert transcript == ""

    def test_format_transcript_none_content(self):
        """Assistant message with content=None (tool_calls only)."""
        from glm_acp.glm_client import GlmClient

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            },
        ]
        transcript = GlmClient._format_transcript(messages)
        assert "write_file" in transcript
        assert "None" not in transcript  # None should not appear as text

    def test_format_transcript_list_content(self):
        """Vision message with list content (multipart blocks)."""
        from glm_acp.glm_client import GlmClient

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]
        transcript = GlmClient._format_transcript(messages)
        assert "What is this image?" in transcript
        assert "image_url" not in transcript  # image data not in transcript

    def test_format_transcript_integer_content(self):
        """Non-string, non-list content should be coerced gracefully."""
        from glm_acp.glm_client import GlmClient

        messages = [
            {"role": "user", "content": 12345},
        ]
        transcript = GlmClient._format_transcript(messages)
        assert "12345" in transcript
