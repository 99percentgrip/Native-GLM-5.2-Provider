"""Tests for glm_acp.agent — session lifecycle, serialization, config, slash commands."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

os = __import__("os")
os.environ.setdefault("ZAI_API_KEY", "test-key")

from glm_acp.agent import GlmAcpAgent, Session, build_system_prompt
from glm_acp.config import (
    DEFAULT_MODEL,
    DEFAULT_API_ENDPOINT,
    CONTEXT_WINDOW_TOKENS,
    API_ENDPOINTS,
)


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


# ============================================================
# System Prompt
# ============================================================

class TestSystemPrompt:
    def test_contains_model_name(self):
        prompt = build_system_prompt(".", "glm-5.2")
        assert "GLM-5.2" in prompt

    def test_contains_project_context(self):
        prompt = build_system_prompt(".")
        assert "Python project" in prompt
        assert "git" in prompt

    def test_empty_dir(self):
        prompt = build_system_prompt("/tmp")
        assert "no project files" in prompt

    def test_contains_rules(self):
        prompt = build_system_prompt(".")
        assert "Read files before editing" in prompt
        assert "update_plan" in prompt


# ============================================================
# Session serialization
# ============================================================

class TestSessionSerialization:
    def test_to_dict_has_all_fields(self, session):
        d = session.to_dict()
        for field in ["cwd", "model", "thought_level", "mode", "api_endpoint",
                       "title", "permission_mode", "plan", "messages",
                       "total_input_tokens", "total_output_tokens"]:
            assert field in d, f"Missing field: {field}"

    def test_round_trip(self, session):
        session.model = "glm-4.7"
        session.api_endpoint = "standard"
        session.plan = [{"content": "task", "status": "pending", "priority": "high"}]
        session.total_input_tokens = 5000
        session.total_output_tokens = 2000

        d = session.to_dict()
        restored = Session.from_dict(d, "new-id")

        assert restored.model == "glm-4.7"
        assert restored.api_endpoint == "standard"
        assert restored.plan == session.plan
        assert restored.total_input_tokens == 5000
        assert restored.total_output_tokens == 2000

    def test_old_session_backward_compat(self):
        old_data = {"cwd": ".", "model": "glm-5.2", "messages": [], "mode": "code"}
        s = Session.from_dict(old_data, "old")
        assert s.plan == []
        assert s.api_endpoint == "coding"
        assert s.permission_mode == "ask"
        assert s.total_input_tokens == 0
        assert s.total_output_tokens == 0


# ============================================================
# Token estimation
# ============================================================

class TestTokenEstimation:
    def test_basic_estimate(self, session):
        session.messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world"},
        ]
        tokens = GlmAcpAgent._estimate_tokens(session.messages)
        assert tokens > 0

    def test_includes_overhead(self):
        messages = [{"role": "user", "content": "a"}]
        tokens = GlmAcpAgent._estimate_tokens(messages)
        # At least 4 tokens of overhead
        assert tokens >= 4

    def test_handles_list_content(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
        ]
        tokens = GlmAcpAgent._estimate_tokens(messages)
        assert tokens > 0

    def test_empty_messages(self):
        assert GlmAcpAgent._estimate_tokens([]) == 0


# ============================================================
# Config option building
# ============================================================

class TestConfigOptions:
    def test_model_option_coding(self, agent, session):
        opt = agent._build_model_option(session)
        assert opt.id == "model"
        assert len(opt.options) == 3  # coding plan = 3 models

    def test_model_option_standard(self, agent, session):
        session.api_endpoint = "standard"
        opt = agent._build_model_option(session)
        assert len(opt.options) == 5  # + vision models

    def test_thought_option_vision(self, agent, session):
        session.model = "glm-4.5v"
        opt = agent._build_thought_option(session)
        assert len(opt.options) == 1  # disabled only

    def test_all_options_present(self, agent, session):
        opts = [
            agent._build_model_option(session),
            agent._build_thought_option(session),
            agent._build_api_endpoint_option(session),
            agent._build_permission_option(session),
        ]
        ids = [o.id for o in opts]
        assert set(ids) == {"model", "thought_level", "api_endpoint", "permission_mode"}


# ============================================================
# Config switching
# ============================================================

class TestConfigSwitch:
    @pytest.mark.asyncio
    async def test_model_switch(self, agent, session):
        agent._sessions[session.id] = session
        await agent.set_config_option("model", session.id, "glm-4.7")
        assert session.model == "glm-4.7"
        assert session.context_size == CONTEXT_WINDOW_TOKENS["glm-4.7"]

    @pytest.mark.asyncio
    async def test_endpoint_switch_fallback(self, agent, session):
        agent._sessions[session.id] = session
        session.api_endpoint = "standard"
        session.model = "glm-4.5v"
        await agent.set_config_option("api_endpoint", session.id, "coding")
        assert session.model == "glm-5.2"  # fell back


# ============================================================
# Slash commands
# ============================================================

class TestSlashCommands:
    @pytest.mark.asyncio
    async def test_status(self, agent, session):
        session.total_input_tokens = 1000
        result = await agent._handle_command(session, "/status")
        assert "Session Status" in result
        assert "1,000 input" in result

    @pytest.mark.asyncio
    async def test_clear_plan(self, agent, session):
        session.plan = [{"content": "x", "status": "pending", "priority": "high"}]
        result = await agent._handle_command(session, "/clear-plan")
        assert session.plan == []
        assert "cleared" in result.lower()

    @pytest.mark.asyncio
    async def test_clear_history(self, agent, session):
        session.messages.append({"role": "user", "content": "test"})
        session.total_input_tokens = 500
        result = await agent._handle_command(session, "/clear-history")
        assert len(session.messages) == 1  # system msg only
        assert session.total_input_tokens == 0
        assert "cleared" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_command(self, agent, session):
        result = await agent._handle_command(session, "/foobar")
        assert "Unknown" in result

    @pytest.mark.asyncio
    async def test_export(self, agent, session, tmp_path):
        session.cwd = str(tmp_path)
        session.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = await agent._handle_command(session, "/export")
        assert "exported" in result.lower()
        # Check file was created
        exports = list(tmp_path.glob("conversation_export_*.md"))
        assert len(exports) == 1
        content = exports[0].read_text()
        assert "hello" in content
        assert "hi there" in content

    @pytest.mark.asyncio
    async def test_diff(self, agent, session, tmp_path):
        session.cwd = str(tmp_path)
        result = await agent._handle_command(session, "/diff")
        assert "git" in result.lower() or "diff" in result.lower() or "no uncommitted" in result.lower()


# ============================================================
# Plan tool
# ============================================================

class TestPlanTool:
    @pytest.mark.asyncio
    async def test_plan_update(self, agent, session):
        args = {"tasks": [
            {"content": "Task 1", "status": "completed", "priority": "high"},
            {"content": "Task 2", "status": "in_progress", "priority": "medium"},
            {"content": "Task 3", "status": "pending", "priority": "low"},
        ]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert "3 tasks" in result
        assert len(session.plan) == 3
        assert session.plan[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_plan_empty(self, agent, session):
        result = await agent._handle_update_plan(session, "tc1", {"tasks": []})
        assert "0 tasks" in result
        assert session.plan == []


# ============================================================
# Plan tool — edge cases and sanitization
# ============================================================

class TestPlanToolEdgeCases:
    @pytest.mark.asyncio
    async def test_invalid_status_normalized(self, agent, session):
        """Model sends 'done' instead of 'completed' — should be sanitized."""
        args = {"tasks": [
            {"content": "Task 1", "status": "done", "priority": "high"},
            {"content": "Task 2", "status": "in-progress", "priority": "low"},
            {"content": "Task 3", "status": "active", "priority": "medium"},
            {"content": "Task 4", "status": "todo", "priority": "medium"},
        ]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert session.plan[0]["status"] == "completed"
        assert session.plan[1]["status"] == "in_progress"
        assert session.plan[2]["status"] == "in_progress"
        assert session.plan[3]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_invalid_priority_normalized(self, agent, session):
        """Model sends 'urgent' instead of 'high' — should be sanitized."""
        args = {"tasks": [
            {"content": "Task 1", "status": "pending", "priority": "urgent"},
            {"content": "Task 2", "status": "pending", "priority": "critical"},
            {"content": "Task 3", "status": "pending", "priority": "normal"},
            {"content": "Task 4", "status": "pending", "priority": "bogus"},
        ]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert session.plan[0]["priority"] == "high"
        assert session.plan[1]["priority"] == "high"
        assert session.plan[2]["priority"] == "medium"
        assert session.plan[3]["priority"] == "medium"  # default fallback

    @pytest.mark.asyncio
    async def test_garbage_status_falls_back(self, agent, session):
        """Completely unrecognized status falls back to 'pending'."""
        args = {"tasks": [
            {"content": "Task", "status": "banana", "priority": "high"},
        ]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert session.plan[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_string_task(self, agent, session):
        """Model sends a bare string instead of a dict — should be handled."""
        args = {"tasks": ["Just a plain string task"]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert "1 tasks" in result
        assert session.plan[0]["content"] == "Just a plain string task"
        assert session.plan[0]["status"] == "pending"  # default
        assert session.plan[0]["priority"] == "medium"  # default

    @pytest.mark.asyncio
    async def test_non_dict_task_skipped(self, agent, session):
        """Non-dict, non-string entries are silently skipped."""
        args = {"tasks": [
            42,
            None,
            {"content": "valid", "status": "pending", "priority": "high"},
            ["nested", "list"],
        ]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert "1 tasks" in result
        assert len(session.plan) == 1
        assert session.plan[0]["content"] == "valid"

    @pytest.mark.asyncio
    async def test_missing_fields_defaulted(self, agent, session):
        """Task dict missing status/priority gets safe defaults."""
        args = {"tasks": [
            {"content": "just content"},
        ]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert session.plan[0]["status"] == "pending"
        assert session.plan[0]["priority"] == "medium"

    @pytest.mark.asyncio
    async def test_missing_tasks_key(self, agent, session):
        """args dict with no 'tasks' key — should produce empty plan."""
        result = await agent._handle_update_plan(session, "tc1", {})
        assert "0 tasks" in result
        assert session.plan == []

    @pytest.mark.asyncio
    async def test_content_coerced_to_string(self, agent, session):
        """Non-string content (e.g. int) should be coerced to str."""
        args = {"tasks": [
            {"content": 12345, "status": "pending", "priority": "high"},
        ]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert session.plan[0]["content"] == "12345"

    @pytest.mark.asyncio
    async def test_plan_summary_counts(self, agent, session):
        """The returned string should have correct counts."""
        args = {"tasks": [
            {"content": "a", "status": "completed", "priority": "high"},
            {"content": "b", "status": "completed", "priority": "high"},
            {"content": "c", "status": "in_progress", "priority": "high"},
            {"content": "d", "status": "in_progress", "priority": "high"},
            {"content": "e", "status": "pending", "priority": "high"},
            {"content": "f", "status": "pending", "priority": "high"},
        ]}
        result = await agent._handle_update_plan(session, "tc1", args)
        assert "2 completed" in result
        assert "2 in progress" in result
        assert "2 pending" in result

    @pytest.mark.asyncio
    async def test_plan_persisted_to_store(self, agent, session, tmp_path):
        """_handle_update_plan should save to session store."""
        agent._store = MagicMock()
        agent._store.save = MagicMock()
        args = {"tasks": [{"content": "task", "status": "pending", "priority": "high"}]}
        await agent._handle_update_plan(session, "tc1", args)
        assert agent._store.save.called


class TestPlanSanitizers:
    """Unit tests for _sanitize_status and _sanitize_priority."""

    def test_sanitize_status_synonyms(self):
        from glm_acp.agent import _sanitize_status
        assert _sanitize_status("done") == "completed"
        assert _sanitize_status("Finished") == "completed"
        assert _sanitize_status("COMPLETE") == "completed"
        assert _sanitize_status("in-progress") == "in_progress"
        assert _sanitize_status("active") == "in_progress"
        assert _sanitize_status("working") == "in_progress"
        assert _sanitize_status("todo") == "pending"
        assert _sanitize_status("not_started") == "pending"

    def test_sanitize_status_valid_passthrough(self):
        from glm_acp.agent import _sanitize_status
        assert _sanitize_status("pending") == "pending"
        assert _sanitize_status("in_progress") == "in_progress"
        assert _sanitize_status("completed") == "completed"

    def test_sanitize_status_unknown(self):
        from glm_acp.agent import _sanitize_status
        assert _sanitize_status("banana") == "pending"
        assert _sanitize_status(None) == "pending"
        assert _sanitize_status("") == "pending"
        assert _sanitize_status(123) == "pending"

    def test_sanitize_priority_synonyms(self):
        from glm_acp.agent import _sanitize_priority
        assert _sanitize_priority("urgent") == "high"
        assert _sanitize_priority("critical") == "high"
        assert _sanitize_priority("p0") == "high"
        assert _sanitize_priority("normal") == "medium"
        assert _sanitize_priority("default") == "medium"
        assert _sanitize_priority("minor") == "low"

    def test_sanitize_priority_valid_passthrough(self):
        from glm_acp.agent import _sanitize_priority
        assert _sanitize_priority("high") == "high"
        assert _sanitize_priority("medium") == "medium"
        assert _sanitize_priority("low") == "low"

    def test_sanitize_priority_unknown(self):
        from glm_acp.agent import _sanitize_priority
        assert _sanitize_priority("bogus") == "medium"
        assert _sanitize_priority(None) == "medium"
        assert _sanitize_priority("") == "medium"


# ============================================================
# Friendly errors
# ============================================================

class TestFriendlyErrors:
    def test_auth_error(self, agent, session):
        from glm_acp.glm_client import GlmApiError
        msg = agent._friendly_error(GlmApiError(401, "bad key"), session)
        assert "Authentication" in msg

    def test_rate_limit_error(self, agent, session):
        from glm_acp.glm_client import GlmApiError
        msg = agent._friendly_error(GlmApiError(429, "slow down"), session)
        assert "Rate limited" in msg

    def test_content_filter(self, agent, session):
        from glm_acp.glm_client import GlmApiError
        msg = agent._friendly_error(GlmApiError(1301, "filtered"), session)
        assert "Content filtered" in msg

    def test_plan_limitation(self, agent, session):
        from glm_acp.glm_client import GlmApiError
        msg = agent._friendly_error(GlmApiError(1311, "no access"), session)
        assert "Plan limitation" in msg

    def test_network_error(self, agent, session):
        msg = agent._friendly_error(RuntimeError("connection timeout"), session)
        assert "timed out" in msg.lower() or "network" in msg.lower()

    def test_api_key_missing(self, agent, session):
        msg = agent._friendly_error(RuntimeError("ZAI_API_KEY not set"), session)
        assert "API key" in msg


# ============================================================
# Initialize / capabilities
# ============================================================

class TestInitialize:
    @pytest.mark.asyncio
    async def test_capabilities(self, agent):
        resp = await agent.initialize(1)
        caps = resp.agent_capabilities
        assert caps.load_session is True
        assert caps.prompt_capabilities.image is True
        sc = caps.session_capabilities
        assert sc.list is not None
        assert sc.resume is not None
        assert sc.close is not None
        assert sc.fork is not None
        assert sc.additional_directories is not None

    @pytest.mark.asyncio
    async def test_agent_info(self, agent):
        resp = await agent.initialize(1)
        assert resp.agent_info.name == "glm-acp"
        assert resp.agent_info.title == "Z.ai GLM"


# ============================================================
# Fork session
# ============================================================

class TestFork:
    @pytest.mark.asyncio
    async def test_fork_copies_state(self, agent, session):
        agent._sessions[session.id] = session
        session.model = "glm-4.7"
        session.api_endpoint = "standard"
        session.plan = [{"content": "x", "status": "pending", "priority": "high"}]
        session.total_input_tokens = 3000
        session.messages.append({"role": "user", "content": "hello"})

        fork = await agent.fork_session(cwd=".", session_id=session.id)
        f = agent._sessions[fork.session_id]
        assert f.id != session.id
        assert f.model == "glm-4.7"
        assert f.api_endpoint == "standard"
        assert f.plan == session.plan
        assert f.total_input_tokens == 3000
        assert len(f.messages) == len(session.messages)


# ============================================================
# Tool titles
# ============================================================

class TestToolTitles:
    def test_all_tools_have_titles(self, agent):
        from glm_acp.tools import TOOL_DEFINITIONS
        for tool in TOOL_DEFINITIONS:
            name = tool["function"]["name"]
            title = agent._tool_title(name)
            assert title != name, f"{name} has no custom title"
