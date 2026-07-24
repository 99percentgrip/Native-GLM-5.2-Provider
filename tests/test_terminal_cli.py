"""Standalone terminal frontend parity and safety tests."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import acp
import pytest
from acp.schema import PermissionOption

from glm_acp.cli import build_parser
from glm_acp.terminal_cli import TerminalClient, _configure


def test_chat_parser_exposes_full_session_configuration(tmp_path):
    args = build_parser().parse_args(
        [
            "chat",
            "--cwd",
            str(tmp_path),
            "--model",
            "glm-5.2",
            "--thought-level",
            "max",
            "--api-endpoint",
            "coding",
            "--permission",
            "bypass",
            "--generation-profile",
            "precise",
            "--auxiliary-model",
            "glm-4.7",
            "--mixture-mode",
            "enabled",
            "--mode",
            "code",
        ]
    )
    assert args.command == "chat"
    assert args.model == "glm-5.2"
    assert args.permission == "bypass"
    assert args.mode == "code"


@pytest.mark.asyncio
async def test_terminal_client_streams_agent_text_and_hides_thinking(capsys):
    client = TerminalClient(show_thinking=False, interactive=False)
    await client.session_update("session", acp.update_agent_thought_text("private"))
    await client.session_update("session", acp.update_agent_message_text("hello"))
    assert client.finish_turn() == "hello"
    output = capsys.readouterr()
    assert "hello" in output.out
    assert "private" not in output.out + output.err


@pytest.mark.asyncio
async def test_noninteractive_permission_fails_closed():
    client = TerminalClient(interactive=False)
    response = await client.request_permission(
        options=[PermissionOption(option_id="allow", kind="allow_once", name="Allow")],
        session_id="session",
        tool_call=SimpleNamespace(title="write file"),
    )
    assert response.outcome.outcome == "cancelled"


def test_permission_details_are_bounded_and_credential_redacted():
    detail = TerminalClient._permission_detail(
        {
            "command": "curl -H 'Authorization: Bearer very-secret-token-value' /api",
            "content": "x" * 10_000,
            "api_key": "must-never-appear",
        }
    )
    assert "must-never-appear" not in detail
    assert "very-secret-token-value" not in detail
    assert "[10000 characters]" in detail


@pytest.mark.asyncio
async def test_terminal_configuration_uses_agent_session_methods():
    calls = []

    class AgentStub:
        async def set_config_option(self, **kwargs):
            calls.append(("config", kwargs["config_id"], kwargs["value"]))

        async def set_session_mode(self, **kwargs):
            calls.append(("mode", kwargs["mode_id"]))

    args = argparse.Namespace(
        model="glm-5.2",
        thought_level="high",
        api_endpoint="coding",
        permission="read",
        generation_profile="balanced",
        auxiliary_model="glm-4.7",
        mixture_mode="enabled",
        mode="ask",
    )
    await _configure(AgentStub(), "session", args)
    assert ("config", "permission_mode", "read") in calls
    assert ("config", "mixture_mode", "enabled") in calls
    assert ("mode", "ask") in calls


@pytest.mark.asyncio
async def test_handle_plain_command_exit_and_quit_break(capsys):
    """``/exit`` and ``/quit`` ask the caller to break the loop."""
    from glm_acp.terminal_cli import _handle_plain_command

    pending: list[str] = []
    for cmd in ("/exit", "/quit", "  /exit  "):
        decision = await _handle_plain_command(
            cmd, agent=None, session_id="s", session=None, pending_images=pending
        )
        assert decision == "break"


@pytest.mark.asyncio
async def test_handle_plain_command_image_queues_and_skips(capsys):
    """``/image <path>`` queues the image and signals skip (don't send to model)."""
    from glm_acp.terminal_cli import _handle_plain_command

    pending: list[str] = []
    decision = await _handle_plain_command(
        "/image /tmp/foo.png",
        agent=None,
        session_id="s",
        session=None,
        pending_images=pending,
    )
    assert decision == "skip"
    assert pending == ["/tmp/foo.png"]


@pytest.mark.asyncio
async def test_handle_plain_command_help_prints_and_skips(capsys):
    """``/help`` prints the available commands and signals skip."""
    from glm_acp.terminal_cli import _handle_plain_command

    decision = await _handle_plain_command(
        "/help", agent=None, session_id="s", session=None, pending_images=[]
    )
    assert decision == "skip"
    captured = capsys.readouterr()
    assert "/max-iterations" in captured.err
    assert "/planmode" in captured.err


@pytest.mark.asyncio
async def test_handle_plain_command_max_iterations_no_arg_shows_current(capsys):
    """``/max-iterations`` with no arg shows the current cap and signals skip."""
    from glm_acp.terminal_cli import _handle_plain_command

    session = SimpleNamespace(max_tool_iterations=50)
    decision = await _handle_plain_command(
        "/max-iterations",
        agent=None,
        session_id="s",
        session=session,
        pending_images=[],
    )
    assert decision == "skip"
    captured = capsys.readouterr()
    assert "50" in captured.err


@pytest.mark.asyncio
async def test_handle_plain_command_max_iterations_with_arg_calls_set_config_option():
    """``/max-iterations 200`` routes through agent.set_config_option."""
    from glm_acp.terminal_cli import _handle_plain_command

    captured_calls = []

    class AgentStub:
        async def set_config_option(self, config_id, session_id, value):
            captured_calls.append((config_id, session_id, value))

    session = SimpleNamespace(max_tool_iterations=50)
    decision = await _handle_plain_command(
        "/max-iterations 200",
        agent=AgentStub(),
        session_id="test-session",
        session=session,
        pending_images=[],
    )
    assert decision == "skip"
    # set_config_option signature is (config_id, session_id, value).
    assert captured_calls == [("max_tool_iterations", "test-session", "200")]


@pytest.mark.asyncio
async def test_handle_plain_command_max_iterations_invalid_value_skips(capsys):
    """``/max-iterations abc`` is rejected without calling the agent."""
    from glm_acp.terminal_cli import _handle_plain_command

    class FailingStub:
        async def set_config_option(self, *args, **kwargs):
            raise AssertionError("set_config_option must not be called for invalid input")

    decision = await _handle_plain_command(
        "/max-iterations abc",
        agent=FailingStub(),
        session_id="s",
        session=SimpleNamespace(max_tool_iterations=50),
        pending_images=[],
    )
    assert decision == "skip"
    captured = capsys.readouterr()
    assert "Invalid value" in captured.err


@pytest.mark.asyncio
async def test_handle_plain_command_planmode_returns_prd_as_text():
    """``/planmode <PRD>`` activates plan mode and returns the PRD to be sent."""
    from glm_acp.terminal_cli import _handle_plain_command

    captured = []

    class AgentStub:
        async def set_session_mode(self, mode_id, session_id):
            captured.append((mode_id, session_id))

    decision = await _handle_plain_command(
        "/planmode build a todo app in React",
        agent=AgentStub(),
        session_id="test-session",
        session=None,
        pending_images=[],
    )
    # The decision is the rewritten text (the PRD) that gets submitted next.
    assert decision == "build a todo app in React"
    assert captured == [("plan", "test-session")]


@pytest.mark.asyncio
async def test_handle_plain_command_planmode_empty_prd_falls_through():
    """``/planmode`` with no argument is not transformed — falls through to model."""
    from glm_acp.terminal_cli import _handle_plain_command

    class FailingStub:
        async def set_session_mode(self, *args, **kwargs):
            raise AssertionError("set_session_mode must not be called for empty PRD")

    decision = await _handle_plain_command(
        "/planmode ",
        agent=FailingStub(),
        session_id="s",
        session=None,
        pending_images=[],
    )
    # No transformation happened; the caller will submit "/planmode " as text.
    assert decision is None


@pytest.mark.asyncio
async def test_handle_plain_command_non_command_returns_none():
    """Plain text (not a slash command) returns None and gets sent to the model."""
    from glm_acp.terminal_cli import _handle_plain_command

    decision = await _handle_plain_command(
        "hello world",
        agent=None,
        session_id="s",
        session=None,
        pending_images=[],
    )
    assert decision is None


def test_suggest_command_catches_missing_slash(capsys):
    """Typing a known command without the leading slash gets a hint, not a silent chat.

    Regression for: ``max-iterations 200`` (no slash) was silently sent to
    the model as a prompt. The user expected the slash command to fire.
    """
    from glm_acp.terminal_cli import _suggest_command_if_close

    # Without slash, with arg → suggest the slash version
    assert _suggest_command_if_close("max-iterations 200") is not None
    assert "/max-iterations" in _suggest_command_if_close("max-iterations 200")
    # Without slash, no arg → also suggested
    assert "/help" in _suggest_command_if_close("help")
    assert "/planmode" in _suggest_command_if_close("planmode build an app")
    assert "/image" in _suggest_command_if_close("image /tmp/foo.png")
    assert "/exit" in _suggest_command_if_close("exit")
    assert "/quit" in _suggest_command_if_close("quit")

    # Case-insensitive
    assert "/max-iterations" in _suggest_command_if_close("MAX-ITERATIONS 200")
    assert "/help" in _suggest_command_if_close("HELP")

    # Properly formed commands (with slash) are NOT flagged
    assert _suggest_command_if_close("/max-iterations 200") is None
    assert _suggest_command_if_close("/help") is None
    assert _suggest_command_if_close("/exit") is None

    # Plain chat text is NOT flagged (would be sent to the model as-is)
    assert _suggest_command_if_close("hello world") is None
    assert _suggest_command_if_close("write me a Python function") is None
    assert _suggest_command_if_close("") is None


@pytest.mark.asyncio
async def test_handle_plain_command_typo_suggests_slash(capsys):
    """``max-iterations 200`` (no slash) prints a hint and skips model submission."""
    from glm_acp.terminal_cli import _handle_plain_command

    class FailingAgent:
        async def prompt(self, *args, **kwargs):
            raise AssertionError("must not be called for typo'd command")

    decision = await _handle_plain_command(
        "max-iterations 200",
        agent=FailingAgent(),
        session_id="s",
        session=None,
        pending_images=[],
    )
    assert decision == "skip"
    captured = capsys.readouterr()
    assert "/max-iterations" in captured.err
    assert "leading" in captured.err.lower()
