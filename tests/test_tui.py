"""Full-screen terminal frontend behavior and shared-runtime tests."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import acp
import pytest
from acp.helpers import update_available_commands
from acp.schema import AvailableCommand, PermissionOption
from textual import events
from textual._xterm_parser import XTermParser
from textual.containers import VerticalScroll
from textual.widgets import ContentSwitcher, Footer, Input, OptionList, Select, Static
from textual.widgets._footer import FooterKey

from glm_acp.cli import build_parser
from glm_acp.glm_client import PlanQuota, PlanUsage
from glm_acp.terminal_cli import run_chat_command
from glm_acp.tui import (
    CONFIG_COMMANDS,
    NativeGlmTui,
    PermissionScreen,
    SettingsScreen,
    _read_system_clipboard,
)


class FakeAgent:
    def __init__(self) -> None:
        self._sessions = {}
        self.conn = None
        self.prompts = []
        self.permission = None
        self.closed = False
        self.config_calls = []
        self.mode_calls = []
        self.usage_calls = 0

    def on_connect(self, conn) -> None:
        self.conn = conn

    async def initialize(self, **kwargs):
        return SimpleNamespace()

    async def new_session(self, cwd, additional_directories=None, **kwargs):
        await self.conn.session_update(
            "tui-session",
            update_available_commands(
                [
                    AvailableCommand(name="help", description="Show harness commands"),
                    AvailableCommand(name="status", description="Show session status"),
                    AvailableCommand(name="checkpoint", description="Manage checkpoints"),
                ]
            ),
        )
        session = SimpleNamespace(
            model="glm-5.2",
            permission_mode="ask",
            mode="code",
            api_endpoint="coding",
            thought_level="enabled",
            generation_profile="balanced",
            auxiliary_model="main",
            mixture_mode="off",
        )
        self._sessions["tui-session"] = session
        return SimpleNamespace(session_id="tui-session")

    async def set_config_option(self, config_id, session_id, value, **kwargs):
        self.config_calls.append((config_id, value))
        session = self._sessions[session_id]
        if config_id == "permission_mode":
            session.permission_mode = value
        elif config_id == "model":
            session.model = value
        else:
            setattr(session, config_id, value)

    async def set_session_mode(self, mode_id, session_id, **kwargs):
        self.mode_calls.append(mode_id)
        self._sessions[session_id].mode = mode_id

    async def query_provider_usage(self, session_id):
        self.usage_calls += 1
        return PlanUsage(
            platform="Z.ai",
            quotas=(
                PlanQuota(
                    kind="TOKENS_LIMIT",
                    unit=3,
                    number=5,
                    limit=1000,
                    used=120,
                    remaining=880,
                    percentage=12,
                    next_reset_ms=None,
                ),
                PlanQuota(
                    kind="TOKENS_LIMIT",
                    unit=6,
                    number=7,
                    limit=None,
                    used=None,
                    remaining=None,
                    percentage=4,
                    next_reset_ms=None,
                ),
                PlanQuota(
                    kind="TIME_LIMIT",
                    unit=5,
                    number=1,
                    limit=100,
                    used=2,
                    remaining=98,
                    percentage=2,
                    next_reset_ms=None,
                ),
            ),
        )

    @staticmethod
    def format_provider_usage(usage):
        return "5-hour model quota: 12% used\nWeekly model quota: 4% used"

    async def prompt(self, prompt, session_id, message_id=None, **kwargs):
        self.prompts.append(prompt)
        text = str(getattr(prompt[0], "text", ""))
        if text.startswith("/"):
            await self.conn.session_update(
                session_id,
                acp.update_agent_message_text(f"Handled {text}"),
            )
            return
        tool_id = "tool-1"
        await self.conn.session_update(
            session_id,
            acp.start_tool_call(tool_id, "Write file", kind="edit", status="pending"),
        )
        self.permission = await self.conn.request_permission(
            options=[
                PermissionOption(option_id="allow", kind="allow_once", name="Allow write_file"),
                PermissionOption(option_id="reject", kind="reject_once", name="Deny"),
            ],
            session_id=session_id,
            tool_call=acp.update_tool_call(
                tool_id,
                status="pending",
                raw_input={"path": "demo.txt", "api_key": "must-never-render"},
            ),
        )
        await self.conn.session_update(
            session_id,
            acp.update_agent_message_text("Permission handled."),
        )

    async def cancel(self, **kwargs):
        return None

    async def aclose(self):
        self.closed = True


class HangingCloseAgent(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = False

    async def aclose(self):
        self.close_started = True
        await asyncio.Event().wait()


class SlowAgent(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_started = asyncio.Event()
        self.prompt_release = asyncio.Event()

    async def prompt(self, prompt, session_id, message_id=None, **kwargs):
        self.prompts.append(prompt)
        self.prompt_started.set()
        await self.prompt_release.wait()
        await self.conn.session_update(
            session_id,
            acp.update_agent_message_text("Finished."),
        )


def _args(tmp_path, *extra):
    return build_parser().parse_args(
        ["chat", "--cwd", str(tmp_path), "--permission", "ask", *extra]
    )


async def _wait_for_agent_ready(app, pilot) -> None:
    for _ in range(40):
        await pilot.pause(0.05)
        if app._agent_ready:
            return
    raise AssertionError("TUI agent initialization did not complete")


@pytest.mark.asyncio
async def test_tui_mounts_full_screen_panels_and_toggles_thinking(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        assert app._agent_ready is True
        assert app.query_one("#composer", Input).disabled is False
        assert "tui-sess…" in str(app.query_one("#session", Static).render())
        for _ in range(20):
            await pilot.pause(0.05)
            if "quota 5h 12% · week 4% · MCP 2%" in str(app.query_one("#session", Static).render()):
                break
        assert "quota 5h 12% · week 4% · MCP 2%" in str(app.query_one("#session", Static).render())
        assert app.query_one("#thinking").has_class("hidden")
        await pilot.press("f2")
        await pilot.pause()
        assert not app.query_one("#thinking").has_class("hidden")
        app.exit(0)

    assert agent.closed is True


@pytest.mark.asyncio
async def test_tui_multiline_paste_is_retained_and_composer_does_not_overlap_footer(
    tmp_path,
):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)
        footer = app.query_one(Footer)

        terminal_bytes = "\x1b[200~\nPlease inspect this pasted prompt.\nKeep its content.\x1b[201~"
        messages = list(XTermParser().feed(terminal_bytes))
        assert len(messages) == 1
        assert isinstance(messages[0], events.Paste)
        assert app._driver is not None
        app._driver.send_message(messages[0])
        await pilot.pause()

        assert composer.value == "Please inspect this pasted prompt. Keep its content."
        assert composer.region.bottom <= footer.region.y
        app.exit(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("shortcut", ["ctrl+v", "ctrl+shift+v"])
async def test_tui_clipboard_shortcuts_read_external_system_clipboard(
    tmp_path, monkeypatch, shortcut
):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)
    monkeypatch.setattr(
        "glm_acp.tui._read_system_clipboard",
        lambda: "\nCopied outside the TUI.\nRetain this prompt.",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)
        assert app.clipboard == ""

        await pilot.press(shortcut)

        assert composer.value == "Copied outside the TUI. Retain this prompt."
        app.exit(0)


def test_system_clipboard_reader_does_not_inherit_credentials(monkeypatch):
    captured = {}
    monkeypatch.setenv("ZAI_API_KEY", "must-not-reach-clipboard-process")
    monkeypatch.setattr("glm_acp.tui.shutil.which", lambda name: f"/safe/{name}")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        captured["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout="external clipboard text")

    monkeypatch.setattr("glm_acp.tui.subprocess.run", fake_run)

    assert _read_system_clipboard() == "external clipboard text"
    assert captured["command"]
    assert captured["timeout"] == 1.0
    assert "ZAI_API_KEY" not in captured["environment"]


@pytest.mark.asyncio
async def test_tui_context_menu_opens_via_keyboard_and_lists_composer_actions(tmp_path):
    """F6 / Ctrl+Right Click should open a Codex-style dropdown menu."""
    from glm_acp.tui import ContextMenuScreen

    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)
        composer.focus()
        await pilot.pause()

        # Keyboard fallback (F6). Ctrl+M is Enter in terminals.
        await pilot.press("f6")
        await pilot.pause()
        modal = [s for s in app.screen_stack if isinstance(s, ContextMenuScreen)]
        assert len(modal) == 1, "F6 should open the context menu"
        option_list = modal[0].query_one("#ctx-list", OptionList)
        labels = [opt.prompt for opt in option_list._options]
        # Composer-focused menu offers Cut / Copy / Paste / Select All.
        assert any("Cut" in lab for lab in labels)
        assert any("Copy" in lab and "last" not in lab.lower() for lab in labels)
        assert any("Paste" in lab for lab in labels)
        assert any("Select all" in lab for lab in labels)
        # Close the menu.
        await pilot.press("escape")
        await pilot.pause()
        modal = [s for s in app.screen_stack if isinstance(s, ContextMenuScreen)]
        assert len(modal) == 0
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_context_menu_opens_on_ctrl_right_click(tmp_path):
    """Ctrl+Right Click (button==2, ctrl=True) opens the context menu.

    ``pilot.click`` only simulates left clicks, so we synthesize a ``Click``
    event with ``button == 2`` and ``ctrl == True`` to mirror SGR-encoded
    right-clicks delivered by real terminals in mouse mode.
    """
    from glm_acp.tui import ContextMenuScreen

    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)

        # Plain left click must not open the menu.
        plain_click = events.Click(
            widget=None,
            x=2,
            y=2,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
        )
        app.on_click(plain_click)
        await pilot.pause()
        modal = [s for s in app.screen_stack if isinstance(s, ContextMenuScreen)]
        assert len(modal) == 0, "plain left-click should NOT open the menu"

        # Ctrl+Right Click (button==2) on the transcript opens it.
        transcript = app.query_one("#transcript")
        ctrl_right_click = events.Click(
            widget=transcript,
            x=2,
            y=2,
            delta_x=0,
            delta_y=0,
            button=2,
            shift=False,
            meta=False,
            ctrl=True,
        )
        app.on_click(ctrl_right_click)
        await pilot.pause()
        modal = [s for s in app.screen_stack if isinstance(s, ContextMenuScreen)]
        assert len(modal) == 1, "Ctrl+Right Click should open the context menu"
        option_list = modal[0].query_one("#ctx-list", OptionList)
        labels = [opt.prompt for opt in option_list._options]
        # Transcript-context menu offers Copy selection + copy last/all.
        assert any("Copy selection" in lab for lab in labels)
        assert any("Copy last" in lab for lab in labels)

        # Shift+Right Click (button==2, shift=True) is also a valid trigger.
        await pilot.press("escape")
        await pilot.pause()
        shift_right_click = events.Click(
            widget=transcript,
            x=2,
            y=2,
            delta_x=0,
            delta_y=0,
            button=2,
            shift=True,
            meta=False,
            ctrl=False,
        )
        app.on_click(shift_right_click)
        await pilot.pause()
        modal = [s for s in app.screen_stack if isinstance(s, ContextMenuScreen)]
        assert len(modal) == 1, "Shift+Right Click should open the context menu"
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_context_menu_copy_composer_dispatches_to_clipboard(tmp_path, monkeypatch):
    """Selecting the Copy entry in the composer menu writes the composer text."""
    from glm_acp.tui import ContextMenuScreen

    captured_text = {}

    def fake_write(text: str) -> bool:
        captured_text["value"] = text
        return True

    monkeypatch.setattr("glm_acp.tui._write_system_clipboard", fake_write)

    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)
        composer.focus()
        await pilot.pause()
        composer.value = "hello-from-composer"
        await pilot.pause()

        await pilot.press("f6")
        await pilot.pause()
        modal = [s for s in app.screen_stack if isinstance(s, ContextMenuScreen)]
        assert modal
        option_list = modal[0].query_one("#ctx-list", OptionList)
        # Find the Copy entry (first runnable option after Cut).
        for index, opt in enumerate(option_list._options):
            if opt.id == "copy_composer":
                option_list.highlighted = index
                break
        await pilot.press("enter")
        await pilot.pause()

        assert captured_text.get("value") == "hello-from-composer"
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_context_menu_select_all_output_invokes_screen_text_select_all(
    tmp_path, monkeypatch
):
    """The Select all output menu entry calls the screen selection API."""
    from textual.screen import Screen

    from glm_acp.tui import ContextMenuScreen

    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    called = {"count": 0}

    def fake_select_all(self_):
        called["count"] += 1

    # Patch the class method so it works regardless of which Screen
    # instance is current when the modal-callback path runs (the captured
    # main-screen instance may differ from the post-dismiss ``app.screen``
    # on Windows / non-Linux CI runners).
    monkeypatch.setattr(Screen, "text_select_all", fake_select_all)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        # Defocus the composer so we get the transcript-context menu.
        composer = app.query_one("#composer", Input)
        composer.disabled = True  # force non-composer focus state
        await pilot.pause()

        app.action_open_context_menu()
        await pilot.pause()
        modal = [s for s in app.screen_stack if isinstance(s, ContextMenuScreen)]
        assert modal
        option_list = modal[0].query_one("#ctx-list", OptionList)
        # Highlight the select_all_output entry.
        for index, opt in enumerate(option_list._options):
            if opt.id == "select_all_output":
                option_list.highlighted = index
                break
        await pilot.press("enter")
        await pilot.pause()

        assert called["count"] == 1
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_activity_line_animates_runtime_states_and_returns_ready(tmp_path):
    agent = SlowAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        activity = app.query_one("#activity-status", Static)
        assert "Ready" in str(activity.render())

        composer = app.query_one("#composer", Input)
        composer.value = "Inspect the repository"
        await pilot.press("enter")
        await asyncio.wait_for(agent.prompt_started.wait(), timeout=1)
        assert "Thinking" in str(activity.render())
        initial_frame = app._activity_frame
        app._advance_activity_animation()
        assert app._activity_frame != initial_frame

        await app.handle_session_update(acp.update_agent_thought_text("Considering evidence"))
        assert "Reasoning" in str(activity.render())

        await app.handle_session_update(
            acp.start_tool_call(
                "tool-animated",
                "Search\nrepository for a deliberately very long bounded tool title",
                kind="search",
                status="pending",
            )
        )
        rendered = str(activity.render())
        assert "Working · Search repository" in rendered
        assert "\n" not in rendered

        agent.prompt_release.set()
        for _ in range(20):
            await pilot.pause(0.05)
            if app._prompt_worker is None:
                break
        assert app._prompt_worker is None
        assert "Completed" in str(activity.render())

        app._activity_hold_until = time.monotonic() - 1
        app._advance_activity_animation()
        assert "Ready" in str(activity.render())
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_activity_animation_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_ACP_TUI_ANIMATION", "off")
    app = NativeGlmTui(_args(tmp_path), agent_factory=FakeAgent)

    async with app.run_test(size=(100, 35)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        app._set_activity("Thinking", active=True)
        initial_frame = app._activity_frame
        app._advance_activity_animation()
        assert app._activity_frame == initial_frame
        assert "◆ Thinking" in str(app.query_one("#activity-status", Static).render())
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_f1_submits_help_and_documented_keys_are_actionable(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 45)) as pilot:
        await _wait_for_agent_ready(app, pilot)

        await pilot.press("f1")
        for _ in range(20):
            await pilot.pause(0.05)
            if app._current_agent_text:
                break
        assert app._current_agent_text == "Handled /help"
        assert str(getattr(agent.prompts[-1][0], "text", "")) == "/help"

        await pilot.press("f2")
        await pilot.pause()
        assert not app.query_one("#thinking").has_class("hidden")
        await pilot.press("f2")
        await pilot.pause()
        assert app.query_one("#thinking").has_class("hidden")

        await pilot.press("f3")
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SettingsScreen):
                break
        assert isinstance(app.screen, SettingsScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SettingsScreen)
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_footer_actions_are_clickable_and_quit_uses_terminal_safe_key(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(130, 45)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        for _ in range(40):
            await pilot.pause(0.05)
            if agent.usage_calls:
                break
        assert agent.usage_calls == 1
        await pilot.pause(0.1)

        required_actions = {"quit_agent", "toggle_thinking", "settings", "show_help"}
        footer_keys = {}
        for _ in range(20):
            await pilot.pause(0.05)
            footer_keys = {key.action: key for key in app.query(FooterKey)}
            if required_actions <= footer_keys.keys():
                break
        assert required_actions <= footer_keys.keys()
        assert footer_keys["quit_agent"].key == "ctrl+x"
        assert all(key.key != "ctrl+q" for key in footer_keys.values())

        await pilot.click(
            next(key for key in app.query(FooterKey) if key.action == "toggle_thinking")
        )
        await pilot.pause()
        assert not app.query_one("#thinking").has_class("hidden")

        await pilot.click(next(key for key in app.query(FooterKey) if key.action == "settings"))
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SettingsScreen):
                break
        assert isinstance(app.screen, SettingsScreen)
        await pilot.press("escape")
        await pilot.pause()

        await pilot.click(next(key for key in app.query(FooterKey) if key.action == "show_help"))
        for _ in range(20):
            await pilot.pause(0.05)
            if app._current_agent_text == "Handled /help" and app._prompt_worker is None:
                break
        assert app._current_agent_text == "Handled /help"
        assert app._prompt_worker is None
        await pilot.click(next(key for key in app.query(FooterKey) if key.action == "quit_agent"))
        await pilot.pause()
        assert app._shutdown_requested is True

    assert agent.closed is True


@pytest.mark.asyncio
async def test_tui_quit_does_not_shadow_textual_message_pump_state(tmp_path):
    app = NativeGlmTui(_args(tmp_path), agent_factory=FakeAgent)

    await app.action_quit_agent()

    assert app._shutdown_requested is True
    assert app._closing is False
    assert app._exit is True


@pytest.mark.asyncio
async def test_tui_quit_is_bounded_when_background_cleanup_hangs(tmp_path):
    agent = HangingCloseAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)
    app.SHUTDOWN_TIMEOUT_SECONDS = 0.05
    started = time.monotonic()

    await app._close_agent_resources()

    assert agent.close_started is True
    assert time.monotonic() - started < 1.0


@pytest.mark.asyncio
async def test_tui_local_slash_controls_and_forwarded_command(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 45)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)

        composer.value = "/reasoning-panel"
        await pilot.press("enter")
        await pilot.pause()
        assert not app.query_one("#thinking").has_class("hidden")
        assert agent.prompts == []

        initial_usage_calls = agent.usage_calls
        composer.value = "/usage"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if agent.usage_calls > initial_usage_calls:
                break
        assert agent.usage_calls == initial_usage_calls + 1

        composer.value = "/settings"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SettingsScreen):
                break
        assert isinstance(app.screen, SettingsScreen)
        await pilot.press("escape")
        await pilot.pause()

        composer.value = "/status"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if app._current_agent_text == "Handled /status":
                break
        assert app._current_agent_text == "Handled /status"

        composer.value = "/clear-view"
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.query("#transcript > *")) == 0
        app.exit(0)


@pytest.mark.asyncio
async def test_slash_menu_filters_live_agent_commands_and_supports_tab_escape(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 45)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)
        menu = app.query_one("#command-menu", OptionList)

        composer.value = "/"
        await pilot.pause()
        assert menu.has_class("visible")
        assert "/status" in app._command_values
        assert "/checkpoint" in app._command_values
        assert "/model" in app._command_values
        assert app._command_values[:3] == ["/plan", "/thinking", "/model"]

        composer.value = "/sta"
        await pilot.pause()
        assert app._command_values == ["/status"]
        await pilot.press("tab")
        await pilot.pause()
        assert composer.value == "/status"
        assert agent.prompts == []

        await pilot.press("ctrl+u")
        await pilot.pause()
        assert composer.value == ""

        composer.value = "/sta"
        await pilot.press("escape")
        await pilot.pause()
        assert not menu.has_class("visible")

        composer.value = ""
        composer.value = "/sta"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if app._current_agent_text == "Handled /status":
                break
        assert app._current_agent_text == "Handled /status"
        app.exit(0)


@pytest.mark.asyncio
async def test_slash_model_menu_navigates_and_changes_shared_session(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 45)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)
        menu = app.query_one("#command-menu", OptionList)

        composer.value = "/model"
        await pilot.pause()
        assert app._command_values == ["/model"]
        await pilot.press("enter")
        await pilot.pause()
        assert composer.value == "/model "
        assert menu.has_class("visible")
        assert any(value.startswith("/model glm-") for value in app._command_values)

        target = next(value for value in app._command_values if value != "/model glm-5.2")
        menu.highlighted = app._command_values.index(target)
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if agent.config_calls:
                break
        selected_model = target.partition(" ")[2]
        assert agent.config_calls[-1] == ("model", selected_model)
        assert agent._sessions["tui-session"].model == selected_model
        assert selected_model in str(app.query_one("#session", Static).render())
        assert not menu.has_class("visible")
        app.exit(0)


@pytest.mark.asyncio
async def test_inline_permission_and_mode_commands_use_shared_agent_methods(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 45)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)

        composer.value = "/permission r"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if agent.config_calls:
                break
        assert agent.config_calls[-1] == ("permission_mode", "read")

        composer.value = "/mode a"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if agent.mode_calls:
                break
        assert agent.mode_calls[-1] == "ask"
        panel = str(app.query_one("#session", Static).render())
        assert "ask · read" in panel
        app.exit(0)


@pytest.mark.asyncio
async def test_api_plan_and_thinking_commands_have_full_zed_parity(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(140, 48)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)

        composer.value = "/plan "
        await pilot.pause()
        assert app._command_values == [
            "/plan coding",
            "/plan standard",
            "/plan bigmodel",
        ]
        menu_text = " ".join(
            str(option.prompt) for option in app.query_one("#command-menu", OptionList).options
        )
        assert "Coding Plan" in menu_text
        assert "Standard API" in menu_text
        assert "BigModel (CN)" in menu_text

        composer.value = "/plan standard"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if ("api_endpoint", "standard") in agent.config_calls:
                break
        assert agent._sessions["tui-session"].api_endpoint == "standard"
        assert "Standard API" in str(app.query_one("#session", Static).render())

        composer.value = "/model "
        await pilot.pause()
        assert "/model glm-5v-turbo" in app._command_values
        assert "/model glm-4.5v" in app._command_values
        assert "/model glm-4.6v" in app._command_values

        composer.value = "/thinking "
        await pilot.pause()
        assert app._command_values == [
            "/thinking disabled",
            "/thinking enabled",
            "/thinking high",
            "/thinking max",
        ]
        composer.value = "/thinking max"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if ("thought_level", "max") in agent.config_calls:
                break
        assert agent._sessions["tui-session"].thought_level == "max"
        assert "Deep · Max" in str(app.query_one("#session", Static).render())
        assert app.query_one("#thinking").has_class("hidden")

        agent._sessions["tui-session"].model = "glm-4.7"
        composer.value = "/thinking "
        await pilot.pause()
        assert app._command_values == [
            "/thinking disabled",
            "/thinking enabled",
        ]
        app.exit(0)


@pytest.mark.asyncio
async def test_every_inline_configuration_command_opens_valid_choices(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 45)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)
        menu = app.query_one("#command-menu", OptionList)

        for command in CONFIG_COMMANDS:
            composer.value = f"{command} "
            await pilot.pause()
            assert menu.has_class("visible"), command
            assert app._command_values, command
            assert all(value.startswith(f"{command} ") for value in app._command_values)
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_permission_modal_is_redacted_and_returns_allow(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        composer = app.query_one("#composer", Input)
        composer.value = "Make the requested edit"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, PermissionScreen):
                break
        assert isinstance(app.screen, PermissionScreen)
        assert "Waiting for approval" in str(app.query_one("#activity-status", Static).render())
        detail = str(app.screen.query_one("#permission-detail", Static).render())
        assert "must-never-render" not in detail
        assert "[REDACTED]" in detail
        await pilot.click("#allow")
        for _ in range(20):
            await pilot.pause(0.05)
            if agent.permission is not None:
                break
        assert agent.permission.outcome.outcome == "selected"
        assert app._current_agent_text == "Permission handled."
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_settings_change_shared_session_configuration(tmp_path):
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 45)) as pilot:
        await _wait_for_agent_ready(app, pilot)
        await pilot.press("f3")
        for _ in range(20):
            await pilot.pause(0.05)
            if isinstance(app.screen, SettingsScreen):
                break
        assert isinstance(app.screen, SettingsScreen)
        app.screen.query_one("#permission_mode", Select).value = "read"
        app.screen.query_one("#thought_level", Select).value = "high"
        await pilot.click("#settings-apply")
        for _ in range(20):
            await pilot.pause(0.05)
            if agent._sessions["tui-session"].permission_mode == "read":
                break
        assert agent._sessions["tui-session"].permission_mode == "read"
        assert agent._sessions["tui-session"].thought_level == "high"
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_settings_initial_choices_follow_current_plan_and_model(tmp_path):
    values = {
        "api_endpoint": "coding",
        "model": "glm-4.7",
        "thought_level": "enabled",
        "permission_mode": "ask",
        "generation_profile": "balanced",
        "auxiliary_model": "main",
        "mixture_mode": "off",
        "session_mode": "code",
    }
    screen = SettingsScreen(values)

    class SettingsHost(NativeGlmTui):
        def on_mount(self) -> None:
            self.push_screen(screen)

    app = SettingsHost(_args(tmp_path), agent_factory=FakeAgent)
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.pause()
        model_values = {str(option[1]) for option in screen.query_one("#model", Select)._options}
        thought_values = {
            str(option[1]) for option in screen.query_one("#thought_level", Select)._options
        }
        assert model_values == {"glm-5.2", "glm-5-turbo", "glm-4.7"}
        assert thought_values == {"disabled", "enabled"}
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_composer_stays_enabled_and_queues_prompts_during_turn(tmp_path):
    """While the agent works, the composer stays enabled and Enter queues prompts."""
    agent = SlowAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)

        composer = app.query_one("#composer", Input)

        # Start first prompt — SlowAgent blocks on prompt_release
        composer.value = "First task"
        await pilot.press("enter")
        await asyncio.wait_for(agent.prompt_started.wait(), timeout=1)
        assert app._prompt_worker is not None

        # Composer must remain enabled while the agent works
        assert composer.disabled is False

        # While busy, queue two more prompts
        composer.value = "Second task"
        await pilot.press("enter")
        assert len(app._prompt_queue) == 1
        assert app._prompt_queue[0] == "Second task"

        composer.value = "Third task"
        await pilot.press("enter")
        assert len(app._prompt_queue) == 2

        # Queue display should show the count and preview
        queue_text = str(app.query_one("#queue-status", Static).render())
        assert "Queue (2)" in queue_text
        assert "Second task" in queue_text

        # Release the first prompt — queue should drain automatically
        agent.prompt_release.set()
        for _ in range(60):
            await pilot.pause(0.05)
            if app._prompt_worker is None and not app._prompt_queue:
                break

        # All three prompts should have been sent to the agent in order
        assert len(agent.prompts) == 3
        assert app._prompt_queue == []
        assert app._prompt_worker is None

        # Queue display should be empty after draining
        assert str(app.query_one("#queue-status", Static).render()) == ""
        app.exit(0)


@pytest.mark.asyncio
async def test_tui_working_tree_panel_toggles_and_cycles_four_views(tmp_path):
    """F4 opens the working-tree panel; repeated F4 cycles through all 4 views then closes."""
    (tmp_path / "hello.py").write_text("print('hi')")
    agent = FakeAgent()
    app = NativeGlmTui(_args(tmp_path), agent_factory=lambda: agent)

    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_for_agent_ready(app, pilot)

        panel = app.query_one("#working-tree-panel")
        assert panel.has_class("hidden")

        await pilot.press("f4")
        await pilot.pause(0.15)
        assert app._wt_visible is True
        assert not panel.has_class("hidden")
        switcher = app.query_one("#wt-switcher", ContentSwitcher)
        assert switcher.current == "wt-changes"

        await pilot.press("f4")
        await pilot.pause(0.15)
        assert switcher.current == "wt-git"

        await pilot.press("f4")
        await pilot.pause(0.15)
        assert switcher.current == "wt-diff"

        await pilot.press("f4")
        await pilot.pause(0.15)
        assert switcher.current == "wt-files"

        files_widget = app.query_one("#wt-files", VerticalScroll)
        assert len(list(files_widget.children)) > 0

        await pilot.press("f4")
        await pilot.pause(0.1)
        assert app._wt_visible is False
        assert panel.has_class("hidden")
        app.exit(0)


def test_interactive_chat_routes_to_tui(monkeypatch, tmp_path):
    args = _args(tmp_path)
    called = []
    monkeypatch.setattr("glm_acp.terminal_cli.sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("glm_acp.terminal_cli.sys.stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        "glm_acp.tui.run_tui_command", lambda received: called.append(received) or 17
    )

    assert run_chat_command(args) == 17
    assert called == [args]
