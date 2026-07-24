"""Full-screen Textual frontend for the shared Native GLM ACP runtime."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse
from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.selection import Selection
from textual.timer import Timer
from textual.widgets import (
    Button,
    ContentSwitcher,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    RichLog,
    Select,
    Static,
)
from textual.widgets.option_list import Option
from textual.worker import Worker

from .agent import GlmAcpAgent
from .config import (
    API_ENDPOINTS,
    DEFAULT_AUXILIARY_MODEL,
    DEFAULT_MODEL,
    GENERATION_PROFILES,
    MODELS,
    THOUGHT_LEVELS,
    VISION_MODELS,
    thought_levels_for_model,
)
from .glm_client import PlanUsage

LOCAL_COMMANDS = {
    "/plan": "Switch between Coding Plan, Standard API, and BigModel (CN)",
    "/thinking": "Change provider thinking: Off, Standard, Deep High, or Deep Max",
    "/model": "Change the active GLM model",
    "/usage": "Refresh live 5-hour, weekly, and MCP Coding Plan quota",
    "/permission": "Change Ask, Read Only, or Bypass permissions",
    "/mode": "Change Ask or Code session mode",
    "/generation": "Change the generation style",
    "/auxiliary": "Change the auxiliary model",
    "/mixture": "Enable or disable Mixture of Agents",
    "/settings": "Open all live session settings",
    "/reasoning": "Alias for /thinking",
    "/api-plan": "Alias for /plan",
    "/endpoint": "Alias for /plan",
    "/reasoning-panel": "Show or hide the live reasoning panel",
    "/toggle-thinking": "Alias for /reasoning-panel",
    "/clear-view": "Clear only the visible transcript",
    "/copy": "Copy the last response to clipboard (or /copy <N> for response N, /copy all)",
    "/planmode": "Activate Plan Mode with a PRD: /planmode <your requirements>",
    "/export last": "Export the last response to a Markdown file",
    "/image": "Queue an image for the next prompt",
    "/exit": "Close the terminal agent",
}

CONFIG_COMMANDS = {
    "/plan": ("api_endpoint", "API plan"),
    "/thinking": ("thought_level", "Thinking"),
    "/model": ("model", "Model"),
    "/permission": ("permission_mode", "Permissions"),
    "/generation": ("generation_profile", "Generation style"),
    "/auxiliary": ("auxiliary_model", "Auxiliary model"),
    "/mixture": ("mixture_mode", "Mixture of Agents"),
    "/mode": ("session_mode", "Session mode"),
    "/reasoning": ("thought_level", "Thinking"),
    "/api-plan": ("api_endpoint", "API plan"),
    "/endpoint": ("api_endpoint", "API plan"),
}

MAX_CLIPBOARD_CHARS = 1_000_000


def _read_system_clipboard() -> str:
    """Read the OS clipboard for an explicit Ctrl-V without invoking a shell."""
    if sys.platform == "darwin":
        commands = [("pbpaste",)]
    elif os.name == "nt":
        commands = [
            ("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"),
            ("pwsh", "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"),
        ]
    else:
        commands = []
        if os.environ.get("WAYLAND_DISPLAY"):
            commands.append(("wl-paste",))
        commands.extend(
            [
                ("xclip", "-selection", "clipboard", "-out"),
                ("xsel", "--clipboard", "--output"),
            ]
        )
    allowed_environment = {
        name: value
        for name in (
            "DISPLAY",
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "SystemRoot",
            "USERPROFILE",
            "WAYLAND_DISPLAY",
            "WINDIR",
            "XAUTHORITY",
            "XDG_RUNTIME_DIR",
        )
        if (value := os.environ.get(name))
    }
    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            result = subprocess.run(
                (executable, *command[1:]),
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                env=allowed_environment,
                timeout=1.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout:
            return result.stdout[:MAX_CLIPBOARD_CHARS].rstrip("\r\n")
    return ""


def _write_system_clipboard(text: str) -> bool:
    """Write text to the OS clipboard without invoking a shell."""
    if sys.platform == "darwin":
        commands = [("pbcopy",)]
    elif os.name == "nt":
        commands = [
            ("powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "$input = [Console]::In::ReadToEnd(); Set-Clipboard -Value $input"),
            ("clip",),
        ]
    else:
        commands = []
        if os.environ.get("WAYLAND_DISPLAY"):
            commands.append(("wl-copy",))
        commands.extend(
            [
                ("xclip", "-selection", "clipboard"),
                ("xsel", "--clipboard", "--input"),
            ]
        )
    allowed_environment = {
        name: value
        for name in (
            "DISPLAY",
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "SystemRoot",
            "USERPROFILE",
            "WAYLAND_DISPLAY",
            "WINDIR",
            "XAUTHORITY",
            "XDG_RUNTIME_DIR",
        )
        if (value := os.environ.get(name))
    }
    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            result = subprocess.run(
                (executable, *command[1:]),
                input=text,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                env=allowed_environment,
                timeout=1.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return True
    return False


class CommandInput(Input):
    """Composer input with command-menu navigation while focus stays in place."""

    BINDINGS = [
        Binding("ctrl+shift+v", "paste_system", show=False, priority=True),
        Binding("tab", "command_complete", show=False, priority=True),
        Binding("up", "command_up", show=False, priority=True),
        Binding("down", "command_down", show=False, priority=True),
        Binding("escape", "command_escape", show=False, priority=True),
    ]

    def action_command_complete(self) -> None:
        app = self.app
        if isinstance(app, NativeGlmTui):
            app.accept_command_completion(submit=False)

    def action_command_up(self) -> None:
        app = self.app
        if isinstance(app, NativeGlmTui):
            app.move_command_highlight(-1)

    def action_command_down(self) -> None:
        app = self.app
        if isinstance(app, NativeGlmTui):
            app.move_command_highlight(1)

    def action_command_escape(self) -> None:
        app = self.app
        if isinstance(app, NativeGlmTui):
            app.hide_command_menu()

    def _on_paste(self, event: events.Paste) -> None:
        """Keep a multiline terminal paste usable in the single-line composer."""
        self._insert_pasted_text(event.text)
        event.stop()

    def _insert_pasted_text(self, text: str) -> None:
        if "\n" in text or "\r" in text:
            text = " ".join(text.splitlines()).strip()
        if text:
            selection = self.selection
            if selection.is_empty:
                self.insert_text_at_cursor(text)
            else:
                self.replace(text, *selection)

    def action_paste(self) -> None:
        """Paste the internal clipboard or explicitly read the OS clipboard."""
        text = self.app.clipboard or _read_system_clipboard()
        self._apply_clipboard_text(text)

    def action_paste_system(self) -> None:
        """Read the OS clipboard for terminals that deliver Ctrl-Shift-V as a key."""
        text = _read_system_clipboard() or self.app.clipboard
        self._apply_clipboard_text(text)

    def _apply_clipboard_text(self, text: str) -> None:
        if text:
            self._insert_pasted_text(text)
        else:
            self.app.notify(
                "Clipboard is empty or unavailable; try the terminal paste shortcut",
                severity="warning",
            )


class PermissionScreen(ModalScreen[bool]):
    """Fail-closed approval dialog for a single tool invocation."""

    BINDINGS = [
        Binding("y", "allow", "Allow", priority=True),
        Binding("n", "deny", "Deny", priority=True),
        Binding("escape", "deny", "Deny", priority=True),
    ]

    CSS = """
    PermissionScreen { align: center middle; background: $background 70%; }
    #permission-dialog {
        width: 76; max-width: 95%; height: auto; max-height: 80%;
        border: thick $warning; background: $surface; padding: 1 2;
    }
    #permission-title { text-style: bold; color: $warning; margin-bottom: 1; }
    #permission-detail { max-height: 18; overflow-y: auto; margin-bottom: 1; }
    #permission-buttons { height: 3; align-horizontal: right; }
    #permission-buttons Button { margin-left: 1; }
    """

    def __init__(self, title: str, detail: str) -> None:
        super().__init__()
        self.title = title
        self.detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-dialog"):
            yield Label("Tool permission required", id="permission-title")
            yield Static(f"{self.title}{self.detail}", id="permission-detail", markup=False)
            with Horizontal(id="permission-buttons"):
                yield Button("Deny [N]", id="deny", variant="error")
                yield Button("Allow once [Y]", id="allow", variant="success")

    @on(Button.Pressed, "#allow")
    def allow_button(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def deny_button(self) -> None:
        self.dismiss(False)

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class SettingsScreen(ModalScreen[dict[str, str] | None]):
    """Runtime settings equivalent to the ACP client configuration controls."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]
    FIELD_IDS = (
        "api_endpoint",
        "model",
        "thought_level",
        "permission_mode",
        "generation_profile",
        "auxiliary_model",
        "mixture_mode",
        "session_mode",
    )

    CSS = """
    SettingsScreen { align: center middle; background: $background 70%; }
    #settings-dialog {
        width: 78; max-width: 96%; height: 90%;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    #settings-title { text-style: bold; color: $accent; margin-bottom: 1; }
    #settings-fields { height: 1fr; }
    .settings-label { margin-top: 1; color: $text-muted; }
    .settings-select { width: 100%; }
    #settings-buttons { height: 3; align-horizontal: right; margin-top: 1; }
    #settings-buttons Button { margin-left: 1; }
    """

    def __init__(self, values: dict[str, str]) -> None:
        super().__init__()
        self.values = values

    @staticmethod
    def _options(mapping: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
        return [(str(info.get("name", key)), key) for key, info in mapping.items()]

    def compose(self) -> ComposeResult:
        endpoint = self.values["api_endpoint"]
        model_keys = [key for key, info in MODELS.items() if endpoint in info.get("plans", [])]
        thought_levels = thought_levels_for_model(self.values["model"])
        auxiliary = [("Use main model", DEFAULT_AUXILIARY_MODEL)] + [
            (str(MODELS[key]["name"]), key) for key in model_keys if key not in VISION_MODELS
        ]
        with Vertical(id="settings-dialog"):
            yield Label("Session settings", id="settings-title")
            with VerticalScroll(id="settings-fields"):
                yield Label("API plan", classes="settings-label")
                yield Select(
                    self._options(API_ENDPOINTS),
                    value=self.values["api_endpoint"],
                    allow_blank=False,
                    id="api_endpoint",
                    classes="settings-select",
                )
                yield Label("Model", classes="settings-label")
                yield Select(
                    [(str(MODELS[key]["name"]), key) for key in model_keys],
                    value=self.values["model"],
                    allow_blank=False,
                    id="model",
                    classes="settings-select",
                )
                yield Label("Reasoning", classes="settings-label")
                yield Select(
                    self._options(thought_levels),
                    value=self.values["thought_level"],
                    allow_blank=False,
                    id="thought_level",
                    classes="settings-select",
                )
                yield Label("Permissions", classes="settings-label")
                yield Select(
                    [("Ask", "ask"), ("Read Only", "read"), ("Bypass", "bypass")],
                    value=self.values["permission_mode"],
                    allow_blank=False,
                    id="permission_mode",
                    classes="settings-select",
                )
                yield Label("Generation style", classes="settings-label")
                yield Select(
                    self._options(GENERATION_PROFILES),
                    value=self.values["generation_profile"],
                    allow_blank=False,
                    id="generation_profile",
                    classes="settings-select",
                )
                yield Label("Auxiliary model", classes="settings-label")
                yield Select(
                    auxiliary,
                    value=self.values["auxiliary_model"],
                    allow_blank=False,
                    id="auxiliary_model",
                    classes="settings-select",
                )
                yield Label("Mixture of Agents", classes="settings-label")
                yield Select(
                    [("Off", "off"), ("Reference review", "enabled")],
                    value=self.values["mixture_mode"],
                    allow_blank=False,
                    id="mixture_mode",
                    classes="settings-select",
                )
                yield Label("Session mode", classes="settings-label")
                yield Select(
                    [("Ask", "ask"), ("Code", "code")],
                    value=self.values["session_mode"],
                    allow_blank=False,
                    id="session_mode",
                    classes="settings-select",
                )
            with Horizontal(id="settings-buttons"):
                yield Button("Cancel", id="settings-cancel")
                yield Button("Apply", id="settings-apply", variant="primary")

    @on(Button.Pressed, "#settings-apply")
    def apply_settings(self) -> None:
        values: dict[str, str] = {}
        for field_id in self.FIELD_IDS:
            value = self.query_one(f"#{field_id}", Select).value
            if isinstance(value, str):
                values[field_id] = value
        self.dismiss(values)

    @on(Button.Pressed, "#settings-cancel")
    def cancel_button(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Select.Changed, "#api_endpoint")
    def endpoint_changed(self, event: Select.Changed) -> None:
        endpoint = str(event.value)
        model_keys = [key for key, info in MODELS.items() if endpoint in info.get("plans", [])]
        model_select = self.query_one("#model", Select)
        current_model = str(model_select.value)
        model_select.set_options([(str(MODELS[key]["name"]), key) for key in model_keys])
        model_select.value = current_model if current_model in model_keys else DEFAULT_MODEL
        auxiliary = [("Main model", DEFAULT_AUXILIARY_MODEL)] + [
            (str(MODELS[key]["name"]), key) for key in model_keys if key not in VISION_MODELS
        ]
        auxiliary_select = self.query_one("#auxiliary_model", Select)
        current_auxiliary = str(auxiliary_select.value)
        auxiliary_select.set_options(auxiliary)
        auxiliary_values = {value for _, value in auxiliary}
        auxiliary_select.value = (
            current_auxiliary if current_auxiliary in auxiliary_values else DEFAULT_AUXILIARY_MODEL
        )

    @on(Select.Changed, "#model")
    def model_changed(self, event: Select.Changed) -> None:
        model = str(event.value)
        levels = thought_levels_for_model(model)
        thought_select = self.query_one("#thought_level", Select)
        current = str(thought_select.value)
        thought_select.set_options([(str(THOUGHT_LEVELS[key]["name"]), key) for key in levels])
        thought_select.value = current if current in levels else "enabled"


class ContextMenuOption(Option):
    """A context-menu entry: ``id`` carries the action identifier."""

    def __init__(self, label: str, action_id: str | None, *, disabled: bool = False) -> None:
        super().__init__(label, id=action_id, disabled=disabled)


class ContextMenuScreen(ModalScreen[str | None]):
    """Codex-style right-click dropdown menu.

    A keyboard-navigable popup that surfaces Copy / Cut / Paste / Select All
    and the response-copy commands. Opens on Ctrl+Right Click (primary,
    matching Codex), Shift+Right Click (terminal-friendly fallback), or
    Ctrl+M (keybinding fallback). The chosen action id is returned via
    :py:meth:`dismiss`.
    """

    BINDINGS = [
        Binding("escape", "dismiss_menu", show=False),
        Binding("up", "cursor_up", show=False),
        Binding("down", "cursor_down", show=False),
        Binding("home", "cursor_top", show=False),
        Binding("end", "cursor_bottom", show=False),
        Binding("enter", "select_current", show=False),
    ]

    CSS = """
    ContextMenuScreen { align: center middle; }
    #ctx-menu {
        width: 46; max-height: 70%; border: round #36a3f7;
        background: #0e1620; padding: 0;
    }
    #ctx-title {
        padding: 0 1; background: #1a2838; color: #d9e7f5;
        text-style: bold;
    }
    #ctx-list { padding: 0; scrollbar-size: 0 0; }
    #ctx-list > .option-list--option { padding: 0 1; }
    #ctx-hint {
        padding: 0 1; background: #111a24; color: #7f96ab;
    }
    """

    def __init__(self, options: list[tuple[str, str | None]]) -> None:
        super().__init__()
        self._options = options  # (label, action_id); None id => separator/dismiss

    def compose(self) -> ComposeResult:
        with Vertical(id="ctx-menu"):
            yield Static("Context menu", id="ctx-title", markup=False)
            yield OptionList(
                *[
                    ContextMenuOption(label, action_id, disabled=(action_id is None))
                    for label, action_id in self._options
                ],
                id="ctx-list",
            )
            yield Static(
                "↑↓ move  ·  Enter run  ·  Esc close",
                id="ctx-hint",
                markup=False,
            )

    def on_mount(self) -> None:
        # Children mount asynchronously; defer highlight to the next refresh
        # so ``query_one("#ctx-list")`` resolves after compose().
        self.call_after_refresh(self._highlight_first_runnable)

    def _highlight_first_runnable(self) -> None:
        try:
            option_list = self.query_one("#ctx-list", OptionList)
        except Exception:
            return
        for index, (_, action_id) in enumerate(self._options):
            if action_id is not None:
                option_list.highlighted = index
                return

    def action_cursor_up(self) -> None:
        option_list = self.query_one("#ctx-list", OptionList)
        index = option_list.highlighted
        if index is None:
            index = 0
        else:
            index = max(0, index - 1)
        # Skip separator rows.
        while index > 0 and self._options[index][1] is None:
            index -= 1
        option_list.highlighted = index

    def action_cursor_down(self) -> None:
        option_list = self.query_one("#ctx-list", OptionList)
        index = option_list.highlighted
        if index is None:
            index = 0
        else:
            index = min(len(self._options) - 1, index + 1)
        # Skip separator rows.
        while index < len(self._options) - 1 and self._options[index][1] is None:
            index += 1
        option_list.highlighted = index

    def action_cursor_top(self) -> None:
        for index, (_, action_id) in enumerate(self._options):
            if action_id is not None:
                self.query_one("#ctx-list", OptionList).highlighted = index
                return

    def action_cursor_bottom(self) -> None:
        for index in range(len(self._options) - 1, -1, -1):
            if self._options[index][1] is not None:
                self.query_one("#ctx-list", OptionList).highlighted = index
                return

    def action_select_current(self) -> None:
        option_list = self.query_one("#ctx-list", OptionList)
        index = option_list.highlighted
        if index is None:
            self.dismiss(None)
            return
        action_id = self._options[index][1]
        self.dismiss(action_id)

    @on(OptionList.OptionSelected, "#ctx-list")
    def option_selected(self, event: OptionList.OptionSelected) -> None:
        """Forward Enter / double-click selection to the action dispatcher."""
        event.stop()
        option_list = self.query_one("#ctx-list", OptionList)
        index = option_list.highlighted
        if index is None:
            self.dismiss(None)
            return
        action_id = self._options[index][1]
        self.dismiss(action_id)

    def action_dismiss_menu(self) -> None:
        self.dismiss(None)


class TuiClient:
    """ACP Client adapter that maps updates to Textual widgets."""

    def __init__(self, app: NativeGlmTui) -> None:
        self.app = app
        self._tool_titles: dict[str, str] = {}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        await self.app.handle_session_update(update)

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **kwargs: Any
    ) -> RequestPermissionResponse:
        from .terminal_cli import TerminalClient

        tool_call_id = str(getattr(tool_call, "tool_call_id", ""))
        title = (
            getattr(tool_call, "title", None)
            or self._tool_titles.get(tool_call_id)
            or "requested tool"
        )
        detail = TerminalClient._permission_detail(getattr(tool_call, "raw_input", None))
        self.app._set_activity("Waiting for approval", tone="warning")
        try:
            allowed = await self.app.push_screen_wait(PermissionScreen(str(title), detail))
        finally:
            if self.app._prompt_worker is not None:
                self.app._set_activity(
                    f"Working · {self.app._bounded_activity_label(str(title))}",
                    active=True,
                )
        allow = next((option for option in options if option.option_id == "allow"), None)
        if allowed and allow is not None:
            return RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", option_id=allow.option_id)
            )
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    def remember_tool(self, update: Any) -> None:
        tool_call_id = str(getattr(update, "tool_call_id", ""))
        title = getattr(update, "title", None)
        if tool_call_id and title:
            self._tool_titles[tool_call_id] = str(title)


class NativeGlmTui(App[int]):
    """Full-screen coding-agent interface backed by one ``GlmAcpAgent``."""

    ACTIVITY_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    ACTIVITY_INTERVAL_SECONDS = 0.12
    ACTIVITY_HOLD_SECONDS = 1.6
    TITLE = "Native GLM ACP"
    SUB_TITLE = "Full harness terminal"
    ENABLE_COMMAND_PALETTE = False
    SHUTDOWN_TIMEOUT_SECONDS = 3.0
    BINDINGS = [
        Binding("ctrl+x", "quit_agent", "Quit", priority=True),
        Binding("f10", "quit_agent", "Quit", show=False, priority=True),
        # Ctrl-Q is swallowed by XON/XOFF flow control in many POSIX terminals.
        # Keep it as a hidden compatibility alias for terminals that deliver it.
        Binding("ctrl+q", "quit_agent", "Quit", show=False, priority=True),
        Binding("ctrl+c", "cancel_turn", "Cancel turn", priority=True),
        Binding("ctrl+l", "clear_transcript", "Clear view", priority=True),
        Binding("f1", "show_help", "Help", priority=True),
        Binding("f2", "toggle_thinking", "Reasoning view", priority=True),
        Binding("f3", "settings", "Settings", priority=True),
        Binding("f4", "toggle_working_tree", "Working tree", priority=True),
        Binding("f5", "toggle_voice", "Push to talk", priority=True),
        Binding("ctrl+y", "copy_last_response", "Copy response", priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy selection", show=False, priority=True),
        # Codex-style right-click context menu.
        # Ctrl+Right Click is the primary trigger; F6 is the keyboard
        # fallback for terminals that bind Ctrl+Click at the OS level
        # (Ctrl+M is the same byte as Enter in terminals and would submit).
        Binding("f6", "open_context_menu", "Context menu", show=False, priority=True),
    ]

    CSS = """
    Screen { layout: vertical; background: #0b1017; }
    Header { background: #111a24; color: #d7e3f4; }
    #workspace { height: 1fr; }
    #conversation { width: 1fr; }
    #working-tree-panel {
        width: 34; min-width: 24; border: round #4a9ee6; padding: 0 1;
        background: #0c1118;
    }
    #working-tree-panel.hidden { display: none; }
    #wt-switcher { height: 1fr; }
    #wt-tabs {
        height: 1; dock: bottom; background: #111a24; color: #7f96ab;
        padding: 0 1;
    }
    #transcript {
        height: 1fr; border: round #2589d8; padding: 0 1;
        background: #0d131b;
    }
    #thinking {
        height: 12; min-height: 6; border: round #8a5fd3; padding: 0 1;
        background: #10131c;
    }
    #thinking.hidden { display: none; }
    #sidebar {
        width: 32; min-width: 26; border: round #d29a32; padding: 0 1;
        background: #0c1118;
    }
    #session { height: auto; padding: 0 0 1 0; color: #c8d6e5; }
    #tools {
        height: 1fr; min-height: 7; border-top: solid #66502a;
        border-bottom: solid #66502a; color: #aebdca;
    }
    #plan { height: auto; max-height: 10; overflow-y: auto; padding-top: 1; }
    #command-menu {
        display: none; height: auto; max-height: 14; margin: 0 1;
        border: round #36a3f7; background: #111a24; color: #d9e7f5;
    }
    #command-menu.visible { display: block; }
    #command-hint {
        display: none; height: 1; margin: 0 2; color: #7f96ab;
        background: #0b1017;
    }
    #command-hint.visible { display: block; }
    #activity-status {
        height: 1; margin: 0 2; color: #85c8ff;
        background: #0b1017;
    }
    #queue-status {
        height: 1; margin: 0 2; color: #f6c85f;
        background: #0b1017;
    }
    #composer {
        height: 3; margin: 0 1; border: tall #2589d8;
        background: #111a24;
    }
    Footer { background: #111a24; }
    .welcome {
        margin: 1 3; padding: 1 2; border-left: thick #36a3f7;
        background: #111a24;
    }
    .user-message { margin: 1 1 0 8; padding: 1; background: #12314b; }
    .agent-message { margin: 1 8 0 1; padding: 1; background: #171d26; }
    .system-message { margin: 1 4; color: $text-muted; }
    """

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        agent_factory: Callable[[], GlmAcpAgent] = GlmAcpAgent,
    ) -> None:
        super().__init__()
        self.args = args
        self.agent = agent_factory()
        self.client = TuiClient(self)
        self.session_id = ""
        self._agent_ready = False
        # Do not use MessagePump._closing: Textual owns it and setting it before
        # App.exit() prevents the queued ExitApp message from being processed.
        self._shutdown_requested = False
        self._agent_closed = False
        self._prompt_worker: Worker[None] | None = None
        self._prompt_queue: list[str] = []
        self._wt_visible: bool = False
        self._wt_view: int = 0
        self._recorder: object | None = None
        self._turn_start_time: float = time.monotonic()
        self._replaying = False
        self._current_agent: Static | None = None
        self._current_agent_text = ""
        self._agent_responses: list[str] = []
        self._last_agent_render: float = 0.0
        self._thinking_text = ""
        self._pending_images = list(args.image)
        self._slash_commands = dict(LOCAL_COMMANDS)
        self._command_values: list[str] = []
        self._provider_usage: PlanUsage | None = None
        self._provider_usage_error = ""
        animation_setting = os.environ.get("GLM_ACP_TUI_ANIMATION", "1")
        self._activity_animation_enabled = animation_setting.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._activity_timer: Timer | None = None
        self._activity_frame = 0
        self._activity_label = "Starting session"
        self._activity_tone = "active"
        self._activity_active = False
        self._activity_started = time.monotonic()
        self._activity_hold_until: float | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="working-tree-panel", classes="hidden"):
                yield ContentSwitcher(
                    VerticalScroll(id="wt-changes"),
                    VerticalScroll(id="wt-git"),
                    VerticalScroll(id="wt-diff"),
                    VerticalScroll(id="wt-files"),
                    initial="wt-changes",
                    id="wt-switcher",
                )
                yield Static("[1]Changes [2]Git [3]Diff [4]Files  (F4)", id="wt-tabs", markup=False)
            with Vertical(id="conversation"):
                yield VerticalScroll(id="transcript")
                yield RichLog(
                    id="thinking",
                    classes="hidden",
                    wrap=True,
                    markup=False,
                    auto_scroll=True,
                )
            with Vertical(id="sidebar"):
                yield Static("Starting…", id="session", markup=False)
                yield RichLog(id="tools", wrap=True, markup=True, auto_scroll=True)
                yield Static("No active plan", id="plan", markup=False)
        yield OptionList(id="command-menu", compact=True)
        yield Static(
            "↑↓ navigate  ·  Enter run/select  ·  Tab complete  ·  Esc close",
            id="command-hint",
            markup=False,
        )
        yield Static("◌ Starting session", id="activity-status", markup=False)
        yield Static("", id="queue-status", markup=False)
        yield CommandInput(
            placeholder="Ask Native GLM ACP… (/help for commands)",
            id="composer",
            disabled=True,
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#transcript").border_title = "Conversation"
        self.query_one("#thinking").border_title = "Reasoning"
        self.query_one("#sidebar").border_title = "Session"
        self.query_one("#tools").border_title = "Activity"
        self.query_one("#command-menu").border_title = "Commands"
        self.query_one("#tools", RichLog).write("[dim]Waiting for tool activity…[/dim]")
        self._activity_timer = self.set_interval(
            self.ACTIVITY_INTERVAL_SECONDS,
            self._advance_activity_animation,
            name="tui-activity-animation",
            pause=True,
        )
        self._set_activity("Starting session", active=True)
        self.agent.on_connect(self.client)
        self.initialize_agent()

    @work(exclusive=True, group="agent-initialize")
    async def initialize_agent(self) -> None:
        from .terminal_cli import _configure

        try:
            await self.agent.initialize(
                protocol_version=1,
                client_info={"name": "glm-acp-tui"},
                client_capabilities={"terminal": True},
            )
            if self.args.resume:
                self._replaying = True
                await self.agent.resume_session(
                    cwd=self.args.cwd,
                    session_id=self.args.resume,
                    additional_directories=self.args.additional_dir,
                )
                self._replaying = False
                self.session_id = self.args.resume
            else:
                response = await self.agent.new_session(
                    cwd=self.args.cwd,
                    additional_directories=self.args.additional_dir,
                )
                self.session_id = response.session_id
            await _configure(self.agent, self.session_id, self.args)
            self._agent_ready = True
            self.query_one("#composer", Input).disabled = False
            self.query_one("#composer", Input).focus()
            self._refresh_session_panel("Ready")
            self._set_activity("Ready", tone="ready")
            await self._append_welcome()
            self.refresh_provider_usage(silent=True)
        except Exception as error:
            await self._append_system(f"Startup failed: {error}")
            self._refresh_session_panel("Startup failed")
            self._set_activity("Startup failed", tone="error")

    @on(Input.Submitted, "#composer")
    async def submit_input(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        completion = self.accept_command_completion()
        if completion == "expanded":
            return
        if completion == "selected":
            text = event.input.value.strip()
        if not text or not self._agent_ready:
            return
        event.input.clear()
        if text.startswith("/planmode ") and self._agent_ready:
            prd = text.partition(" ")[2].strip()
            if prd:
                await self.agent.set_session_mode(
                    mode_id="plan", session_id=self.session_id
                )
                self.notify("Plan Mode activated — read-only research mode", severity="information")
                text = prd
            else:
                return
        if await self._handle_local_command(text):
            return
        if text in {"/exit", "/quit"}:
            await self.action_quit_agent()
            return
        if text.startswith("/image "):
            path = text.partition(" ")[2].strip()
            if path:
                self._pending_images.append(path)
                await self._append_system(f"Queued image for the next prompt: {path}")
            return
        if self._prompt_worker is not None:
            self._prompt_queue.append(text)
            self._refresh_queue_display()
            return
        await self._append_user(text)
        self._current_agent = None
        self._current_agent_text = ""
        self._thinking_text = ""
        self.query_one("#thinking", RichLog).clear()
        self._refresh_session_panel("Running")
        self._set_activity("Thinking", active=True)
        self._prompt_worker = self.run_prompt(text, list(self._pending_images))
        self._pending_images.clear()

    @on(Input.Changed, "#composer")
    def composer_changed(self, event: Input.Changed) -> None:
        if event.value == event.input.value:
            self.refresh_command_menu(event.value)

    @on(OptionList.OptionSelected, "#command-menu")
    async def command_selected(self, event: OptionList.OptionSelected) -> None:
        result = self.accept_command_completion(index=event.option_index)
        if result == "selected":
            await self.query_one("#composer", CommandInput).action_submit()

    async def _handle_local_command(self, text: str) -> bool:
        """Handle presentation-only commands without entering the model loop."""
        if text in {"/reasoning-panel", "/toggle-thinking"}:
            self.action_toggle_thinking()
            return True
        if text == "/usage":
            self.refresh_provider_usage(silent=False)
            return True
        if text == "/settings":
            self.action_settings()
            return True
        if text == "/clear-view":
            await self.action_clear_transcript()
            self.notify("Transcript view cleared", severity="information")
            return True
        if text == "/copy" or text == "/copy last":
            await self._copy_response(None)
            return True
        if text.startswith("/copy "):
            arg = text.partition(" ")[2].strip()
            if arg.isdigit():
                await self._copy_response(int(arg))
                return True
            if arg == "all":
                await self._copy_all_responses()
                return True
        if text == "/export last":
            await self._export_last_response()
            return True
        command, separator, argument = text.partition(" ")
        if command in CONFIG_COMMANDS:
            if not separator or not argument.strip():
                composer = self.query_one("#composer", CommandInput)
                composer.value = f"{command} "
                composer.cursor_position = len(composer.value)
                self.refresh_command_menu(composer.value)
                return True
            await self._apply_inline_config(command, argument.strip())
            return True
        return False

    async def _apply_inline_config(self, command: str, value: str) -> None:
        choices = {item for item, _description in self._configuration_values(command)}
        _config_id, label = CONFIG_COMMANDS[command]
        if value not in choices:
            await self._append_system(
                f"Unknown {label.lower()} value: {value}. Available: {', '.join(sorted(choices))}"
            )
            return
        if command == "/mode":
            await self.agent.set_session_mode(mode_id=value, session_id=self.session_id)
        else:
            config_id = CONFIG_COMMANDS[command][0]
            await self.agent.set_config_option(
                config_id=config_id,
                session_id=self.session_id,
                value=value,
            )
        session = self.agent._sessions[self.session_id]
        actual = value if command == "/mode" else str(getattr(session, CONFIG_COMMANDS[command][0]))
        self._refresh_session_panel("Ready")
        await self._append_system(f"✓ {label}: {self._display_config_value(command, actual)}")
        if CONFIG_COMMANDS[command][0] == "api_endpoint":
            self.refresh_provider_usage(silent=True)

    def _configuration_values(self, command: str) -> list[tuple[str, str]]:
        session = self.agent._sessions.get(self.session_id)
        if session is None:
            return []
        config_id = CONFIG_COMMANDS.get(command, ("", ""))[0]
        values: list[tuple[str, str]]
        if config_id == "model":
            values = [
                (
                    key,
                    f"{info['name']} — {info['description']} ({info['context_window']} context)",
                )
                for key, info in MODELS.items()
                if session.api_endpoint in info.get("plans", [])
            ]
        elif config_id == "thought_level":
            values = [
                (
                    key,
                    f"{THOUGHT_LEVELS[key]['name']} — {THOUGHT_LEVELS[key]['description']}",
                )
                for key in thought_levels_for_model(session.model)
            ]
        elif config_id == "permission_mode":
            values = [
                ("ask", "Ask before changes"),
                ("read", "Read Only"),
                ("bypass", "Bypass"),
            ]
        elif config_id == "session_mode":
            values = [("ask", "Ask / explain"), ("code", "Code / act")]
        elif config_id == "api_endpoint":
            values = [
                (key, f"{info['name']} — {info['description']}")
                for key, info in API_ENDPOINTS.items()
            ]
        elif config_id == "generation_profile":
            values = [
                (key, f"{info['name']} — {info['description']}")
                for key, info in GENERATION_PROFILES.items()
            ]
        elif config_id == "auxiliary_model":
            values = [(DEFAULT_AUXILIARY_MODEL, "Use main model")]
            values.extend(
                (key, str(MODELS[key]["name"]))
                for key, info in MODELS.items()
                if session.api_endpoint in info.get("plans", []) and key not in VISION_MODELS
            )
        elif config_id == "mixture_mode":
            values = [("off", "Off"), ("enabled", "Reference review")]
        else:
            return []
        current = self._current_config_value(command)
        return [
            (item, f"{description} · current" if item == current else description)
            for item, description in values
        ]

    def _current_config_value(self, command: str) -> str:
        session = self.agent._sessions.get(self.session_id)
        if session is None:
            return ""
        if command == "/mode":
            return str(session.mode)
        config_id = CONFIG_COMMANDS.get(command, ("", ""))[0]
        return str(getattr(session, config_id, ""))

    @staticmethod
    def _display_config_value(command: str, value: str) -> str:
        config_id = CONFIG_COMMANDS.get(command, ("", ""))[0]
        if config_id == "api_endpoint":
            return str(API_ENDPOINTS.get(value, {}).get("name", value))
        if config_id == "model":
            return str(MODELS.get(value, {}).get("name", value))
        if config_id == "thought_level":
            return str(THOUGHT_LEVELS.get(value, {}).get("name", value))
        if config_id == "generation_profile":
            return str(GENERATION_PROFILES.get(value, {}).get("name", value))
        if config_id == "auxiliary_model" and value == DEFAULT_AUXILIARY_MODEL:
            return "Use main model"
        return value

    def refresh_command_menu(self, value: str) -> None:
        menu = self.query_one("#command-menu", OptionList)
        if not value.startswith("/") or self._prompt_worker is not None:
            self.hide_command_menu()
            return
        command, separator, argument = value.partition(" ")
        choices: list[tuple[str, str]]
        if command in CONFIG_COMMANDS and separator:
            query = argument.strip().lower()
            available = [
                (f"{command} {item}", description)
                for item, description in self._configuration_values(command)
            ]
            prefix_matches = [
                item for item in available if item[0].partition(" ")[2].lower().startswith(query)
            ]
            choices = (
                available
                if not query
                else prefix_matches or [item for item in available if query in item[1].lower()]
            )
        elif separator:
            self.hide_command_menu()
            return
        else:
            query = value.lower()
            available = [
                (
                    name,
                    (
                        f"{description} · current "
                        f"{self._display_config_value(name, self._current_config_value(name))}"
                        if name in CONFIG_COMMANDS
                        else description
                    ),
                )
                for name, description in self._slash_commands.items()
            ]
            exact_matches = [item for item in available if item[0] == query]
            prefix_matches = [item for item in available if item[0].startswith(query)]
            choices = (
                available
                if query == "/"
                else exact_matches
                or prefix_matches
                or [item for item in available if query[1:] in item[1].lower()]
            )
        choices = choices[:50]
        if not choices:
            self.hide_command_menu()
            return
        self._command_values = [choice for choice, _description in choices]
        menu.set_options(
            [
                Option(
                    Text.assemble(
                        (choice, "bold #62b5f5"),
                        "  ",
                        (
                            description
                            if len(description) <= 88
                            else description[:87].rstrip() + "…",
                            "dim",
                        ),
                    ),
                    id=f"choice-{index}",
                )
                for index, (choice, description) in enumerate(choices)
            ]
        )
        menu.highlighted = 0
        menu.add_class("visible")
        self.query_one("#command-hint").add_class("visible")

    def hide_command_menu(self) -> None:
        self.query_one("#command-menu").remove_class("visible")
        self.query_one("#command-hint").remove_class("visible")
        self._command_values = []

    def move_command_highlight(self, direction: int) -> None:
        menu = self.query_one("#command-menu", OptionList)
        if not menu.has_class("visible"):
            return
        if direction < 0:
            menu.action_cursor_up()
        else:
            menu.action_cursor_down()

    def accept_command_completion(
        self, *, index: int | None = None, submit: bool = True
    ) -> str | None:
        menu = self.query_one("#command-menu", OptionList)
        if not menu.has_class("visible") or not self._command_values:
            return None
        selected_index = menu.highlighted if index is None else index
        if selected_index is None or not 0 <= selected_index < len(self._command_values):
            return None
        value = self._command_values[selected_index]
        composer = self.query_one("#composer", CommandInput)
        if value in CONFIG_COMMANDS:
            composer.value = f"{value} "
            composer.cursor_position = len(composer.value)
            self.refresh_command_menu(composer.value)
            return "expanded"
        composer.value = value
        composer.cursor_position = len(value)
        if not submit:
            self.refresh_command_menu(value)
            return "expanded"
        self.hide_command_menu()
        return "selected"

    @work(exclusive=True, group="agent-prompt", exit_on_error=False)
    async def run_prompt(self, text: str, images: list[str]) -> None:
        from .terminal_cli import _prompt_blocks

        outcome = "completed"
        try:
            while True:
                self._turn_start_time = time.monotonic()
                try:
                    await self.agent.prompt(
                        prompt=_prompt_blocks(text, images),
                        session_id=self.session_id,
                        message_id=str(uuid4()),
                    )
                except asyncio.CancelledError:
                    outcome = "cancelled"
                    await self._append_system("Turn cancelled.")
                    raise
                except Exception as error:
                    outcome = "failed"
                    await self._append_system(f"Turn failed: {error}")
                    return

                if self._current_agent is not None and self._current_agent_text:
                    self._current_agent.update(RichMarkdown(self._current_agent_text))
                    self._last_agent_render = time.monotonic()

                if not self._prompt_queue or self._shutdown_requested:
                    return

                text = self._prompt_queue.pop(0)
                images = []
                self._refresh_queue_display()
                await self._append_user(text)
                self._current_agent = None
                self._current_agent_text = ""
                self._thinking_text = ""
                self.query_one("#thinking", RichLog).clear()
                self._refresh_session_panel("Running from queue")
                self._set_activity("Thinking", active=True)
        finally:
            self._prompt_worker = None
            if not self._shutdown_requested:
                from .voice import play_sound, send_notification

                turn_duration = time.monotonic() - self._turn_start_time
                if outcome == "completed":
                    play_sound("success")
                    send_notification(
                        "GLM ACP", "Task completed", turn_duration=turn_duration
                    )
                elif outcome == "failed":
                    play_sound("error")
                    send_notification(
                        "GLM ACP", "Turn failed", error=True, turn_duration=turn_duration
                    )
                elif outcome == "cancelled":
                    play_sound("warning")

                composer = self.query_one("#composer", Input)
                composer.focus()
                self._refresh_session_panel("Ready")
                if outcome == "completed":
                    self._set_activity(
                        "Completed",
                        tone="success",
                        hold=self.ACTIVITY_HOLD_SECONDS,
                    )
                elif outcome == "cancelled":
                    self._set_activity(
                        "Cancelled",
                        tone="warning",
                        hold=self.ACTIVITY_HOLD_SECONDS,
                    )
                else:
                    self._set_activity(
                        "Turn failed",
                        tone="error",
                        hold=self.ACTIVITY_HOLD_SECONDS,
                    )

    @work(exclusive=True, group="provider-usage", exit_on_error=False)
    async def refresh_provider_usage(self, *, silent: bool) -> None:
        """Refresh provider-reported quota data without blocking the composer."""
        try:
            usage = await self._query_provider_usage()
        except Exception as error:
            self._provider_usage = None
            self._provider_usage_error = str(error)[:300]
            self._refresh_session_panel("Running" if self._prompt_worker is not None else "Ready")
            if not silent:
                await self._append_system(
                    f"Coding Plan usage is unavailable: {self._provider_usage_error}"
                )
            return
        self._provider_usage = usage
        self._provider_usage_error = ""
        self._refresh_session_panel("Running" if self._prompt_worker is not None else "Ready")
        if not silent:
            rendered = self.agent.format_provider_usage(usage).replace("**", "").replace("_", "")
            await self._append_system(rendered)

    async def _query_provider_usage(self) -> PlanUsage:
        """Run synchronous DNS/HTTP away from the UI loop in a daemon thread."""
        sync_query = getattr(self.agent, "query_provider_usage_sync", None)
        if not callable(sync_query):
            return await self.agent.query_provider_usage(self.session_id)

        loop = asyncio.get_running_loop()
        result: asyncio.Future[PlanUsage] = loop.create_future()

        def resolve() -> None:
            try:
                usage = sync_query(self.session_id)
            except Exception as error:
                outcome: tuple[PlanUsage | None, Exception | None] = (None, error)
            else:
                outcome = (usage, None)

            def deliver() -> None:
                if result.done():
                    return
                usage_value, error_value = outcome
                if error_value is not None:
                    result.set_exception(error_value)
                else:
                    assert usage_value is not None
                    result.set_result(usage_value)

            try:
                loop.call_soon_threadsafe(deliver)
            except RuntimeError:
                pass

        threading.Thread(
            target=resolve,
            name="glm-acp-provider-usage",
            daemon=True,
        ).start()
        return await result

    async def handle_session_update(self, update: Any) -> None:
        kind = str(getattr(update, "session_update", ""))
        if kind == "available_commands_update":
            for command in getattr(update, "available_commands", []):
                name = "/" + str(getattr(command, "name", "")).lstrip("/")
                description = str(getattr(command, "description", "") or "Harness command")
                if name != "/":
                    self._slash_commands[name] = description
        elif kind == "agent_message_chunk":
            if self._prompt_worker is not None:
                self._set_activity("Responding", active=True)
            await self._append_agent(self._content_text(update))
        elif kind == "user_message_chunk" and self._replaying:
            await self._append_user(self._content_text(update), history=True)
        elif kind == "agent_thought_chunk":
            if self._prompt_worker is not None:
                self._set_activity("Reasoning", active=True)
            text = self._content_text(update)
            self._thinking_text += text
            self.query_one("#thinking", RichLog).write(text, scroll_end=True)
        elif kind in {"tool_call", "tool_call_update"}:
            self.client.remember_tool(update)
            tool_call_id = str(getattr(update, "tool_call_id", ""))
            title = getattr(update, "title", None) or self.client._tool_titles.get(tool_call_id)
            status = getattr(update, "status", None)
            if title:
                if self._prompt_worker is not None:
                    self._set_activity(
                        f"Working · {self._bounded_activity_label(str(title))}",
                        active=True,
                    )
                self.query_one("#tools", RichLog).write(
                    f"[bold]{escape(str(title))}[/bold] · {escape(str(status or 'tool'))}",
                    scroll_end=True,
                )
            content_items = getattr(update, "content", None)
            if content_items:
                for item in content_items:
                    text = self._extract_tool_content_text(item)
                    if text:
                        for line in text.strip().splitlines()[:25]:
                            self.query_one("#tools", RichLog).write(
                                f"  [dim]{escape(line)}[/dim]",
                                scroll_end=True,
                            )
        elif kind == "plan":
            lines = ["Plan"]
            for entry in getattr(update, "entries", []):
                marker = {"completed": "✓", "in_progress": "▶"}.get(
                    str(getattr(entry, "status", "")), "○"
                )
                lines.append(f"{marker} {getattr(entry, 'content', '')}")
            self.query_one("#plan", Static).update("\n".join(lines))
        elif kind == "usage_update":
            self._refresh_session_panel(
                "Running" if self._prompt_worker is not None else "Ready",
                used=int(getattr(update, "used", 0)),
                size=int(getattr(update, "size", 0)),
            )
        elif kind == "current_mode_update":
            mode_id = str(getattr(update, "current_mode_id", ""))
            if mode_id:
                session = self.agent._sessions.get(self.session_id)
                if session is not None:
                    session.mode = mode_id
                self._refresh_session_panel(
                    "Running" if self._prompt_worker is not None else "Ready"
                )
        elif kind == "session_info_update":
            self.sub_title = str(getattr(update, "title", "Full harness terminal"))

    @staticmethod
    def _content_text(update: Any) -> str:
        return str(getattr(getattr(update, "content", None), "text", ""))

    @staticmethod
    def _extract_tool_content_text(item: Any) -> str:
        """Extract readable text from a ToolCallContent item."""
        inner = getattr(item, "content", None)
        if inner is None:
            text = getattr(item, "text", None)
            return str(text) if text else ""
        if isinstance(inner, str):
            return inner
        parts: list[str] = []
        if isinstance(inner, (list, tuple)):
            for block in inner:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    async def _append_user(self, text: str, *, history: bool = False) -> None:
        self._current_agent = None
        label = "You · history" if history else "You"
        widget = Static(f"{label}\n{text}", classes="user-message", markup=False)
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(widget)
        transcript.scroll_end(animate=False)

    async def _append_agent(self, text: str) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        if self._current_agent is None:
            if self._current_agent_text:
                self._agent_responses.append(self._current_agent_text)
            self._current_agent_text = ""
            self._current_agent = Static("", classes="agent-message", markup=False)
            await transcript.mount(self._current_agent)
        self._current_agent_text += text
        now = time.monotonic()
        if now - self._last_agent_render > 0.12:
            self._last_agent_render = now
            self._current_agent.update(RichMarkdown(self._current_agent_text))
        transcript.scroll_end(animate=False)

    async def _append_system(self, text: str) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(Static(text, classes="system-message", markup=False))
        transcript.scroll_end(animate=False)

    async def _append_welcome(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(
            Static(
                RichMarkdown(
                    "### Native GLM ACP\n"
                    "Full coding-agent runtime is ready. Type **`/`** for commands, "
                    "**`/plan`** to switch APIs, **`/thinking`** for reasoning depth, "
                    "or **F3** for all settings."
                ),
                classes="welcome",
                markup=False,
            )
        )
        transcript.scroll_end(animate=False)

    def _refresh_session_panel(self, state: str, *, used: int = 0, size: int = 0) -> None:
        session = getattr(self.agent, "_sessions", {}).get(self.session_id)
        model = getattr(session, "model", self.args.model or "default")
        reasoning = getattr(session, "thought_level", "default")
        endpoint = getattr(session, "api_endpoint", "default")
        permission = getattr(session, "permission_mode", self.args.permission or "ask")
        mode = getattr(session, "mode", self.args.mode or "code")
        context = f"{used:,}/{size:,}" if size else "waiting"
        reasoning_name = str(THOUGHT_LEVELS.get(reasoning, {}).get("name", reasoning))
        endpoint_name = str(API_ENDPOINTS.get(endpoint, {}).get("name", endpoint))
        quota = self._quota_summary()
        awareness = self._awareness_summary()
        self.query_one("#session", Static).update(
            f"● {state}\n"
            f"{(self.session_id[:8] + '…') if self.session_id else 'starting'}\n\n"
            f"{model} · {reasoning_name}\n"
            f"{endpoint_name}\n"
            f"{mode} · {permission}\n"
            f"context {context}\n"
            f"{awareness}\n"
            f"{quota}"
        )

    def _awareness_summary(self) -> str:
        """Compact one-line epistemic and metacognitive state for the panel."""
        session = getattr(self.agent, "_sessions", {}).get(self.session_id)
        if session is None:
            return "awareness —"
        awareness = getattr(session, "awareness", None)
        metacog = getattr(session, "metacognition", None)
        if awareness is None or metacog is None:
            return "awareness —"
        active = awareness.active_records()
        contradictions = sum(1 for r in active if r.kind == "contradiction")
        observations = sum(1 for r in active if r.kind == "observation")
        assessment = getattr(metacog, "assessment", None)
        exec_mode = getattr(assessment, "execution_mode", None) or "direct"
        risk = getattr(assessment, "risk_score", 0)
        if contradictions:
            return f"⚠ {exec_mode} · {contradictions} contradiction · /awareness"
        if observations:
            return f"⬡ {exec_mode} · {observations} evidence · risk {risk}"
        return f"⬡ {exec_mode} · risk {risk}"

    def _refresh_queue_display(self) -> None:
        """Update the queue-status widget to show queued prompts."""
        widget = self.query_one("#queue-status", Static)
        if not self._prompt_queue:
            widget.update("")
            return
        count = len(self._prompt_queue)
        preview = " · ".join(
            f"[{i + 1}] {item[:60]}" for i, item in enumerate(self._prompt_queue[:3])
        )
        suffix = f" (+{count - 3} more)" if count > 3 else ""
        widget.update(f"📋 Queue ({count}): {preview}{suffix}")

    def _quota_summary(self) -> str:
        if self._provider_usage is None:
            return "quota unavailable · /usage" if self._provider_usage_error else "quota loading…"
        windows: list[str] = []
        for quota in self._provider_usage.quotas:
            if quota.percentage is None:
                continue
            if quota.kind == "TOKENS_LIMIT" and quota.unit == 3:
                windows.append(f"5h {quota.percentage:g}%")
            elif quota.kind == "TOKENS_LIMIT" and quota.unit == 6:
                windows.append(f"week {quota.percentage:g}%")
            elif quota.kind == "TIME_LIMIT":
                windows.append(f"MCP {quota.percentage:g}%")
        return "quota " + (" · ".join(windows) if windows else "reported · /usage")

    @staticmethod
    def _bounded_activity_label(label: str, limit: int = 56) -> str:
        """Keep streamed tool titles to one bounded terminal line."""
        normalized = " ".join(label.split())
        return normalized if len(normalized) <= limit else normalized[: limit - 1].rstrip() + "…"

    def _set_activity(
        self,
        label: str,
        *,
        active: bool = False,
        tone: str = "active",
        hold: float | None = None,
    ) -> None:
        """Set presentation-only activity without becoming session truth."""
        bounded_label = self._bounded_activity_label(label)
        if (
            hold is None
            and self._activity_hold_until is None
            and bounded_label == self._activity_label
            and tone == self._activity_tone
            and active == self._activity_active
        ):
            return
        self._activity_label = bounded_label
        self._activity_tone = tone
        self._activity_active = active
        self._activity_frame = 0
        self._activity_started = time.monotonic()
        self._activity_hold_until = (
            self._activity_started + hold if hold is not None and hold > 0 else None
        )
        self._render_activity()
        if self._activity_timer is not None:
            if (active and self._activity_animation_enabled) or self._activity_hold_until:
                self._activity_timer.resume()
            else:
                self._activity_timer.pause()

    def _advance_activity_animation(self) -> None:
        if self._activity_hold_until is not None and time.monotonic() >= self._activity_hold_until:
            self._set_activity("Ready", tone="ready")
            return
        if self._activity_active and self._activity_animation_enabled:
            self._activity_frame = (self._activity_frame + 1) % len(self.ACTIVITY_FRAMES)
        self._render_activity()

    def _render_activity(self) -> None:
        widget = self.query_one("#activity-status", Static)
        styles = {
            "active": "bold #85c8ff",
            "success": "bold #68d391",
            "warning": "bold #f6c85f",
            "error": "bold #ff7b72",
            "ready": "#7f96ab",
        }
        symbols = {
            "success": "✓",
            "warning": "○",
            "error": "!",
            "ready": "●",
        }
        if self._activity_active:
            symbol = (
                self.ACTIVITY_FRAMES[self._activity_frame]
                if self._activity_animation_enabled
                else "◆"
            )
        else:
            symbol = symbols.get(self._activity_tone, "•")
        rendered = Text()
        rendered.append(f"{symbol} {self._activity_label}", style=styles[self._activity_tone])
        if self._activity_active and self._activity_animation_enabled:
            elapsed = max(0.0, time.monotonic() - self._activity_started)
            rendered.append(f"  {elapsed:.1f}s", style="dim #7f96ab")
        widget.update(rendered)

    async def action_cancel_turn(self) -> None:
        if self._prompt_worker is None:
            self.notify("No active turn", severity="information")
            return
        self._set_activity("Cancelling", active=True, tone="warning")
        await self.agent.cancel(session_id=self.session_id)
        self._prompt_worker.cancel()

    async def action_quit_agent(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.exit(0)

    async def _close_agent_resources(self) -> None:
        if self._agent_closed:
            return
        self._agent_closed = True
        if self._recorder is not None:
            getattr(self._recorder, "cleanup", lambda: None)()
            self._recorder = None
        try:
            if self._prompt_worker is not None and self.session_id:
                await self.agent.cancel(session_id=self.session_id)
                self._prompt_worker.cancel()
            await asyncio.wait_for(
                self.agent.aclose(),
                timeout=self.SHUTDOWN_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    async def action_clear_transcript(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.remove_children()
        self._current_agent = None
        self._current_agent_text = ""

    async def action_show_help(self) -> None:
        composer = self.query_one("#composer", Input)
        if composer.disabled:
            self.notify("Help is unavailable while a turn is running", severity="warning")
            return
        composer.value = "/help"
        composer.focus()
        await composer.action_submit()

    def action_toggle_thinking(self) -> None:
        thinking = self.query_one("#thinking", RichLog)
        thinking.toggle_class("hidden")
        state = "hidden" if thinking.has_class("hidden") else "shown"
        self.notify(f"Reasoning panel {state}", severity="information")

    def action_settings(self) -> None:
        if self._agent_ready and self._prompt_worker is None:
            self.open_settings()
        else:
            self.notify("Settings are unavailable while a turn is running", severity="warning")

    async def action_toggle_working_tree(self) -> None:
        """Cycle: closed → Changes → Git → Diff → Files → closed."""
        panel = self.query_one("#working-tree-panel")
        if not self._wt_visible:
            self._wt_visible = True
            self._wt_view = 0
            panel.remove_class("hidden")
            await self._switch_wt_view(0)
            return
        self._wt_view += 1
        if self._wt_view >= 4:
            self._wt_visible = False
            panel.add_class("hidden")
            self.notify("Working tree panel closed", severity="information")
        else:
            await self._switch_wt_view(self._wt_view)

    _WT_VIEW_IDS = ("wt-changes", "wt-git", "wt-diff", "wt-files")
    _WT_VIEW_LABELS = ("Changes", "Git", "Diff", "Files")

    async def _switch_wt_view(self, view: int) -> None:
        switcher = self.query_one("#wt-switcher", ContentSwitcher)
        switcher.current = self._WT_VIEW_IDS[view]
        tabs = []
        for i, label in enumerate(self._WT_VIEW_LABELS):
            prefix = "▶ " if i == view else "  "
            tabs.append(f"{prefix}[{i + 1}]{label}")
        self.query_one("#wt-tabs", Static).update(" ".join(tabs) + "  (F4)")
        await self._refresh_wt_view(view)

    async def _refresh_wt_view(self, view: int) -> None:
        refreshers = (
            self._refresh_wt_changes,
            self._refresh_wt_git,
            self._refresh_wt_diff,
            self._refresh_wt_files,
        )
        await refreshers[view]()

    def _session_cwd(self) -> str:
        session = getattr(self.agent, "_sessions", {}).get(self.session_id)
        return str(getattr(session, "cwd", os.getcwd()))

    async def _refresh_wt_changes(self) -> None:
        widget = self.query_one("#wt-changes", VerticalScroll)
        await widget.remove_children()
        session = getattr(self.agent, "_sessions", {}).get(self.session_id)
        verification = getattr(session, "verification", None)
        paths = getattr(verification, "changed_paths", None) or []
        if not paths:
            await widget.mount(Static("No files changed this session yet.", markup=False))
            return
        for path in sorted(set(str(p) for p in paths)):
            await widget.mount(Static(f"📝 {path}", markup=False))

    async def _refresh_wt_git(self) -> None:
        widget = self.query_one("#wt-git", VerticalScroll)
        await widget.remove_children()
        cwd = self._session_cwd()
        result = await asyncio.to_thread(self._run_git, cwd, "status", "--short", "--porcelain")
        if result is None:
            await widget.mount(Static("Not a git repository.", markup=False))
            return
        if not result.strip():
            await widget.mount(Static("Working tree clean — no changes.", markup=False))
            return
        for line in result.strip().splitlines():
            await widget.mount(Static(line, markup=False))

    async def _refresh_wt_diff(self) -> None:
        widget = self.query_one("#wt-diff", VerticalScroll)
        await widget.remove_children()
        cwd = self._session_cwd()
        result = await asyncio.to_thread(self._run_git, cwd, "diff", "--stat")
        if not result or not result.strip():
            await widget.mount(Static("No uncommitted changes to diff.", markup=False))
            return
        await widget.mount(Static(result[:8000], markup=False))

    async def _refresh_wt_files(self) -> None:
        widget = self.query_one("#wt-files", VerticalScroll)
        await widget.remove_children()
        cwd = self._session_cwd()
        try:
            entries = sorted(os.listdir(cwd))
        except OSError:
            await widget.mount(Static("Cannot read directory.", markup=False))
            return
        shown = 0
        for entry in entries:
            if entry.startswith("."):
                continue
            full = os.path.join(cwd, entry)
            marker = "📁" if os.path.isdir(full) else "📄"
            await widget.mount(Static(f"{marker} {entry}", markup=False))
            shown += 1
            if shown >= 200:
                await widget.mount(Static(f"… ({len(entries) - shown} more)", markup=False))
                break

    @staticmethod
    def _run_git(cwd: str, *args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            return result.stdout
        except (OSError, subprocess.TimeoutExpired):
            return None

    async def action_toggle_voice(self) -> None:
        """Push-to-talk: F5 toggles recording, transcribes via local Whisper."""
        from .voice import (
            VoiceRecorder,
            is_voice_available,
            suppress_sound_during_recording,
            transcribe_audio,
        )

        if not is_voice_available():
            self.notify(
                "Voice requires: uv pip install -e '.[voice]'",
                title="Push to talk unavailable",
                severity="warning",
            )
            return

        if self._recorder is not None and getattr(self._recorder, "recording", False):
            wav_path = self._recorder.stop()
            suppress_sound_during_recording()
            self._set_activity("Transcribing…", active=True)
            if wav_path:
                text = await transcribe_audio(wav_path)
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
                if text:
                    composer = self.query_one("#composer", Input)
                    if composer.value and not composer.value.endswith(" "):
                        composer.value += " "
                    composer.value += text
                    composer.cursor_position = len(composer.value)
                    composer.focus()
                    self._set_activity(f"✓ {text[:50]}", tone="success", hold=3.0)
                else:
                    self._set_activity("Transcription empty", tone="warning", hold=3.0)
            else:
                self._set_activity("Recording failed", tone="error", hold=3.0)
            self._recorder = None
        else:
            recorder = VoiceRecorder()
            if recorder.start():
                self._recorder = recorder
                suppress_sound_during_recording()
                self._set_activity("🎤 Recording… (F5 to stop)", active=True)
            else:
                self._recorder = None
                self.notify("Microphone unavailable", severity="warning")

    async def action_copy_last_response(self) -> None:
        """Ctrl+Y: copy the last agent response to the system clipboard."""
        await self._copy_response(None)

    async def action_copy_selection(self) -> None:
        """Ctrl+Shift+C: copy the currently selected text to the clipboard."""
        try:
            text = self.screen.get_selected_text()
        except Exception:
            text = ""
        if text:
            if _write_system_clipboard(text):
                self.notify(
                    f"Copied {len(text)} characters to clipboard", severity="success"
                )
            else:
                self.notify("Clipboard unavailable (install xclip or xsel)", severity="warning")
        else:
            self.notify(
                "No text selected — click and drag to select, then Ctrl+Shift+C",
                severity="information",
            )

    async def _copy_response(self, index: int | None) -> None:
        """Copy a specific agent response to the clipboard.

        index=None copies the most recent response.
        index=1 copies the first response, index=2 the second, etc.
        """
        if index is None:
            text = self._current_agent_text or (
                self._agent_responses[-1] if self._agent_responses else ""
            )
        else:
            responses = self._agent_responses[:]
            if self._current_agent_text:
                responses.append(self._current_agent_text)
            if 1 <= index <= len(responses):
                text = responses[index - 1]
            else:
                self.notify(
                    f"Response {index} not found (have {len(responses)})", severity="warning"
                )
                return
        if not text:
            self.notify("No response to copy", severity="warning")
            return
        if _write_system_clipboard(text):
            preview = text[:60].replace("\n", " ")
            self.notify(f"Copied to clipboard: {preview}…", severity="success")
        else:
            self.notify("Clipboard unavailable (install xclip or xsel)", severity="warning")

    async def _copy_all_responses(self) -> None:
        """Copy all agent responses concatenated to the clipboard."""
        responses = self._agent_responses[:]
        if self._current_agent_text:
            responses.append(self._current_agent_text)
        if not responses:
            self.notify("No responses to copy", severity="warning")
            return
        text = "\n\n---\n\n".join(responses)
        if _write_system_clipboard(text):
            self.notify(f"Copied {len(responses)} responses to clipboard", severity="success")
        else:
            self.notify("Clipboard unavailable", severity="warning")

    # ------------------------------------------------------------------
    # Codex-style right-click context menu (Ctrl+Right Click / Shift+Right
    # Click / Ctrl+M keyboard fallback). Provides Copy / Cut / Paste /
    # Select All for the composer and Copy selection / Copy last response /
    # Copy all responses / Select all output for the transcript area.
    # ------------------------------------------------------------------

    def on_click(self, event: events.Click) -> None:
        """Pop a Codex-style dropdown menu on Ctrl+Right Click or Shift+Right Click.

        ``button == 2`` is the right mouse button in Textual's SGR encoding.
        Many Linux terminals bind Ctrl+Click to native actions (open link,
        terminal menu) that swallow the event before Textual receives it,
        so Shift+Right Click is accepted as a terminal-friendly fallback
        and F6 is exposed as a pure-keyboard alternative.
        """
        if event.button != 2:
            return
        if not (event.ctrl or event.shift):
            return
        event.stop()
        event.prevent_default()
        # event.widget is the widget under the cursor (set by Textual during
        # event dispatch). When it is the composer (or a descendant), offer
        # Cut/Copy/Paste/Select All; otherwise offer the transcript-context
        # Copy selection / Copy last response / Copy all / Select all output.
        self._open_context_menu(clicked_widget=event.widget)

    def action_open_context_menu(self) -> None:
        """F6: keyboard fallback to open the right-click context menu."""
        self._open_context_menu(clicked_widget=None)

    def _open_context_menu(self, clicked_widget: Any = None) -> None:
        composer = self.query_one("#composer", CommandInput)
        if clicked_widget is not None:
            composer_clicked = (
                clicked_widget is composer or composer in clicked_widget.ancestors_with_self
            )
        else:
            # Keyboard (F6) path: fall back to the currently focused widget.
            composer_clicked = self.focused is composer
        if composer_clicked:
            options: list[tuple[str, str | None]] = [
                ("✂   Cut", "cut_composer"),
                ("⧉   Copy", "copy_composer"),
                ("⎘   Paste from clipboard", "paste"),
                ("▭   Select all", "select_all_composer"),
                ("────────────────────────────", None),
                ("⧉   Copy last response", "copy_last"),
                ("⧉   Copy all responses", "copy_all"),
                ("────────────────────────────", None),
                ("✕   Close menu", None),
            ]
        else:
            options = [
                ("⧉   Copy selection", "copy_selection"),
                ("▭   Select all output", "select_all_output"),
                ("⧉   Copy last response", "copy_last"),
                ("⧉   Copy all responses", "copy_all"),
                ("⎘   Paste to composer", "paste"),
                ("────────────────────────────", None),
                ("✕   Close menu", None),
            ]
        self.push_screen(ContextMenuScreen(options), self._context_menu_callback)

    def _context_menu_callback(self, action: str | None) -> None:
        if action is None:
            return
        sync_handlers: dict[str, Callable[[], Any]] = {
            "select_all_output": self.action_select_all_output,
            "paste": self.action_paste_to_composer,
            "cut_composer": self.action_cut_composer,
            "copy_composer": self.action_copy_composer,
            "select_all_composer": self.action_select_all_composer,
        }
        async_handlers: dict[str, Callable[[], Awaitable[Any]]] = {
            "copy_selection": self.action_copy_selection,
            "copy_last": self.action_copy_last_response,
            "copy_all": self._copy_all_responses,
        }
        if action in async_handlers:
            self.run_worker(async_handlers[action](), exclusive=True, group="ctx-menu")
        elif action in sync_handlers:
            sync_handlers[action]()

    def action_select_all_output(self) -> None:
        """Highlight all text in the transcript for a subsequent Copy."""
        try:
            self.screen.text_select_all()
        except Exception:
            self.notify("Selection is unavailable in this terminal", severity="warning")
            return
        self.notify(
            "Selected all output — press Ctrl+Shift+C or open the menu again to copy",
            severity="information",
        )

    def action_paste_to_composer(self) -> None:
        """Paste the OS clipboard into the composer (used by the context menu)."""
        composer = self.query_one("#composer", CommandInput)
        composer.focus()
        text = _read_system_clipboard() or self.app.clipboard
        if not text:
            self.notify("Clipboard is empty or unavailable", severity="warning")
            return
        composer._apply_clipboard_text(text)
        self.notify(f"Pasted {len(text)} characters", severity="success")

    def action_cut_composer(self) -> None:
        """Cut the composer selection to the OS clipboard."""
        composer = self.query_one("#composer", CommandInput)
        composer.focus()
        selected = composer.selected_text
        if not selected:
            self.notify("No text selected in composer", severity="information")
            return
        if not _write_system_clipboard(selected):
            self.notify("Clipboard unavailable (install xclip or xsel)", severity="warning")
            return
        composer.delete_selection()
        self.notify(f"Cut {len(selected)} characters", severity="success")

    def action_copy_composer(self) -> None:
        """Copy the composer selection (or whole line if none) to the clipboard."""
        composer = self.query_one("#composer", CommandInput)
        composer.focus()
        selected = composer.selected_text or composer.value
        if not selected:
            self.notify("Nothing to copy", severity="information")
            return
        if _write_system_clipboard(selected):
            self.notify(f"Copied {len(selected)} characters", severity="success")
        else:
            self.notify("Clipboard unavailable (install xclip or xsel)", severity="warning")

    def action_select_all_composer(self) -> None:
        """Select all text in the composer."""
        composer = self.query_one("#composer", CommandInput)
        composer.focus()
        composer.selection = Selection.from_offsets(
            (0, 0), (0, len(composer.value))
        )
        self.notify("Selected all text in composer", severity="information")

    async def _export_last_response(self) -> None:
        """Export the last agent response to a timestamped Markdown file."""
        text = self._current_agent_text or (
            self._agent_responses[-1] if self._agent_responses else ""
        )
        if not text:
            self.notify("No response to export", severity="warning")
            return
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"glm-acp-export-{timestamp}.md"
        cwd = self._session_cwd()
        filepath = os.path.join(cwd, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# GLM ACP Response\n\n{text}\n")
            self.notify(f"Exported to {filename}", severity="success")
        except OSError as exc:
            self.notify(f"Export failed: {exc}", severity="error")

    @work(exclusive=True, group="settings")
    async def open_settings(self) -> None:
        session = self.agent._sessions[self.session_id]
        values = {
            "api_endpoint": session.api_endpoint,
            "model": session.model,
            "thought_level": session.thought_level,
            "permission_mode": session.permission_mode,
            "generation_profile": session.generation_profile,
            "auxiliary_model": session.auxiliary_model,
            "mixture_mode": session.mixture_mode,
            "session_mode": session.mode,
        }
        selected = await self.push_screen_wait(SettingsScreen(values))
        if not selected:
            return
        for config_id in (
            "api_endpoint",
            "model",
            "thought_level",
            "permission_mode",
            "generation_profile",
            "auxiliary_model",
            "mixture_mode",
        ):
            await self.agent.set_config_option(
                config_id=config_id,
                session_id=self.session_id,
                value=selected[config_id],
            )
        await self.agent.set_session_mode(
            mode_id=selected["session_mode"], session_id=self.session_id
        )
        self._refresh_session_panel("Ready")
        await self._append_system("Session settings updated.")

    async def on_unmount(self) -> None:
        self._shutdown_requested = True
        await self._close_agent_resources()


def run_tui_command(args: argparse.Namespace) -> int:
    """Run the full-screen frontend and return its process exit status."""
    from pathlib import Path

    if not Path(args.cwd).is_dir():
        print(f"Workspace does not exist: {args.cwd}", file=__import__("sys").stderr)
        return 2
    app = NativeGlmTui(args)
    exit_code = 0
    try:
        result = app.run()
        exit_code = int(result or 0)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as error:
        print(f"Native GLM ACP TUI failed: {error}", file=__import__("sys").stderr)
        exit_code = 1
    session_id = getattr(app, "session_id", "")
    if session_id:
        print(
            f"\n📋 Session saved. To resume this conversation:\n"
            f"   glm-acp chat --resume {session_id}\n",
            file=sys.stderr,
        )
    return exit_code
