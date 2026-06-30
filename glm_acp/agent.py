"""ACP agent implementation for the Z.ai GLM API.

This is the main entry point. The agent speaks the Agent Client Protocol
over stdio, so Zed (or any ACP client) launches it as a subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import acp
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    CloseSessionResponse,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionOption,
    PromptCapabilities,
    PromptResponse,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionInfo,
    SessionInfoUpdate,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SessionAdditionalDirectoriesCapabilities,
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    UsageUpdate,
)

from .config import (
    API_ENDPOINTS,
    CHARS_PER_TOKEN,
    COMPACTION_KEEP_RECENT,
    COMPACTION_THRESHOLD,
    CONTEXT_WINDOW_TOKENS,
    DEFAULT_API_ENDPOINT,
    DEFAULT_MODEL,
    DESTRUCTIVE_TOOLS,
    MODELS,
    THOUGHT_LEVELS,
    VISION_MODELS,
    models_for_plan,
    thought_levels_for_model,
)
from .glm_client import GlmClient
from .session_store import SessionStore
from .tools import TOOL_DEFINITIONS, TOOL_KINDS, Sandbox, ToolError, execute_tool

logger = logging.getLogger("glm_acp")

SYSTEM_PROMPT = """\
You are an expert software engineer working inside the user's editor via ACP.
You have tools to read, write, and edit files, search code, and run commands.

Rules:
- Read files before editing them. Understand the codebase structure first.
- Make minimal, surgical changes. Match existing conventions.
- Use run_command for builds, tests, and git operations.
- When writing or editing files, briefly explain what you're changing.
- If a task spans many files, work through them systematically.

Planning:
- For any task with 3+ steps, call update_plan FIRST to lay out your plan.
- Keep tasks concise (one action per entry). Break large work into sub-tasks.
- Update the plan as you work: mark the current task 'in_progress', mark \
finished ones 'completed'.
- The plan helps the user track your progress — keep it accurate.
- For simple questions or single-step tasks, skip the plan.
"""

MODE_LIST = [
    SessionMode(id="ask", name="Ask", description="Answer questions and read files without making changes"),
    SessionMode(id="code", name="Code", description="Full access: read, write, edit, and run commands"),
]

# Markers wrapping the compaction summary so the model knows it's a summary,
# not a real user message.
_COMPACTION_MARKER_OPEN = (
    "<conversation_summary>\n"
    "The following is a summary of the earlier part of this conversation. "
    "The original messages have been compacted to save context space. "
    "Use this summary as authoritative context for everything discussed so far.\n\n"
)
_COMPACTION_MARKER_CLOSE = "\n</conversation_summary>"


class Session:
    """Per-session state."""

    def __init__(self, session_id: str, cwd: str, additional_dirs: list[str] | None = None):
        self.id = session_id
        self.cwd = cwd
        self.sandbox = Sandbox(cwd, additional_dirs)
        self.model = DEFAULT_MODEL
        self.thought_level = "enabled"
        self.mode = "code"
        self.api_endpoint = DEFAULT_API_ENDPOINT
        self.title: str | None = None
        # Permission mode: "ask" (approve destructive tools), "bypass" (auto-approve), "read" (read-only)
        self.permission_mode = "ask"
        # Current task plan — list of PlanEntry-like dicts
        self.plan: list[dict[str, str]] = []
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        # Token tracking — updated after each API call
        self.context_size: int = CONTEXT_WINDOW_TOKENS.get(self.model, 1_000_000)
        self.estimated_tokens: int = 0
        self.last_reported_tokens: int = -1

    # ------------------------------------------------------------------
    # Serialization for persistence across process restarts
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the conversation state for on-disk storage."""
        return {
            "version": 1,
            "cwd": self.cwd,
            "model": self.model,
            "thought_level": self.thought_level,
            "mode": self.mode,
            "api_endpoint": self.api_endpoint,
            "title": self.title,
            "permission_mode": self.permission_mode,
            "plan": self.plan,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], session_id: str) -> "Session":
        """Rebuild a Session from persisted data.

        The cwd is taken from *data* so the sandbox matches the original
        workspace even if the stored path differs from what the client
        passes in load_session.
        """
        cwd = data.get("cwd", ".")
        session = cls(session_id, cwd)
        session.model = data.get("model", DEFAULT_MODEL)
        session.thought_level = data.get("thought_level", "enabled")
        session.mode = data.get("mode", "code")
        session.api_endpoint = data.get("api_endpoint", DEFAULT_API_ENDPOINT)
        session.plan = data.get("plan", [])
        session.title = data.get("title")
        session.permission_mode = data.get("permission_mode", "ask")
        messages = data.get("messages")
        if messages:
            session.messages = messages
        session.context_size = CONTEXT_WINDOW_TOKENS.get(session.model, 1_000_000)
        return session


class GlmAcpAgent(acp.Agent):
    _conn: Client
    _sessions: dict[str, Session]

    def __init__(self) -> None:
        self._sessions = {}
        self._store = SessionStore()

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(
                    image=True,
                    audio=False,
                    embedded_context=True,
                ),
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                    close=SessionCloseCapabilities(),
                    additional_directories=SessionAdditionalDirectoriesCapabilities(),
                ),
            ),
            agent_info=Implementation(
                name="glm-acp",
                title="Z.ai GLM",
                version="0.1.0",
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session = Session(str(uuid4()), cwd, additional_directories)
        self._sessions[session.id] = session
        self._store.save(session.id, session.to_dict())

        await self._send_available_commands(session)

        config_options = [
            self._build_model_option(session),
            self._build_thought_option(session),
            self._build_api_endpoint_option(session),
            self._build_permission_option(session),
        ]

        return NewSessionResponse(
            session_id=session.id,
            modes=SessionModeState(
                current_mode_id=session.mode,
                available_modes=MODE_LIST,
            ),
            config_options=config_options,
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        """Recreate a previously-created session so Zed can resume its chat.

        The conversation state (message history, model, mode) is persisted
        to disk after every turn, so a process restart no longer loses the
        conversation.  If no saved state exists we fall back gracefully to
        a fresh session.
        """
        data = self._store.load(session_id)
        if data:
            session = Session.from_dict(data, session_id)
            logger.info(
                "Restored session %s with %d messages from disk",
                session_id,
                len(session.messages),
            )
        else:
            session = Session(session_id, cwd, additional_directories)
            logger.info("No saved state for session %s — starting fresh", session_id)

        self._sessions[session.id] = session

        # Recalculate token estimate from restored messages and report it
        session.estimated_tokens = self._estimate_tokens(session.messages)
        session.last_reported_tokens = -1

        # The ACP load_session / resume_session responses only carry config
        # options and modes — NOT the message history.  To make the previous
        # conversation visible in the editor UI we must *replay* it back via
        # session_update notifications (user_message_chunk /
        # agent_message_chunk), the same channel used during a live prompt.
        await self._replay_history(session)

        # Replay the task plan if one exists
        if session.plan:
            await self._send_plan(session)

        await self._send_available_commands(session)

        config_options = [
            self._build_model_option(session),
            self._build_thought_option(session),
            self._build_api_endpoint_option(session),
            self._build_permission_option(session),
        ]

        return LoadSessionResponse(
            modes=SessionModeState(
                current_mode_id=session.mode,
                available_modes=MODE_LIST,
            ),
            config_options=config_options,
        )

    async def list_sessions(
        self,
        cwd: str | None = None,
        additional_directories: list[str] | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """Return persisted sessions so Zed can populate its history sidebar."""
        all_sessions = self._store.list()
        # Filter by cwd if the client asked for a specific workspace.
        if cwd:
            all_sessions = [s for s in all_sessions if s.get("cwd") == cwd]
        sessions = [
            SessionInfo(
                session_id=s["session_id"],
                cwd=s.get("cwd", ""),
                title=s.get("title"),
                updated_at=s.get("updated_at"),
            )
            for s in all_sessions
        ]
        return ListSessionsResponse(sessions=sessions)

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        """Resume a session — same restore logic as load_session."""
        data = self._store.load(session_id)
        if data:
            session = Session.from_dict(data, session_id)
            logger.info(
                "Resumed session %s with %d messages from disk",
                session_id,
                len(session.messages),
            )
        else:
            session = Session(session_id, cwd, additional_directories)
            logger.info("No saved state for session %s — starting fresh", session_id)

        self._sessions[session.id] = session

        # Recalculate token estimate from restored messages and report it
        session.estimated_tokens = self._estimate_tokens(session.messages)
        session.last_reported_tokens = -1

        # Replay the conversation history so it shows up in the UI.
        await self._replay_history(session)

        # Replay the task plan if one exists
        if session.plan:
            await self._send_plan(session)

        await self._send_available_commands(session)

        config_options = [
            self._build_model_option(session),
            self._build_thought_option(session),
            self._build_api_endpoint_option(session),
            self._build_permission_option(session),
        ]

        return ResumeSessionResponse(
            modes=SessionModeState(
                current_mode_id=session.mode,
                available_modes=MODE_LIST,
            ),
            config_options=config_options,
        )

    async def close_session(
        self,
        session_id: str,
        **kwargs: Any,
    ) -> CloseSessionResponse:
        """Clean up a closed session and remove its persisted state."""
        self._sessions.pop(session_id, None)
        self._store.delete(session_id)
        return CloseSessionResponse()

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> SetSessionConfigOptionResponse | None:
        session = self._sessions[session_id]
        if config_id == "model":
            session.model = str(value)
            session.context_size = CONTEXT_WINDOW_TOKENS.get(session.model, 1_000_000)
            # If the new model doesn't support the current thought level,
            # fall back to the closest supported level.
            if session.thought_level not in thought_levels_for_model(session.model):
                session.thought_level = "enabled" if "enabled" in thought_levels_for_model(session.model) else "disabled"
        elif config_id == "thought_level":
            session.thought_level = str(value)
        elif config_id == "api_endpoint":
            session.api_endpoint = str(value)
            # If the current model isn't available on the new plan,
            # fall back to the default model.
            if session.model not in models_for_plan(session.api_endpoint):
                session.model = DEFAULT_MODEL
                session.context_size = CONTEXT_WINDOW_TOKENS.get(session.model, 1_000_000)
                if session.thought_level not in thought_levels_for_model(session.model):
                    session.thought_level = "enabled" if "enabled" in thought_levels_for_model(session.model) else "disabled"
        elif config_id == "permission_mode":
            session.permission_mode = str(value)

        self._store.save(session.id, session.to_dict())

        return SetSessionConfigOptionResponse(
            config_options=[
                self._build_model_option(session),
                self._build_thought_option(session),
                self._build_api_endpoint_option(session),
                self._build_permission_option(session),
            ],
        )

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> Any:
        session = self._sessions[session_id]
        session.mode = mode_id
        self._store.save(session.id, session.to_dict())
        from acp.schema import SetSessionModeResponse
        return SetSessionModeResponse()

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        session = self._sessions[session_id]

        # Extract images and text from the ACP prompt blocks.
        content, images = self._extract_prompt_parts(prompt)

        # --- Slash commands ---
        # Intercept messages starting with "/" and handle them directly
        # without sending anything to the model.
        stripped = content.strip()
        if stripped.startswith("/") and not images:
            # Show the user's command in the chat
            await self._send_user_message(session.id, stripped)
            response = await self._handle_command(session, stripped)
            if response:
                await self._send_message(session.id, f"\n\n{response}\n")
            return PromptResponse(stop_reason="end_turn", user_message_id=message_id)

        # Text-only models can't process images. Vision models can.
        is_vision_model = session.model in VISION_MODELS

        if images:
            saved_paths = await self._save_images(session, images)
            if is_vision_model:
                # Vision model: pass image data inline so the model can see it
                content = (content or "") + self._format_image_prompt(images, saved_paths)
            else:
                # Text-only model: save to disk, tell the user
                file_note = (
                    f"\n\n[The user shared {len(images)} screenshot"
                    f"{'s' if len(images) > 1 else ''}. "
                    f"Saved to: {', '.join(saved_paths)}. "
                    f"Note: {session.model} is a text-only model and cannot view "
                    f"images directly. To analyze screenshots, enable the Z.ai Vision "
                    f"MCP Server or describe the screenshot in text.]"
                )
                content = (content or "") + file_note
                await self._send_message(
                    session.id,
                    f"\n\n📸 Screenshot saved to: {', '.join(saved_paths)}\n"
                    f"_{session.model} is text-only — switch to a Vision model "
                    f"or use the Vision MCP Server to analyze images._\n",
                )

        session.messages.append({"role": "user", "content": content})

        # Derive a title from any text in the prompt.
        text_only = self._extract_text(prompt)
        if not session.title:
            session.title = text_only[:60].strip() or "New Chat"
            await self._send_session_info(session)

        try:
            stop_reason = await self._run_turn(session)
            return PromptResponse(stop_reason=stop_reason, user_message_id=message_id)
        except asyncio.CancelledError:
            return PromptResponse(stop_reason="cancelled")
        except Exception as e:
            logger.exception("Prompt turn failed")
            await self._send_message(session.id, f"\n\n**Error:** {e}")
            return PromptResponse(stop_reason="end_turn")
        finally:
            # Persist the updated conversation so it survives restarts.
            self._store.save(session.id, session.to_dict())

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        pass

    async def _run_turn(self, session: Session) -> str:
        """Execute the full model-turn loop: stream → tool calls → repeat."""
        thought_config = THOUGHT_LEVELS.get(session.thought_level, {})
        base_url = API_ENDPOINTS.get(session.api_endpoint, {}).get(
            "base_url", API_ENDPOINTS[DEFAULT_API_ENDPOINT]["base_url"]
        )
        client = GlmClient(
            session.model,
            thought_level=thought_config.get("thinking_type", "enabled"),
            reasoning_effort=thought_config.get("reasoning_effort"),
            base_url=base_url,
        )
        tools = TOOL_DEFINITIONS if session.mode == "code" else [
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] in ("read_file", "list_directory", "search_files", "grep", "update_plan")
        ]

        try:
            for _ in range(50):
                # --- Compaction check before each API call ---
                await self._maybe_compact(session, client)

                result = await client.stream_completion(
                    messages=session.messages,
                    tools=tools,
                    on_reasoning=lambda chunk: self._send_thought(session.id, chunk),
                    on_content=lambda chunk: self._send_message(session.id, chunk),
                    on_tool_call_started=lambda tc_id, name: self._start_tool(session.id, tc_id, name),
                )

                # --- Update token estimates and notify Zed ---
                self._update_usage(session, result.usage)
                await self._report_usage(session)

                if result.reasoning and not result.content:
                    pass

                if not result.tool_calls:
                    session.messages.append({"role": "assistant", "content": result.content})
                    return "end_turn"

                assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.content}
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["function"]["name"],
                                  "arguments": json.dumps(tc["function"]["arguments"])}}
                    for tc in result.tool_calls
                ]
                session.messages.append(assistant_msg)

                for tc in result.tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]
                    tc_id = tc["id"]

                    # --- Plan tool: handled in-agent, not via sandbox ---
                    if tool_name == "update_plan":
                        await self._complete_tool(session.id, tc_id, "Plan updated.")
                        output = await self._handle_update_plan(session, tc_id, tool_args)
                        session.messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": output,
                        })
                        continue

                    await self._update_tool(session.id, tc_id, status="in_progress")

                    # --- Permission check ---
                    permitted, deny_reason = await self._check_permission(
                        session, tc_id, tool_name, tool_args,
                    )
                    if not permitted:
                        output = deny_reason
                        await self._fail_tool(session.id, tc_id, deny_reason)
                        session.messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": output,
                        })
                        continue

                    try:
                        output = await execute_tool(tool_name, tool_args, session.sandbox)
                        await self._complete_tool(session.id, tc_id, output)
                    except ToolError as e:
                        error_msg = str(e)
                        output = f"Error: {error_msg}"
                        await self._fail_tool(session.id, tc_id, error_msg)

                    session.messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": output,
                    })
        finally:
            await client.aclose()

        return "end_turn"

    # ------------------------------------------------------------------
    # Token estimation & usage reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """Rough token estimate based on character count heuristic.

        Accounts for message content, tool call arguments, tool result
        content, and per-message structural overhead (role tags, etc.).
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total_chars += len(content)
            # Tool call arguments + names
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                total_chars += len(fn.get("name", ""))
                total_chars += len(fn.get("arguments", ""))
            # Per-message overhead (role, delimiters, etc.) ~ 4 tokens
            total_chars += CHARS_PER_TOKEN * 4
        return total_chars // CHARS_PER_TOKEN

    @staticmethod
    def _update_usage(session: Session, usage: dict[str, int] | None) -> None:
        """Update estimated token count. Prefer API-reported values."""
        if usage and usage.get("input_tokens"):
            session.estimated_tokens = usage["input_tokens"]
        else:
            session.estimated_tokens = GlmAcpAgent._estimate_tokens(session.messages)

    async def _report_usage(self, session: Session) -> None:
        """Send a UsageUpdate notification to the client if usage changed."""
        used = session.estimated_tokens
        if used != session.last_reported_tokens:
            session.last_reported_tokens = used
            update = UsageUpdate(
                session_update="usage_update",
                size=session.context_size,
                used=used,
            )
            await self._conn.session_update(session_id=session.id, update=update)

    # ------------------------------------------------------------------
    # Task plan / todo list
    # ------------------------------------------------------------------

    async def _handle_update_plan(
        self,
        session: Session,
        tool_call_id: str,
        args: dict[str, Any],
    ) -> str:
        """Handle the update_plan tool call.

        Converts the model's plan into an ACP ``plan`` session update so
        Zed renders the checklist, stores it on the session for
        persistence, and returns a confirmation string as the tool result.
        """
        tasks = args.get("tasks", [])
        entries = []
        for task in tasks:
            entries.append({
                "content": task.get("content", ""),
                "priority": task.get("priority", "medium"),
                "status": task.get("status", "pending"),
            })

        session.plan = entries
        await self._send_plan(session)
        self._store.save(session.id, session.to_dict())

        # Return a compact summary as the tool result
        n_pending = sum(1 for e in entries if e["status"] == "pending")
        n_progress = sum(1 for e in entries if e["status"] == "in_progress")
        n_done = sum(1 for e in entries if e["status"] == "completed")
        return (
            f"Plan updated: {len(entries)} tasks "
            f"({n_done} completed, {n_progress} in progress, {n_pending} pending)."
        )

    async def _send_plan(self, session: Session) -> None:
        """Send the current plan as an ACP plan session update."""
        from acp.helpers import plan_entry

        entries = [
            plan_entry(
                content=e["content"],
                priority=e.get("priority", "medium"),
                status=e.get("status", "pending"),
            )
            for e in session.plan
        ]
        update = acp.update_plan(entries)
        await self._conn.session_update(session_id=session.id, update=update)

    # ------------------------------------------------------------------
    # Permission system
    # ------------------------------------------------------------------

    async def _check_permission(
        self,
        session: Session,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check whether a tool call is permitted under the session's mode.

        Returns ``(True, "")`` if permitted, or ``(False, reason)`` if denied.
        The reason is sent back to the model as the tool result.
        """
        mode = session.permission_mode

        # "bypass" — auto-approve everything
        if mode == "bypass":
            return True, ""

        # "read" — only read-only tools
        if mode == "read":
            if tool_name in DESTRUCTIVE_TOOLS:
                msg = (
                    f"Permission denied: '{tool_name}' is blocked because the "
                    f"session is in read-only mode. Ask the user to switch to "
                    f"Ask or Bypass mode to allow file edits and commands."
                )
                await self._send_message(session.id, f"\n\n⚠️ {msg}\n")
                return False, msg
            return True, ""

        # "ask" — read-only tools auto-approved, destructive tools need approval
        if tool_name not in DESTRUCTIVE_TOOLS:
            return True, ""

        # Request permission via ACP request_permission
        tc_update = acp.update_tool_call(
            tool_call_id=tool_call_id,
            status="pending",
        )
        opts = [
            PermissionOption(
                option_id="allow",
                kind="allow_once",
                name=f"Allow {tool_name}",
            ),
            PermissionOption(
                option_id="reject",
                kind="reject_once",
                name="Deny",
            ),
        ]
        resp = await self._conn.request_permission(
            options=opts,
            session_id=session.id,
            tool_call=tc_update,
        )

        outcome = resp.outcome
        if outcome.outcome == "selected" and outcome.option_id == "allow":
            return True, ""

        # User denied
        msg = f"User denied the request to run '{tool_name}'."
        await self._send_message(session.id, f"\n\n🚫 {msg}\n")
        return False, msg

    async def _send_available_commands(self, session: Session) -> None:
        """Advertise slash commands to the client so Zed shows them in the UI."""
        from acp.helpers import update_available_commands
        from acp.schema import AvailableCommand

        commands = [
            AvailableCommand(
                name="compact",
                description="Manually trigger context compaction — summarize older messages to free up context space",
            ),
            AvailableCommand(
                name="clear-plan",
                description="Clear the current task plan / todo list",
            ),
            AvailableCommand(
                name="clear-history",
                description="Clear all conversation history and start fresh (keeps current model/plan settings)",
            ),
            AvailableCommand(
                name="status",
                description="Show current model, plan, API endpoint, permission mode, and context usage",
            ),
        ]
        update = update_available_commands(commands)
        await self._conn.session_update(session_id=session.id, update=update)

    async def _handle_command(self, session: Session, command: str) -> str:
        """Handle a slash command typed by the user.

        Returns the human-readable response to display.
        """
        command = command.strip()

        if command == "/compact":
            await self._send_message(
                session.id,
                "\n\n_Manually compacting conversation context…_\n\n",
            )
            thought_config = THOUGHT_LEVELS.get(session.thought_level, {})
            base_url = API_ENDPOINTS.get(session.api_endpoint, {}).get(
                "base_url", API_ENDPOINTS[DEFAULT_API_ENDPOINT]["base_url"]
            )
            client = GlmClient(
                session.model,
                thought_level=thought_config.get("thinking_type", "enabled"),
                reasoning_effort=thought_config.get("reasoning_effort"),
                base_url=base_url,
            )
            try:
                await self._maybe_compact(session, client, force=True)
            finally:
                await client.aclose()
            return ""

        elif command == "/clear-plan":
            session.plan = []
            await self._send_plan(session)
            self._store.save(session.id, session.to_dict())
            return "📋 Task plan cleared."

        elif command == "/clear-history":
            system_msg = session.messages[0] if session.messages and session.messages[0].get("role") == "system" else None
            session.messages = [system_msg] if system_msg else []
            session.plan = []
            session.estimated_tokens = 0
            session.last_reported_tokens = -1
            await self._send_plan(session)
            await self._report_usage(session)
            self._store.save(session.id, session.to_dict())
            return "🧹 Conversation history cleared."

        elif command == "/status":
            n_msgs = len(session.messages)
            return (
                f"\n📊 **Session Status**\n"
                f"- **Model:** {session.model}\n"
                f"- **API Plan:** {API_ENDPOINTS.get(session.api_endpoint, {}).get('name', session.api_endpoint)}\n"
                f"- **Reasoning:** {session.thought_level}\n"
                f"- **Permissions:** {session.permission_mode}\n"
                f"- **Context:** {session.estimated_tokens:,} / {session.context_size:,} tokens "
                f"({session.estimated_tokens * 100 // max(session.context_size, 1)}%)\n"
                f"- **Messages:** {n_msgs}\n"
                f"- **Plan tasks:** {len(session.plan)}\n"
            )

        else:
            return f"Unknown command: {command}\nAvailable commands: /compact, /clear-plan, /clear-history, /status"

    # ------------------------------------------------------------------
    # Context compaction
    # ------------------------------------------------------------------

    async def _maybe_compact(self, session: Session, client: GlmClient, force: bool = False) -> None:
        """Trigger context compaction if estimated usage exceeds threshold.

        Mirrors Claude Code's approach: when the conversation approaches the
        context window limit, summarize older messages into a compact summary
        block while preserving the most recent N messages verbatim.

        If *force* is True, compaction runs regardless of threshold (used by
        the /compact slash command).
        """
        threshold_tokens = int(session.context_size * COMPACTION_THRESHOLD)

        # Always estimate first
        session.estimated_tokens = self._estimate_tokens(session.messages)

        if not force and session.estimated_tokens <= threshold_tokens:
            return

        logger.info(
            "Compacting context: %d estimated tokens exceeds %d threshold (%.0f%% of %d)",
            session.estimated_tokens,
            threshold_tokens,
            COMPACTION_THRESHOLD * 100,
            session.context_size,
        )

        # Notify the user that compaction is happening (skip if forced —
        # the /compact command already shows its own message)
        if not force:
            await self._send_message(
                session.id,
                "\n\n_Compacting conversation context…_\n\n",
            )

        # --- Identify the system prompt ---
        messages = session.messages
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

        # --- Partition: summarize everything except system + recent ---
        # Use index 0 as the system message reference — avoid identity check
        # (is) since deserialized messages are different objects.
        compactable = messages[1:] if messages and messages[0].get("role") == "system" else messages
        if len(compactable) <= COMPACTION_KEEP_RECENT:
            return  # not enough to compact

        to_summarize = compactable[:-COMPACTION_KEEP_RECENT]
        keep_recent = compactable[-COMPACTION_KEEP_RECENT:]

        # --- Adjust boundary so we don't split tool-call ↔ tool-result pairs ---
        # If the first kept message is a tool result, its corresponding
        # assistant tool_call was summarized away — move it to summarize too.
        # Conversely, if the last summarized message has tool_calls, those
        # tool results must also be summarized (not kept).
        while keep_recent and keep_recent[0].get("role") == "tool":
            to_summarize.append(keep_recent.pop(0))

        # --- Summarize ---
        summary = await client.summarize_messages(to_summarize)

        # --- Rebuild message list ---
        new_messages: list[dict[str, Any]] = []
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append({
            "role": "user",
            "content": _COMPACTION_MARKER_OPEN
                       + summary
                       + _COMPACTION_MARKER_CLOSE,
        })
        new_messages.extend(keep_recent)

        session.messages = new_messages

        # Update token estimate after compaction
        session.estimated_tokens = self._estimate_tokens(session.messages)
        session.last_reported_tokens = -1  # force re-report
        await self._report_usage(session)

        logger.info(
            "Compaction complete: %d messages → %d messages, ~%d tokens",
            len(messages),
            len(new_messages),
            session.estimated_tokens,
        )

    def _extract_text(self, prompt: list[Any]) -> str:
        parts = []
        for block in prompt:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "resource":
                    res = block.get("resource", {})
                    uri = res.get("uri", "")
                    text = res.get("text", "")
                    parts.append(f"[File: {uri}]\n{text}")
            elif isinstance(block, str):
                parts.append(block)
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "\n".join(parts) if parts else ""

    def _extract_prompt_parts(
        self, prompt: list[Any]
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract text and image data from an ACP prompt list.

        Returns ``(text, images)`` where ``images`` is a list of dicts
        with ``data`` (base64) and ``mime_type`` keys.
        """
        text_parts: list[str] = []
        images: list[dict[str, str]] = []

        for block in prompt:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)

            if btype == "text":
                txt = getattr(block, "text", None) or (block.get("text", "") if isinstance(block, dict) else "")
                if txt:
                    text_parts.append(txt)
            elif btype == "image":
                data = getattr(block, "data", None) or (block.get("data") if isinstance(block, dict) else None)
                mime = getattr(block, "mime_type", None) or (
                    block.get("mime_type") if isinstance(block, dict) else None
                ) or "image/png"
                if data:
                    images.append({"data": data, "mime_type": mime})
            elif btype == "resource":
                if isinstance(block, dict):
                    res = block.get("resource", {})
                    uri = res.get("uri", "")
                    rtext = res.get("text", "")
                    text_parts.append(f"[File: {uri}]\n{rtext}")
                else:
                    res = getattr(block, "resource", None)
                    if res:
                        text_parts.append(f"[File: {getattr(res, 'uri', '')}]\n{getattr(res, 'text', '')}")
            elif isinstance(block, str):
                text_parts.append(block)
            else:
                txt = getattr(block, "text", None)
                if txt:
                    text_parts.append(txt)

        return "\n".join(text_parts) if text_parts else "", images

    async def _save_images(
        self, session: Session, images: list[dict[str, str]]
    ) -> list[str]:
        """Save pasted images to a temp directory inside the workspace.

        Returns the list of saved file paths.
        """
        import base64
        from datetime import datetime

        img_dir = Path(session.cwd) / ".glm-acp-images"
        img_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, img in enumerate(images):
            ext = "png"
            mime = img.get("mime_type", "image/png")
            if "jpeg" in mime or "jpg" in mime:
                ext = "jpg"
            elif "webp" in mime:
                ext = "webp"
            elif "gif" in mime:
                ext = "gif"

            filename = f"screenshot_{timestamp}_{i}.{ext}"
            filepath = img_dir / filename
            filepath.write_bytes(base64.b64decode(img["data"]))
            saved.append(str(filepath))
            logger.info("Saved pasted image to %s", filepath)

        return saved

    async def _send_thought(self, session_id: str, text: str) -> None:
        chunk = acp.update_agent_thought_text(text)
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _send_message(self, session_id: str, text: str) -> None:
        chunk = acp.update_agent_message_text(text)
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _send_user_message(self, session_id: str, text: str) -> None:
        """Echo a user message back to the UI (for slash commands)."""
        chunk = acp.update_user_message_text(text)
        await self._conn.session_update(session_id=session_id, update=chunk)

    @staticmethod
    def _format_image_prompt(
        images: list[dict[str, str]], saved_paths: list[str]
    ) -> str:
        """Build a content string for vision models that includes image URLs.

        Vision models accept image_url content blocks in the messages API.
        We reference the saved file paths so the model can access them.
        """
        parts = ["\n\n[The user shared the following images:]"]
        for path in saved_paths:
            parts.append(f"\n  - {path}")
        parts.append("\n\nPlease analyze the image(s) above.")
        return "".join(parts)

    async def _send_session_info(self, session: Session) -> None:
        """Notify the client of the session title for its history sidebar."""
        update = SessionInfoUpdate(
            session_update="session_info_update",
            title=session.title,
        )
        await self._conn.session_update(session_id=session.id, update=update)

    async def _replay_history(self, session: Session) -> None:
        """Replay persisted messages as session_update notifications.

        ACP's ``session/load`` and ``session/resume`` responses only carry
        config options and modes — never the message history.  For the
        previous conversation to appear in the editor UI the agent must
        re-send every user/assistant turn via ``user_message_chunk`` /
        ``agent_message_chunk`` notifications, the same channel used during
        a live ``prompt``.

        System messages and tool-call internal bookkeeping are skipped —
        only human-visible turns are replayed.  Each message is sent as a
        single complete chunk (no streaming) so it appears instantly.
        """
        for msg in session.messages:
            role = msg.get("role")
            content = msg.get("content", "") or ""
            if role == "user":
                if not content:
                    continue
                chunk = acp.update_user_message_text(content)
            elif role == "assistant":
                if not content:
                    continue
                chunk = acp.update_agent_message_text(content)
            else:
                # Skip system messages, tool results, and internal entries.
                continue
            await self._conn.session_update(session_id=session.id, update=chunk)

    async def _start_tool(self, session_id: str, tool_call_id: str, name: str) -> None:
        kind = TOOL_KINDS.get(name, "other")
        chunk = acp.start_tool_call(
            tool_call_id=tool_call_id,
            title=self._tool_title(name),
            kind=kind,
            status="pending",
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _update_tool(self, session_id: str, tool_call_id: str, status: str) -> None:
        chunk = acp.update_tool_call(tool_call_id=tool_call_id, status=status)
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _complete_tool(self, session_id: str, tool_call_id: str, output: str) -> None:
        chunk = acp.update_tool_call(
            tool_call_id=tool_call_id,
            status="completed",
            content=[acp.tool_content(acp.text_block(output[:8000]))],
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _fail_tool(self, session_id: str, tool_call_id: str, error: str) -> None:
        chunk = acp.update_tool_call(
            tool_call_id=tool_call_id,
            status="failed",
            content=[acp.tool_content(acp.text_block(error[:2000]))],
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

    def _tool_title(self, name: str) -> str:
        return {
            "read_file": "Reading file",
            "write_file": "Writing file",
            "edit_file": "Editing file",
            "list_directory": "Listing directory",
            "search_files": "Searching files",
            "grep": "Searching code",
            "run_command": "Running command",
            "update_plan": "Updating plan",
        }.get(name, name)

    def _build_model_option(self, session: Session) -> SessionConfigOptionSelect:
        plan_models = models_for_plan(session.api_endpoint)
        # If the current model isn't in this plan, still show it as selected
        # (the set_config_option handler will re-validate on switch)
        return SessionConfigOptionSelect(
            id="model",
            name="Model",
            description="GLM model to use",
            category="model",
            type="select",
            current_value=session.model,
            options=[
                SessionConfigSelectOption(
                    value=model_id,
                    name=info["name"],
                    description=f'{info["description"]} ({info["context_window"]} context)',
                )
                for model_id, info in plan_models.items()
            ],
        )

    def _build_thought_option(self, session: Session) -> SessionConfigOptionSelect:
        levels = thought_levels_for_model(session.model)
        return SessionConfigOptionSelect(
            id="thought_level",
            name="Reasoning",
            description="Live reasoning trace level",
            category="thought_level",
            type="select",
            current_value=session.thought_level,
            options=[
                SessionConfigSelectOption(
                    value=level_id,
                    name=info["name"],
                    description=info["description"],
                )
                for level_id, info in levels.items()
            ],
        )

    def _build_permission_option(self, session: Session) -> SessionConfigOptionSelect:
        return SessionConfigOptionSelect(
            id="permission_mode",
            name="Permissions",
            description="Tool execution permission mode",
            category="permissions",
            type="select",
            current_value=session.permission_mode,
            options=[
                SessionConfigSelectOption(
                    value="ask",
                    name="Ask",
                    description="Approve file edits and commands before they run",
                ),
                SessionConfigSelectOption(
                    value="read",
                    name="Read Only",
                    description="Block all file edits and commands — read-only mode",
                ),
                SessionConfigSelectOption(
                    value="bypass",
                    name="Bypass",
                    description="Auto-approve everything — no prompts",
                ),
            ],
        )

    def _build_api_endpoint_option(self, session: Session) -> SessionConfigOptionSelect:
        return SessionConfigOptionSelect(
            id="api_endpoint",
            name="API Plan",
            description="Z.ai API plan / endpoint",
            category="api_endpoint",
            type="select",
            current_value=session.api_endpoint,
            options=[
                SessionConfigSelectOption(
                    value=endpoint_id,
                    name=info["name"],
                    description=info["description"],
                )
                for endpoint_id, info in API_ENDPOINTS.items()
            ],
        )


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # use_unstable_protocol=True enables session/resume, session/fork,
    # session/close, and session/list which Zed relies on to restore
    # conversations after a restart.
    await acp.run_agent(GlmAcpAgent(), use_unstable_protocol=True)
