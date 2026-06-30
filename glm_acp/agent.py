"""ACP agent implementation for the Z.ai GLM API.

This is the main entry point. The agent speaks the Agent Client Protocol
over stdio, so Zed (or any ACP client) launches it as a subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
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
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    UsageUpdate,
)

from .config import (
    CHARS_PER_TOKEN,
    COMPACTION_KEEP_RECENT,
    COMPACTION_THRESHOLD,
    CONTEXT_WINDOW_TOKENS,
    DEFAULT_MODEL,
    MODELS,
    THOUGHT_LEVELS,
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
        self.title: str | None = None
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
            "title": self.title,
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
        session.title = data.get("title")
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

        config_options = [
            self._build_model_option(session),
            self._build_thought_option(session),
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

        config_options = [
            self._build_model_option(session),
            self._build_thought_option(session),
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

        config_options = [
            self._build_model_option(session),
            self._build_thought_option(session),
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
        elif config_id == "thought_level":
            session.thought_level = str(value)

        self._store.save(session.id, session.to_dict())

        return SetSessionConfigOptionResponse(
            config_options=[
                self._build_model_option(session),
                self._build_thought_option(session),
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
        user_text = self._extract_text(prompt)
        session.messages.append({"role": "user", "content": user_text})

        # Derive a session title from the first user message if we don't
        # have one yet, and tell Zed so it shows up in the history sidebar.
        if not session.title:
            session.title = user_text[:60].strip() or "New Chat"
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
        client = GlmClient(session.model, session.thought_level)
        tools = TOOL_DEFINITIONS if session.mode == "code" else [
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] in ("read_file", "list_directory", "search_files", "grep")
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

                    await self._update_tool(session.id, tc_id, status="in_progress")

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
        """Rough token estimate based on character count heuristic."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total_chars += len(content)
            for tc in msg.get("tool_calls", []):
                total_chars += len(json.dumps(tc.get("function", {}).get("arguments", "")))
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
    # Context compaction
    # ------------------------------------------------------------------

    async def _maybe_compact(self, session: Session, client: GlmClient) -> None:
        """Trigger context compaction if estimated usage exceeds threshold.

        Mirrors Claude Code's approach: when the conversation approaches the
        context window limit, summarize older messages into a compact summary
        block while preserving the most recent N messages verbatim.
        """
        threshold_tokens = int(session.context_size * COMPACTION_THRESHOLD)

        # Always estimate first
        session.estimated_tokens = self._estimate_tokens(session.messages)

        if session.estimated_tokens <= threshold_tokens:
            return

        logger.info(
            "Compacting context: %d estimated tokens exceeds %d threshold (%.0f%% of %d)",
            session.estimated_tokens,
            threshold_tokens,
            COMPACTION_THRESHOLD * 100,
            session.context_size,
        )

        # Notify the user that compaction is happening
        await self._send_message(
            session.id,
            "\n\n_Compacting conversation context…_\n\n",
        )

        # --- Identify the system prompt ---
        messages = session.messages
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

        # --- Partition: summarize everything except system + recent ---
        compactable = [m for m in messages if m is not system_msg]
        if len(compactable) <= COMPACTION_KEEP_RECENT:
            return  # not enough to compact

        to_summarize = compactable[:-COMPACTION_KEEP_RECENT]
        keep_recent = compactable[-COMPACTION_KEEP_RECENT:]

        # --- Summarize ---
        summary = await client.summarize_messages(to_summarize)

        # --- Rebuild message list ---
        new_messages: list[dict[str, Any]] = []
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

    async def _send_thought(self, session_id: str, text: str) -> None:
        chunk = acp.update_agent_thought_text(text)
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _send_message(self, session_id: str, text: str) -> None:
        chunk = acp.update_agent_message_text(text)
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _send_session_info(self, session: Session) -> None:
        """Notify the client of the session title for its history sidebar."""
        update = SessionInfoUpdate(
            session_update="session_info_update",
            title=session.title,
        )
        await self._conn.session_update(session_id=session.id, update=update)

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
        }.get(name, name)

    def _build_model_option(self, session: Session) -> SessionConfigOptionSelect:
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
                for model_id, info in MODELS.items()
            ],
        )

    def _build_thought_option(self, session: Session) -> SessionConfigOptionSelect:
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
                for level_id, info in THOUGHT_LEVELS.items()
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
