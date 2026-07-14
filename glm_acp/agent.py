"""ACP agent implementation for the Z.ai GLM API.

This is the main entry point. The agent speaks the Agent Client Protocol
over stdio, so Zed (or any ACP client) launches it as a subprocess.
"""

from __future__ import annotations

import asyncio
import copy
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
    AuthenticateResponse,
    CloseSessionResponse,
    ForkSessionResponse,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionOption,
    PromptCapabilities,
    PromptResponse,
    ResumeSessionResponse,
    SessionAdditionalDirectoriesCapabilities,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionForkCapabilities,
    SessionInfo,
    SessionInfoUpdate,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    TerminalAuthMethod,
    UsageUpdate,
)

from . import __version__
from .config import (
    API_ENDPOINTS,
    AUTH_METHOD_ID,
    COMPACTION_KEEP_RECENT,
    COMPACTION_THRESHOLD,
    CONTEXT_WINDOW_TOKENS,
    DEFAULT_API_ENDPOINT,
    DEFAULT_GENERATION_PROFILE,
    DEFAULT_MODEL,
    DESTRUCTIVE_TOOLS,
    GENERATION_PROFILES,
    MAX_REPEATED_TOOL_BATCHES,
    MAX_TOOL_ITERATIONS,
    MODELS,
    THOUGHT_LEVELS,
    VISION_MODELS,
    has_api_key,
    models_for_plan,
    persist_reasoning,
    thought_levels_for_model,
)
from .glm_client import GlmClient
from .mcp import MCP_TOOL_DEFINITIONS, McpError, McpManager
from .memory import project_knowledge
from .session_store import SessionStore
from .tools import (
    MAX_TOOL_OUTPUT_CHARS,
    TOOL_DEFINITIONS,
    TOOL_KINDS,
    Sandbox,
    ToolError,
    ToolResult,
    execute_tool,
)

logger = logging.getLogger("glm_acp")

SYSTEM_PROMPT_TEMPLATE = """\
You are {model_name}, an expert software engineer working inside the user's \
editor via ACP. You have tools to read, write, and edit files, search code, \
and run commands.

Rules:
- Before editing, locate and obey every applicable AGENTS.md or contributor \
instruction from the repository root to the target file.
- Read files before editing them. Understand the codebase structure first.
- Make minimal, surgical changes. Match existing conventions.
- Preserve unrelated user changes and avoid unrelated refactors.
- Use run_command for builds, tests, and git operations.
- When writing or editing files, briefly explain what you're changing.
- If a task spans many files, work through them systematically.
- Verify changed behavior with the narrowest relevant checks. Do not claim a \
change is complete or working when verification was not run or failed; report \
unverified work and remaining risk explicitly.

Planning:
- For any task with 3+ steps, call update_plan FIRST to lay out your plan.
- Keep tasks concise (one action per entry). Break large work into sub-tasks.
- Update the plan as you work: mark the current task 'in_progress', mark \
finished ones 'completed'.
- The plan helps the user track your progress — keep it accurate.
- For simple questions or single-step tasks, skip the plan.

Project context:
{project_context}

Loaded project instructions and opt-in memory:
{project_knowledge}
"""


def build_system_prompt(cwd: str, model: str = DEFAULT_MODEL) -> str:
    """Build the system prompt with auto-detected project context and model identity."""

    model_name = MODELS.get(model, {}).get("name", model)
    context_parts: list[str] = []

    # Detect language / framework
    cwd_path = Path(cwd)
    try:
        files = {f.name for f in cwd_path.iterdir() if f.is_file()} if cwd_path.exists() else set()
    except (PermissionError, OSError):
        files = set()

    if "pyproject.toml" in files:
        context_parts.append("- Python project (pyproject.toml detected)")
    elif "setup.py" in files or "setup.cfg" in files:
        context_parts.append("- Python project (setup.py detected)")
    elif "package.json" in files:
        try:
            pkg = json.loads((cwd_path / "package.json").read_text())
            lang = (
                "TypeScript"
                if "typescript" in json.dumps(pkg.get("devDependencies", {}))
                else "JavaScript"
            )
            framework = pkg.get("dependencies", {})
            if "next" in framework:
                context_parts.append(f"- {lang} project (Next.js)")
            elif "react" in framework:
                context_parts.append(f"- {lang} project (React)")
            elif "vue" in framework:
                context_parts.append(f"- {lang} project (Vue)")
            else:
                context_parts.append(f"- {lang} project (Node.js)")
        except (json.JSONDecodeError, OSError):
            context_parts.append("- Node.js project (package.json)")
    elif "Cargo.toml" in files:
        context_parts.append("- Rust project (Cargo.toml)")
    elif "go.mod" in files:
        context_parts.append("- Go project (go.mod)")
    elif "Gemfile" in files:
        context_parts.append("- Ruby project (Gemfile)")
    elif "pom.xml" in files or "build.gradle" in files:
        context_parts.append("- Java project (Maven/Gradle)")

    # Detect package manager
    if "uv.lock" in files:
        context_parts.append("- Package manager: uv")
    elif "poetry.lock" in files:
        context_parts.append("- Package manager: Poetry")
    elif "yarn.lock" in files:
        context_parts.append("- Package manager: Yarn")
    elif "pnpm-lock.yaml" in files:
        context_parts.append("- Package manager: pnpm")
    elif "package-lock.json" in files:
        context_parts.append("- Package manager: npm")

    # Detect git
    try:
        git_dir = cwd_path / ".git"
        if git_dir.exists():
            context_parts.append("- Version control: git")
    except (PermissionError, OSError):
        pass

    project_context = "\n".join(context_parts) if context_parts else "(no project files detected)"
    knowledge = project_knowledge(cwd) or "(none loaded; inspect nested instructions before edits)"
    return SYSTEM_PROMPT_TEMPLATE.format(
        project_context=project_context,
        project_knowledge=knowledge,
        model_name=model_name,
    )


MODE_LIST = [
    SessionMode(
        id="ask", name="Ask", description="Answer questions and read files without making changes"
    ),
    SessionMode(
        id="code", name="Code", description="Full access: read, write, edit, and run commands"
    ),
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

# Valid ACP plan entry literals (must match acp.schema PlanEntryStatus/Priority)
_VALID_STATUSES = {"pending", "in_progress", "completed"}
_VALID_PRIORITIES = {"high", "medium", "low"}


def _sanitize_status(value: Any) -> str:
    """Coerce a model-supplied status to a valid ACP PlanEntryStatus."""
    v = str(value).strip().lower()
    # Common synonyms the model might use
    if v in ("done", "finished", "complete"):
        return "completed"
    if v in ("in-progress", "active", "working", "current"):
        return "in_progress"
    if v in ("todo", "not_started", "new"):
        return "pending"
    return v if v in _VALID_STATUSES else "pending"


def _sanitize_priority(value: Any) -> str:
    """Coerce a model-supplied priority to a valid ACP PlanEntryPriority."""
    v = str(value).strip().lower()
    if v in ("critical", "urgent", "p0"):
        return "high"
    if v in ("normal", "default"):
        return "medium"
    if v in ("minor", "low-priority"):
        return "low"
    return v if v in _VALID_PRIORITIES else "medium"


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
        self.generation_profile = DEFAULT_GENERATION_PROFILE
        self.title: str | None = None
        # Permission mode: ask for destructive tools, bypass prompts, or read-only.
        self.permission_mode = "ask"
        # Current task plan — list of PlanEntry-like dicts
        self.plan: list[dict[str, str]] = []
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(cwd, self.model)},
        ]
        # Token tracking — updated after each API call
        self.context_size: int = CONTEXT_WINDOW_TOKENS.get(self.model, 1_000_000)
        self.estimated_tokens: int = 0
        self.last_reported_tokens: int = -1
        # Cumulative cost tracking per session
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cached_tokens: int = 0
        # Runtime-only state. These fields are intentionally not persisted.
        self.prompt_lock = asyncio.Lock()
        self.client: GlmClient | None = None
        self.client_key: tuple[str, str, str, str | None, float | None, float | None] | None = None

    def refresh_system_prompt(self) -> None:
        """Keep the managed system prompt aligned with the selected model."""
        prompt = {"role": "system", "content": build_system_prompt(self.cwd, self.model)}
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = prompt
        else:
            self.messages.insert(0, prompt)

    # ------------------------------------------------------------------
    # Serialization for persistence across process restarts
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the conversation state for on-disk storage."""
        messages = self.messages
        if not persist_reasoning():
            messages = [
                {key: value for key, value in message.items() if key != "reasoning_content"}
                for message in self.messages
            ]
        return {
            "version": 1,
            "cwd": self.cwd,
            "model": self.model,
            "thought_level": self.thought_level,
            "mode": self.mode,
            "api_endpoint": self.api_endpoint,
            "generation_profile": self.generation_profile,
            "title": self.title,
            "permission_mode": self.permission_mode,
            "plan": self.plan,
            "messages": messages,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "estimated_tokens": self.estimated_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], session_id: str) -> Session:
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
        session.generation_profile = data.get("generation_profile", DEFAULT_GENERATION_PROFILE)
        session.plan = data.get("plan", [])
        session.title = data.get("title")
        session.permission_mode = data.get("permission_mode", "ask")
        session.total_input_tokens = data.get("total_input_tokens", 0)
        session.total_output_tokens = data.get("total_output_tokens", 0)
        session.total_cached_tokens = data.get("total_cached_tokens", 0)
        session.estimated_tokens = data.get("estimated_tokens", 0)
        messages = data.get("messages")
        if messages:
            session.messages = messages
        session.context_size = CONTEXT_WINDOW_TOKENS.get(session.model, 1_000_000)
        session.refresh_system_prompt()
        return session


class GlmAcpAgent(acp.Agent):
    _conn: Client
    _sessions: dict[str, Session]

    def __init__(self) -> None:
        self._sessions = {}
        self._store = SessionStore()
        self._tool_io_semaphore = asyncio.Semaphore(4)
        self._mcp = McpManager()

    async def _save_session(self, session: Session) -> None:
        data = copy.deepcopy(session.to_dict())
        await asyncio.to_thread(self._store.save, session.id, data)

    async def _load_stored_session(self, session_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._store.load, session_id)

    def _client_for_session(self, session: Session) -> GlmClient:
        """Return a connection-pooled client matching the session config."""
        thought_config = THOUGHT_LEVELS.get(session.thought_level, {})
        base_url = API_ENDPOINTS.get(session.api_endpoint, {}).get(
            "base_url", API_ENDPOINTS[DEFAULT_API_ENDPOINT]["base_url"]
        )
        profile = GENERATION_PROFILES.get(
            session.generation_profile, GENERATION_PROFILES[DEFAULT_GENERATION_PROFILE]
        )
        key = (
            session.model,
            base_url,
            thought_config.get("thinking_type", "enabled"),
            thought_config.get("reasoning_effort"),
            profile.get("temperature"),
            profile.get("top_p"),
        )
        if session.client is None or session.client_key != key:
            session.client = GlmClient(
                session.model,
                thought_level=key[2],
                reasoning_effort=key[3],
                base_url=base_url,
                temperature=key[4],
                top_p=key[5],
            )
            session.client_key = key
        return session.client

    async def _invalidate_session_client(self, session: Session) -> None:
        client = session.client
        session.client = None
        session.client_key = None
        if client is not None:
            client.cancel()
            await client.aclose()

    async def aclose(self) -> None:
        """Close pooled HTTP clients without deleting persisted sessions."""
        await asyncio.gather(
            *(self._invalidate_session_client(session) for session in self._sessions.values()),
            return_exceptions=True,
        )
        await self._mcp.aclose()

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
                    fork=SessionForkCapabilities(),
                    additional_directories=SessionAdditionalDirectoriesCapabilities(),
                ),
            ),
            agent_info=Implementation(
                name="glm-acp",
                title="Native Z.ai GLM",
                version=__version__,
            ),
            auth_methods=[
                TerminalAuthMethod(
                    id=AUTH_METHOD_ID,
                    name="Configure Z.ai API key",
                    description=(
                        "Open a terminal setup that stores the Z.ai API key "
                        "for future Native GLM ACP sessions."
                    ),
                    type="terminal",
                    args=["--setup"],
                )
            ],
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        if method_id != AUTH_METHOD_ID or not has_api_key():
            return None
        return AuthenticateResponse()

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session = Session(str(uuid4()), cwd, additional_directories)
        self._sessions[session.id] = session
        await self._save_session(session)

        await self._send_available_commands(session)

        config_options = [
            self._build_model_option(session),
            self._build_thought_option(session),
            self._build_api_endpoint_option(session),
            self._build_permission_option(session),
            self._build_generation_profile_option(session),
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
        existing = self._sessions.get(session_id)
        if existing is not None:
            await self._invalidate_session_client(existing)
        data = await self._load_stored_session(session_id)
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
            self._build_generation_profile_option(session),
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
        all_sessions = await asyncio.to_thread(self._store.list)
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
        existing = self._sessions.get(session_id)
        if existing is not None:
            await self._invalidate_session_client(existing)
        data = await self._load_stored_session(session_id)
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
            self._build_generation_profile_option(session),
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
        session = self._sessions.get(session_id)
        if session is not None:
            if session.client is not None:
                session.client.cancel()
            async with session.prompt_lock:
                await self._invalidate_session_client(session)
            self._sessions.pop(session_id, None)
        await asyncio.to_thread(self._store.delete, session_id)
        return CloseSessionResponse()

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        """Fork a session — copy all state to a new session ID.

        The new session gets a copy of the conversation history, plan,
        and config so the user can experiment with a different approach.
        """
        parent = self._sessions.get(session_id)
        if not parent:
            raise RuntimeError(f"Cannot fork: session {session_id} not found")

        new_session = Session(str(uuid4()), cwd, additional_directories)
        # Copy all state from parent
        new_session.model = parent.model
        new_session.thought_level = parent.thought_level
        new_session.mode = parent.mode
        new_session.api_endpoint = parent.api_endpoint
        new_session.generation_profile = parent.generation_profile
        new_session.permission_mode = parent.permission_mode
        new_session.plan = [dict(e) for e in parent.plan]  # deep copy plan entries
        new_session.messages = copy.deepcopy(parent.messages)  # deep copy messages
        new_session.context_size = parent.context_size
        new_session.total_input_tokens = parent.total_input_tokens
        new_session.total_output_tokens = parent.total_output_tokens
        new_session.total_cached_tokens = parent.total_cached_tokens
        new_session.estimated_tokens = parent.estimated_tokens

        self._sessions[new_session.id] = new_session
        await self._save_session(new_session)

        config_options = [
            self._build_model_option(new_session),
            self._build_thought_option(new_session),
            self._build_api_endpoint_option(new_session),
            self._build_permission_option(new_session),
            self._build_generation_profile_option(new_session),
        ]

        return ForkSessionResponse(
            session_id=new_session.id,
            modes=SessionModeState(
                current_mode_id=new_session.mode,
                available_modes=MODE_LIST,
            ),
            config_options=config_options,
        )

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> SetSessionConfigOptionResponse | None:
        session = self._sessions[session_id]
        async with session.prompt_lock:
            return await self._set_config_option_locked(config_id, session_id, value, **kwargs)

    async def _set_config_option_locked(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> SetSessionConfigOptionResponse | None:
        session = self._sessions[session_id]
        previous_client_key = (
            session.model,
            session.thought_level,
            session.api_endpoint,
            session.generation_profile,
        )
        if config_id == "model":
            requested = str(value)
            # Validate that this model is available on the current plan
            if requested not in models_for_plan(session.api_endpoint):
                # Invalid model for this plan — keep current, log warning
                logger.warning("Model %s not available on plan %s", requested, session.api_endpoint)
            else:
                session.model = requested
                session.context_size = CONTEXT_WINDOW_TOKENS.get(session.model, 1_000_000)
                # If the new model doesn't support the current thought level,
                # fall back to the closest supported level.
                if session.thought_level not in thought_levels_for_model(session.model):
                    session.thought_level = (
                        "enabled"
                        if "enabled" in thought_levels_for_model(session.model)
                        else "disabled"
                    )
        elif config_id == "thought_level":
            requested = str(value)
            # Validate that this thought level is available for the current model
            available = thought_levels_for_model(session.model)
            if requested in available:
                session.thought_level = requested
            # Silently ignore invalid levels — keep the current setting
        elif config_id == "api_endpoint":
            session.api_endpoint = str(value)
            # If the current model isn't available on the new plan,
            # fall back to the default model.
            if session.model not in models_for_plan(session.api_endpoint):
                session.model = DEFAULT_MODEL
                session.context_size = CONTEXT_WINDOW_TOKENS.get(session.model, 1_000_000)
                if session.thought_level not in thought_levels_for_model(session.model):
                    session.thought_level = (
                        "enabled"
                        if "enabled" in thought_levels_for_model(session.model)
                        else "disabled"
                    )
        elif config_id == "permission_mode":
            session.permission_mode = str(value)
        elif config_id == "generation_profile":
            requested = str(value)
            if requested in GENERATION_PROFILES:
                session.generation_profile = requested

        current_client_key = (
            session.model,
            session.thought_level,
            session.api_endpoint,
            session.generation_profile,
        )
        if current_client_key != previous_client_key:
            session.refresh_system_prompt()
            await self._invalidate_session_client(session)

        await self._save_session(session)

        return SetSessionConfigOptionResponse(
            config_options=[
                self._build_model_option(session),
                self._build_thought_option(session),
                self._build_api_endpoint_option(session),
                self._build_permission_option(session),
                self._build_generation_profile_option(session),
            ],
        )

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> Any:
        session = self._sessions[session_id]
        async with session.prompt_lock:
            return await self._set_session_mode_locked(mode_id, session_id, **kwargs)

    async def _set_session_mode_locked(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> Any:
        session = self._sessions[session_id]
        # Validate mode_id against known modes
        valid_modes = {m.id for m in MODE_LIST}
        if mode_id not in valid_modes:
            logger.warning("Invalid mode %s — ignoring", mode_id)
            from acp.schema import SetSessionModeResponse

            return SetSessionModeResponse()
        session.mode = mode_id
        await self._save_session(session)
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
        async with session.prompt_lock:
            return await self._prompt_locked(prompt, session_id, message_id=message_id, **kwargs)

    async def _prompt_locked(
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

        # Guard: if content is empty/whitespace-only and there are no
        # images, don't send an empty message to the API.
        if not content.strip() and not images:
            return PromptResponse(stop_reason="end_turn", user_message_id=message_id)

        # Text-only models can't process images. Vision models can.
        is_vision_model = session.model in VISION_MODELS

        if images:
            if is_vision_model:
                # Vision model: store image data inline in the message as
                # multipart content blocks so the API can process them.
                content_blocks: list[dict[str, Any]] = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for img in images:
                    mime = img.get("mime_type", "image/png")
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img['data']}"},
                        }
                    )
                session.messages.append(
                    {
                        "role": "user",
                        "content": content_blocks,
                    }
                )
            else:
                # Text-only model: save to disk, tell the user
                saved_paths = await self._save_images(session, images)
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
        else:
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
            friendly = self._friendly_error(e, session)
            await self._send_message(session.id, f"\n\n**Error:** {friendly}")
            return PromptResponse(stop_reason="end_turn")
        finally:
            # Persist the updated conversation so it survives restarts.
            await self._save_session(session)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel the currently running turn for this session."""
        session = self._sessions.get(session_id)
        if session and session.client is not None:
            session.client.cancel()
            logger.info("Cancellation requested for session %s", session_id)

    @staticmethod
    def _tool_batch_signature(tool_calls: list[dict[str, Any]]) -> str:
        """Return a stable signature that ignores provider-generated call IDs."""
        normalized = [
            {
                "name": call.get("function", {}).get("name", ""),
                "arguments": call.get("function", {}).get("arguments", {}),
            }
            for call in tool_calls
        ]
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _tool_arguments_issue(arguments: Any) -> str | None:
        if not isinstance(arguments, dict):
            return "Tool arguments must be a JSON object. Correct the call and try again."
        if "_raw" in arguments:
            return (
                "Tool arguments contained malformed JSON. Send a valid JSON object using "
                "the tool schema; do not repeat the malformed call."
            )
        return None

    async def _run_turn(self, session: Session) -> str:
        """Execute the full model-turn loop: stream → tool calls → repeat."""
        client = self._client_for_session(session)
        client.begin_turn()
        all_tools = TOOL_DEFINITIONS + MCP_TOOL_DEFINITIONS
        tools = (
            all_tools
            if session.mode == "code"
            else [
                t
                for t in all_tools
                if t["function"]["name"]
                in (
                    "read_file",
                    "list_directory",
                    "search_files",
                    "grep",
                    "recall_memory",
                    "web_search",
                    "web_reader",
                    "mcp_list_tools",
                    "update_plan",
                )
            ]
        )

        previous_tool_batch = ""
        repeated_tool_batches = 0

        for turn_iter in range(MAX_TOOL_ITERATIONS):
            # --- Check for cancellation ---
            if client.cancelled:
                await self._send_message(
                    session.id,
                    "\n\n_⏹ Generation cancelled by user._\n",
                )
                return "end_turn"

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

            if result.tool_calls:
                tool_batch = self._tool_batch_signature(result.tool_calls)
                if tool_batch == previous_tool_batch:
                    repeated_tool_batches += 1
                else:
                    previous_tool_batch = tool_batch
                    repeated_tool_batches = 1

                if repeated_tool_batches >= MAX_REPEATED_TOOL_BATCHES:
                    assistant_msg = {
                        "role": "assistant",
                        "content": result.content or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": json.dumps(tc["function"]["arguments"]),
                                },
                            }
                            for tc in result.tool_calls
                        ],
                    }
                    if getattr(client, "preserve_thinking", False) and result.reasoning:
                        assistant_msg["reasoning_content"] = result.reasoning
                    session.messages.append(assistant_msg)
                    recovery = (
                        "Tool-loop recovery: this exact tool batch was requested "
                        f"{repeated_tool_batches} consecutive times. Inspect prior tool "
                        "results, change the approach, or explain what blocks progress."
                    )
                    for tc in result.tool_calls:
                        await self._fail_tool(session.id, tc["id"], recovery)
                        session.messages.append(
                            {"role": "tool", "tool_call_id": tc["id"], "content": recovery}
                        )
                    if repeated_tool_batches > MAX_REPEATED_TOOL_BATCHES:
                        await self._send_message(
                            session.id,
                            "\n\n⚠️ **Stopped a repeated tool-call loop.** "
                            "Review the last tool result and try a different approach.\n",
                        )
                        logger.warning("Turn ended: repeated tool batch did not recover")
                        return "end_turn"
                    continue

            if result.reasoning and not result.content:
                pass

            if not result.tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": result.content,
                }
                if getattr(client, "preserve_thinking", False) and result.reasoning:
                    assistant_message["reasoning_content"] = result.reasoning
                session.messages.append(assistant_message)
                notice = self._finish_reason_notice(result.finish_reason)
                if notice:
                    await self._send_message(session.id, f"\n\n{notice}\n")
                if result.finish_reason == "cancelled":
                    return "cancelled"
                return "end_turn"

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.content or None}
            if getattr(client, "preserve_thinking", False) and result.reasoning:
                assistant_msg["reasoning_content"] = result.reasoning
            if result.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": json.dumps(tc["function"]["arguments"]),
                        },
                    }
                    for tc in result.tool_calls
                ]
            session.messages.append(assistant_msg)

            # Independent read/search calls can run concurrently. Mixed
            # batches remain ordered so edits and commands retain their
            # original semantics.
            read_only_batch = len(result.tool_calls) > 1 and all(
                TOOL_KINDS.get(tc["function"]["name"]) in {"read", "search"}
                for tc in result.tool_calls
            )
            if read_only_batch:
                for tc in result.tool_calls:
                    await self._start_tool_with_location(
                        session.id,
                        tc["id"],
                        tc["function"]["name"],
                        tc["function"]["arguments"],
                    )
                    await self._update_tool(session.id, tc["id"], status="in_progress")

                async def execute_read(tc: dict[str, Any]):
                    issue = self._tool_arguments_issue(tc["function"]["arguments"])
                    if issue:
                        raise ToolError(issue)
                    async with self._tool_io_semaphore:
                        return await execute_tool(
                            tc["function"]["name"],
                            tc["function"]["arguments"],
                            session.sandbox,
                        )

                read_results = await asyncio.gather(
                    *(execute_read(tc) for tc in result.tool_calls),
                    return_exceptions=True,
                )
                for tc, tool_result in zip(result.tool_calls, read_results):
                    if isinstance(tool_result, BaseException):
                        error_msg = str(tool_result)
                        output = f"Error: {error_msg}"
                        await self._fail_tool(session.id, tc["id"], error_msg)
                    else:
                        output = tool_result.output
                        await self._complete_tool(session.id, tc["id"], tool_result)
                    session.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": output,
                        }
                    )
                continue

            for tc in result.tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = tc["function"]["arguments"]
                tc_id = tc["id"]

                arguments_issue = self._tool_arguments_issue(tool_args)
                if arguments_issue:
                    await self._fail_tool(session.id, tc_id, arguments_issue)
                    session.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": f"Error: {arguments_issue}",
                        }
                    )
                    continue

                # --- Plan tool: handled in-agent, not via sandbox ---
                if tool_name == "update_plan":
                    try:
                        output = await self._handle_update_plan(session, tc_id, tool_args)
                        await self._complete_tool(session.id, tc_id, output)
                    except Exception as e:
                        logger.exception("update_plan failed")
                        output = f"Error updating plan: {e}"
                        await self._fail_tool(session.id, tc_id, output)
                    session.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": output,
                        }
                    )
                    continue

                # Send location update for file tools (enables Zed follow)
                await self._start_tool_with_location(session.id, tc_id, tool_name, tool_args)

                await self._update_tool(session.id, tc_id, status="in_progress")

                # --- Permission check ---
                permitted, deny_reason = await self._check_permission(
                    session,
                    tc_id,
                    tool_name,
                    tool_args,
                )
                if not permitted:
                    output = deny_reason
                    await self._fail_tool(session.id, tc_id, deny_reason)
                    session.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": output,
                        }
                    )
                    continue

                try:
                    if tool_name in {"web_search", "web_reader", "vision_analyze"}:
                        if tool_name == "vision_analyze":
                            tool_args = {
                                **tool_args,
                                "path": str(
                                    session.sandbox.resolve(str(tool_args.get("path", "")))
                                ),
                            }
                        value = await self._mcp.invoke_preset(tool_name, tool_args)
                        tool_result = ToolResult(
                            output=json.dumps(value, ensure_ascii=False)[:MAX_TOOL_OUTPUT_CHARS]
                        )
                    elif tool_name == "mcp_list_tools":
                        value = await self._mcp.list_tools(str(tool_args.get("server", "")))
                        tool_result = ToolResult(
                            output=json.dumps(value, ensure_ascii=False)[:MAX_TOOL_OUTPUT_CHARS]
                        )
                    elif tool_name == "mcp_call":
                        value = await self._mcp.call(
                            str(tool_args.get("server", "")),
                            str(tool_args.get("tool", "")),
                            tool_args.get("arguments", {}),
                        )
                        tool_result = ToolResult(
                            output=json.dumps(value, ensure_ascii=False)[:MAX_TOOL_OUTPUT_CHARS]
                        )
                    else:
                        on_output = None
                        if tool_name == "run_command":

                            async def on_command_output(stream: str, chunk: str) -> None:
                                await self._stream_tool_output(session.id, tc_id, stream, chunk)

                            on_output = on_command_output
                        tool_result = await execute_tool(
                            tool_name,
                            tool_args,
                            session.sandbox,
                            on_output=on_output,
                        )
                    await self._complete_tool(session.id, tc_id, tool_result)
                    output = tool_result.output
                except (ToolError, McpError) as e:
                    error_msg = str(e)
                    output = f"Error: {error_msg}"
                    await self._fail_tool(session.id, tc_id, error_msg)

                session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": output,
                    }
                )
        else:
            # Loop exhausted without break — hit the configured iteration cap
            await self._send_message(
                session.id,
                "\n\n⚠️ **Reached the maximum number of tool-call iterations "
                f"({MAX_TOOL_ITERATIONS}).** "
                "The conversation will continue, but the current task may be incomplete. "
                "Ask me to continue if needed.\n",
            )
            logger.warning("Turn ended: %d-iteration limit reached", MAX_TOOL_ITERATIONS)

        return "end_turn"

    @staticmethod
    def _finish_reason_notice(finish_reason: str) -> str:
        notices = {
            "network_error": (
                "⚠️ The model stream ended because of a network interruption. "
                "The partial response was preserved; ask to continue or retry."
            ),
            "model_context_window_exceeded": (
                "⚠️ The model exceeded its context window. Compact or clear older "
                "history before retrying."
            ),
            "sensitive": (
                "⚠️ The model stopped because the provider's safety filter was triggered."
            ),
            "continuation_limit": (
                "⚠️ The response reached the automatic continuation limit and may be "
                "incomplete. Ask to continue."
            ),
        }
        return notices.get(finish_reason, "")

    # ------------------------------------------------------------------
    # Token estimation & usage reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """Rough token estimate based on character count heuristic.

        Accounts for message content, tool call arguments, tool result
        content, and per-message structural overhead (role tags, etc.).
        Uses a conservative 3.5 chars/token ratio (code is denser than
        natural language, which averages ~4 chars/token).
        """
        chars_per_token = 3.5
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Base64 size is not a useful proxy for provider image-token
                # accounting. Count text exactly and use a conservative fixed
                # allowance per image block.
                text_chars = sum(
                    len(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                image_count = sum(
                    1
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "image_url"
                )
                total_chars += text_chars + image_count * int(chars_per_token * 1024)
                content = ""
            else:
                content = content or ""
            total_chars += len(content)
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                total_chars += len(fn.get("name", ""))
                total_chars += len(fn.get("arguments", ""))
            total_chars += int(chars_per_token * 4)  # per-message overhead
        return int(total_chars // chars_per_token)

    @staticmethod
    def _update_usage(session: Session, usage: dict[str, int] | None) -> None:
        """Update estimated token count. Prefer API-reported values."""
        if usage and usage.get("input_tokens"):
            session.estimated_tokens = usage["input_tokens"]
            session.total_input_tokens += usage.get(
                "api_input_tokens", usage.get("input_tokens", 0)
            )
            session.total_output_tokens += usage.get("output_tokens", 0)
            session.total_cached_tokens += usage.get("cached_tokens", 0)
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
            # Defensive: model may send strings or malformed objects
            if isinstance(task, str):
                task = {"content": task}
            if not isinstance(task, dict):
                continue
            # Sanitize status/priority to valid ACP literals
            entries.append(
                {
                    "content": str(task.get("content", "")),
                    "priority": _sanitize_priority(task.get("priority", "medium")),
                    "status": _sanitize_status(task.get("status", "pending")),
                }
            )

        session.plan = entries
        await self._send_plan(session)
        await self._save_session(session)

        # Return a compact summary as the tool result
        n_pending = sum(1 for e in entries if e["status"] == "pending")
        n_progress = sum(1 for e in entries if e["status"] == "in_progress")
        n_done = sum(1 for e in entries if e["status"] == "completed")
        return (
            f"Plan updated: {len(entries)} tasks "
            f"({n_done} completed, {n_progress} in progress, {n_pending} pending)."
        )

    async def _send_plan(self, session: Session) -> None:
        """Send the current plan as an ACP plan session update.

        Sanitizes status/priority on each entry defensively — entries may
        come from deserialized session data on disk.
        """
        from acp.helpers import plan_entry

        entries = [
            plan_entry(
                content=str(e.get("content", "")),
                priority=_sanitize_priority(e.get("priority", "medium")),
                status=_sanitize_status(e.get("status", "pending")),
            )
            for e in session.plan
            if isinstance(e, dict)
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
        try:
            resp = await self._conn.request_permission(
                options=opts,
                session_id=session.id,
                tool_call=tc_update,
            )
        except Exception as e:
            logger.warning("request_permission failed: %s — defaulting to deny", e)
            msg = f"Could not request permission for '{tool_name}': {e}"
            await self._send_message(session.id, f"\n\n⚠️ {msg}\n")
            return False, msg

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
                description=(
                    "Manually trigger context compaction — summarize older "
                    "messages to free up context space"
                ),
            ),
            AvailableCommand(
                name="clear-plan",
                description="Clear the current task plan / todo list",
            ),
            AvailableCommand(
                name="clear-history",
                description=(
                    "Clear all conversation history and start fresh "
                    "(keeps current model/plan settings)"
                ),
            ),
            AvailableCommand(
                name="diff",
                description="Show a git diff of all changes made during this session",
            ),
            AvailableCommand(
                name="export",
                description="Export the conversation as a Markdown file",
            ),
            AvailableCommand(
                name="status",
                description=(
                    "Show current model, plan, API endpoint, permission mode, and context usage"
                ),
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
            client = self._client_for_session(session)
            try:
                await self._maybe_compact(session, client, force=True)
                return ""
            except Exception as e:
                logger.exception("/compact failed")
                return f"❌ Compaction failed: {e}\nYour conversation is unchanged."
            finally:
                await self._save_session(session)

        elif command == "/clear-plan":
            session.plan = []
            await self._send_plan(session)
            await self._save_session(session)
            return "📋 Task plan cleared."

        elif command == "/clear-history":
            system_msg = (
                session.messages[0]
                if session.messages and session.messages[0].get("role") == "system"
                else None
            )
            session.messages = [system_msg] if system_msg else []
            session.plan = []
            session.estimated_tokens = 0
            session.total_input_tokens = 0
            session.total_output_tokens = 0
            session.total_cached_tokens = 0
            session.last_reported_tokens = -1
            await self._send_plan(session)
            await self._report_usage(session)
            await self._save_session(session)
            return "🧹 Conversation history cleared."

        elif command == "/diff":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "diff",
                    "--stat",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=session.cwd,
                )
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=10)
                if proc.returncode != 0:
                    return "❌ Not a git repository or git error."
                output = stdout_b.decode().strip()
                if not output:
                    return "✅ No uncommitted changes in the working tree."
                # Get full diff too
                proc_full = await asyncio.create_subprocess_exec(
                    "git",
                    "diff",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=session.cwd,
                )
                stdout_f, _ = await asyncio.wait_for(proc_full.communicate(), timeout=10)
                summary = stdout_f.decode().strip()[:4000]
                return (
                    f"\n📝 **Git diff**\n\n```\n{output}\n```\n\n"
                    f"<details><summary>Full diff</summary>\n\n```diff\n{summary}"
                    "\n```\n\n</details>"
                )
            except Exception as e:
                return f"❌ Error running git diff: {e}"

        elif command == "/export":
            from datetime import datetime

            lines: list[str] = [
                "# Conversation Export",
                "",
                f"- **Model:** {session.model}",
                "- **API Plan:** "
                + API_ENDPOINTS.get(session.api_endpoint, {}).get("name", session.api_endpoint),
                f"- **Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- **Total usage:** {session.total_input_tokens:,} input + "
                f"{session.total_output_tokens:,} output tokens",
                f"- **Cached input:** {session.total_cached_tokens:,} tokens",
                "",
                "---",
                "",
            ]
            for msg in session.messages:
                role = msg.get("role", "")
                content = msg.get("content")
                # Normalize: None -> "", list -> extracted text
                if content is None:
                    content = ""
                elif isinstance(content, list):
                    content = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if role == "system":
                    continue
                elif role == "user":
                    lines.append(f"## 👤 User\n\n{content}\n")
                elif role == "assistant":
                    if msg.get("tool_calls"):
                        tc_names = [
                            tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]
                        ]
                        lines.append(
                            f"## 🤖 Assistant _(tools: {', '.join(tc_names)})_\n\n"
                            f"{content or '*(no text)*'}\n"
                        )
                    else:
                        lines.append(f"## 🤖 Assistant\n\n{content}\n")
                elif role == "tool":
                    lines.append(
                        "<details><summary>🔧 Tool result</summary>\n\n```\n"
                        f"{content[:2000]}\n```\n\n</details>\n"
                    )
            md_text = "\n".join(lines)
            export_path = (
                Path(session.cwd)
                / f"conversation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            export_path.write_text(md_text, encoding="utf-8")
            return f"📄 Conversation exported to: `{export_path}`"

        elif command == "/status":
            n_msgs = len(session.messages)
            return (
                f"\n📊 **Session Status**\n"
                f"- **Model:** {session.model}\n"
                "- **API Plan:** "
                + API_ENDPOINTS.get(session.api_endpoint, {}).get("name", session.api_endpoint)
                + "\n"
                f"- **Reasoning:** {session.thought_level}\n"
                f"- **Generation style:** {session.generation_profile}\n"
                f"- **Permissions:** {session.permission_mode}\n"
                f"- **Context:** {session.estimated_tokens:,} / {session.context_size:,} tokens "
                f"({session.estimated_tokens * 100 // max(session.context_size, 1)}%)\n"
                f"- **Messages:** {n_msgs}\n"
                f"- **Plan tasks:** {len(session.plan)}\n"
                f"- **Total usage:** {session.total_input_tokens:,} input + "
                f"{session.total_output_tokens:,} output tokens\n"
                f"- **Cached input:** {session.total_cached_tokens:,} tokens\n"
            )

        else:
            return (
                f"Unknown command: {command}\nAvailable commands: /compact, "
                "/clear-plan, /clear-history, /diff, /export, /status"
            )

    # ------------------------------------------------------------------
    # Context compaction
    # ------------------------------------------------------------------

    async def _maybe_compact(
        self, session: Session, client: GlmClient, force: bool = False
    ) -> None:
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
        if not summary or not summary.strip():
            raise RuntimeError("Compaction summary was empty; conversation is unchanged")

        # --- Rebuild message list ---
        new_messages: list[dict[str, Any]] = []
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append(
            {
                "role": "user",
                "content": _COMPACTION_MARKER_OPEN + summary + _COMPACTION_MARKER_CLOSE,
            }
        )
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
                    parts.append(self._bounded_resource(uri, text))
            elif isinstance(block, str):
                parts.append(block)
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "\n".join(parts) if parts else ""

    def _extract_prompt_parts(self, prompt: list[Any]) -> tuple[str, list[dict[str, str]]]:
        """Extract text and image data from an ACP prompt list.

        Returns ``(text, images)`` where ``images`` is a list of dicts
        with ``data`` (base64) and ``mime_type`` keys.
        """
        text_parts: list[str] = []
        images: list[dict[str, str]] = []

        for block in prompt:
            btype = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )

            if btype == "text":
                txt = getattr(block, "text", None) or (
                    block.get("text", "") if isinstance(block, dict) else ""
                )
                if txt:
                    text_parts.append(txt)
            elif btype == "image":
                data = getattr(block, "data", None) or (
                    block.get("data") if isinstance(block, dict) else None
                )
                mime = (
                    getattr(block, "mime_type", None)
                    or (block.get("mime_type") if isinstance(block, dict) else None)
                    or "image/png"
                )
                if data:
                    images.append({"data": data, "mime_type": mime})
            elif btype == "resource":
                if isinstance(block, dict):
                    res = block.get("resource", {})
                    uri = res.get("uri", "")
                    rtext = res.get("text", "")
                    text_parts.append(self._bounded_resource(uri, rtext))
                else:
                    res = getattr(block, "resource", None)
                    if res:
                        text_parts.append(
                            self._bounded_resource(
                                getattr(res, "uri", ""), getattr(res, "text", "")
                            )
                        )
            elif isinstance(block, str):
                text_parts.append(block)
            else:
                txt = getattr(block, "text", None)
                if txt:
                    text_parts.append(txt)

        return "\n".join(text_parts) if text_parts else "", images

    @staticmethod
    def _bounded_resource(uri: str, text: str) -> str:
        if len(text) <= MAX_TOOL_OUTPUT_CHARS:
            return f"[File: {uri}]\n{text}"
        return (
            f"[File: {uri}]\n{text[:MAX_TOOL_OUTPUT_CHARS]}\n"
            f"... (embedded resource truncated at {MAX_TOOL_OUTPUT_CHARS} characters)"
        )

    async def _save_images(self, session: Session, images: list[dict[str, str]]) -> list[str]:
        """Save pasted images to a temp directory inside the workspace.

        Returns the list of saved file paths.
        """
        return await asyncio.to_thread(self._save_images_sync, session, images)

    @staticmethod
    def _save_images_sync(session: Session, images: list[dict[str, str]]) -> list[str]:
        """Decode and persist pasted images outside the event-loop thread."""
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
            raw_data = img.get("data")
            if not raw_data:
                logger.warning("Image %d has no data — skipping", i)
                continue
            try:
                filepath.write_bytes(base64.b64decode(raw_data))
            except Exception as e:
                logger.warning("Failed to decode/save image %d: %s", i, e)
                continue
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
            content = msg.get("content", "")
            # Handle list content (vision messages with multipart blocks)
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not content:
                continue
            if role == "user":
                chunk = acp.update_user_message_text(content)
            elif role == "assistant":
                chunk = acp.update_agent_message_text(content)
            else:
                # Skip system messages, tool results, and internal entries.
                continue
            await self._conn.session_update(session_id=session.id, update=chunk)

    async def _start_tool(self, session_id: str, tool_call_id: str, name: str) -> None:
        """Send initial tool-call update when streaming begins.

        Location info is NOT available yet (args are streamed incrementally).
        A follow-up _start_tool_with_location call adds the file path once
        the full arguments are parsed.
        """
        kind = TOOL_KINDS.get(name, "other")
        chunk = acp.start_tool_call(
            tool_call_id=tool_call_id,
            title=self._tool_title(name),
            kind=kind,
            status="pending",
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _start_tool_with_location(
        self, session_id: str, tool_call_id: str, name: str, args: dict[str, Any]
    ) -> None:
        """Send an update with file location when a tool starts executing.

        This fires after the tool call is streamed (we now have the full
        args) and before execution begins. Zed uses this to open the file.
        """
        if name not in (
            "read_file",
            "write_file",
            "edit_file",
            "apply_patch",
            "list_directory",
            "vision_analyze",
        ):
            return
        path = args.get("path")
        if not path:
            return
        from acp.schema import ToolCallLocation

        loc_kwargs: dict[str, Any] = {"path": path}
        line = args.get("start_line") or args.get("line")
        if line:
            loc_kwargs["line"] = line
        chunk = acp.update_tool_call(
            tool_call_id=tool_call_id,
            locations=[ToolCallLocation(**loc_kwargs)],
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _update_tool(self, session_id: str, tool_call_id: str, status: str) -> None:
        chunk = acp.update_tool_call(tool_call_id=tool_call_id, status=status)
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _stream_tool_output(
        self, session_id: str, tool_call_id: str, stream: str, text: str
    ) -> None:
        """Publish bounded live command output while the process is running."""
        chunk = acp.update_tool_call(
            tool_call_id=tool_call_id,
            status="in_progress",
            content=[acp.tool_content(acp.text_block(f"{stream}:\n{text[-4000:]}"))],
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _complete_tool(
        self, session_id: str, tool_call_id: str, result: ToolResult | str
    ) -> None:
        # Build content blocks based on tool result type
        if isinstance(result, ToolResult):
            content = []
            # If this is a file edit with diff info, send a diff block
            if result.file_path and result.old_text is not None and result.new_text is not None:
                content.append(
                    acp.tool_diff_content(
                        path=result.file_path,
                        new_text=result.new_text[:8000],
                        old_text=result.old_text[:8000],
                    )
                )
            elif result.file_path and result.new_text is not None:
                # write_file creating a new file (old_text is None)
                content.append(
                    acp.tool_diff_content(
                        path=result.file_path,
                        new_text=result.new_text[:8000],
                        old_text=None,
                    )
                )
            # Always include the text output
            content.append(acp.tool_content(acp.text_block(result.output[:8000])))
        else:
            # Plain string (e.g. plan tool)
            content = [acp.tool_content(acp.text_block(result[:8000]))]

        # Build locations for follow
        locations = None
        if isinstance(result, ToolResult) and result.file_path:
            from acp.schema import ToolCallLocation

            loc_kwargs: dict[str, Any] = {"path": result.file_path}
            if result.line is not None:
                loc_kwargs["line"] = result.line
            locations = [ToolCallLocation(**loc_kwargs)]

        chunk = acp.update_tool_call(
            tool_call_id=tool_call_id,
            status="completed",
            content=content,
            locations=locations,
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

    async def _fail_tool(self, session_id: str, tool_call_id: str, error: str) -> None:
        chunk = acp.update_tool_call(
            tool_call_id=tool_call_id,
            status="failed",
            content=[acp.tool_content(acp.text_block(error[:2000]))],
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

    def _friendly_error(self, error: Exception, session: Session) -> str:
        """Convert raw exceptions into user-friendly error messages."""
        error_str = str(error)

        # GlmApiError with status code
        if hasattr(error, "status_code"):
            code = error.status_code
            if code == 401:
                return (
                    "🔐 **Authentication failed.** Your API key is invalid or expired. "
                    "Get a new key at https://z.ai/ and update `ZAI_API_KEY` in Zed settings."
                )
            elif code == 429:
                return (
                    "⏳ **Rate limited.** The API is throttling requests. "
                    "Wait a moment and try again, or switch to a different model."
                )
            elif code == 1301:
                return (
                    "🚫 **Content filtered.** The Z.ai content filter blocked this request. "
                    "This is often a false positive — try rephrasing your request."
                )
            elif code == 1311:
                return (
                    "💳 **Plan limitation.** Your subscription doesn't include this model. "
                    "Switch to a different API plan or model in the dropdown."
                )
            elif code >= 500:
                return (
                    f"🔧 **Z.ai server error ({code}).** This is a temporary issue on Z.ai's side. "
                    f"The agent already retried — try again in a moment."
                )
            else:
                return f"API error {code}: {error_str[:300]}"

        # Network errors
        if "timeout" in error_str.lower() or "timed out" in error_str.lower():
            return (
                "⏱ **Request timed out.** The API didn't respond in time. "
                "Try again, or reduce the complexity of the request."
            )

        if "connection" in error_str.lower() or "network" in error_str.lower():
            return (
                "📡 **Network error.** Could not reach the Z.ai API. "
                "Check your internet connection and try again."
            )

        # API key missing
        if "ZAI_API_KEY" in error_str:
            return (
                "🔑 **No API key.** Set `ZAI_API_KEY` in the agent server's "
                "`env` block in Zed settings. Get a key at https://z.ai/."
            )

        # Fallback — show the raw error but truncated
        return error_str[:500]

    def _tool_title(self, name: str) -> str:
        return {
            "read_file": "Reading file",
            "write_file": "Writing file",
            "edit_file": "Editing file",
            "apply_patch": "Applying patch",
            "list_directory": "Listing directory",
            "search_files": "Searching files",
            "grep": "Searching code",
            "run_command": "Running command",
            "recall_memory": "Reading project memory",
            "store_memory": "Updating project memory",
            "web_search": "Searching the web",
            "web_reader": "Reading web page",
            "vision_analyze": "Analyzing image",
            "mcp_list_tools": "Listing MCP tools",
            "mcp_call": "Calling MCP tool",
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
                    description=f"{info['description']} ({info['context_window']} context)",
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

    def _build_generation_profile_option(self, session: Session) -> SessionConfigOptionSelect:
        return SessionConfigOptionSelect(
            id="generation_profile",
            name="Generation Style",
            description="Sampling profile; changes only one sampling control at a time",
            category="other",
            type="select",
            current_value=session.generation_profile,
            options=[
                SessionConfigSelectOption(
                    value=profile_id,
                    name=info["name"],
                    description=info["description"],
                )
                for profile_id, info in GENERATION_PROFILES.items()
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
    agent = GlmAcpAgent()
    try:
        await acp.run_agent(agent, use_unstable_protocol=True)
    finally:
        await agent.aclose()
