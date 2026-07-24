"""Standalone terminal frontend for the exact Native GLM ACP agent runtime."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import acp
from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

from .agent import GlmAcpAgent
from .config import API_ENDPOINTS, GENERATION_PROFILES, MODELS, THOUGHT_LEVELS


def add_chat_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "chat",
        help="run Native GLM ACP as a standalone interactive or one-shot coding agent",
    )
    parser.add_argument(
        "--cwd", default=os.getcwd(), help="workspace root (default: current directory)"
    )
    parser.add_argument("--resume", metavar="SESSION_ID", help="resume a persisted session")
    parser.add_argument("--prompt", help="run one prompt and exit")
    parser.add_argument("--stdin", action="store_true", help="read one prompt from standard input")
    parser.add_argument("--image", action="append", default=[], help="attach an image path")
    parser.add_argument("--additional-dir", action="append", default=[])
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--thought-level", choices=sorted(THOUGHT_LEVELS))
    parser.add_argument("--api-endpoint", choices=sorted(API_ENDPOINTS))
    parser.add_argument("--permission", choices=("ask", "read", "bypass"))
    parser.add_argument("--generation-profile", choices=sorted(GENERATION_PROFILES))
    parser.add_argument("--auxiliary-model")
    parser.add_argument("--mixture-mode", choices=("off", "enabled"))
    parser.add_argument("--mode", choices=("ask", "code"))
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSONL events")
    parser.add_argument("--no-thinking", action="store_true", help="hide streamed reasoning")
    parser.add_argument(
        "--plain", action="store_true", help="use the line-oriented REPL instead of the full TUI"
    )


class TerminalClient:
    """ACP Client implementation that renders notifications in a terminal."""

    def __init__(
        self,
        *,
        as_json: bool = False,
        show_thinking: bool = True,
        interactive: bool = True,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.as_json = as_json
        self.show_thinking = show_thinking
        self.interactive = interactive
        self.input_fn = input_fn
        self.last_message = ""
        self._stream_kind = ""
        self.replaying = False
        self._tool_titles: dict[str, str] = {}

    def _emit_json(self, session_id: str, update: Any) -> None:
        payload = update.model_dump(by_alias=True, exclude_none=True)
        print(
            json.dumps({"sessionId": session_id, "update": payload}, ensure_ascii=False), flush=True
        )

    @staticmethod
    def _content_text(update: Any) -> str:
        content = getattr(update, "content", None)
        return str(getattr(content, "text", ""))

    def _break_stream(self) -> None:
        if self._stream_kind:
            if self._stream_kind == "thinking" and not self.as_json:
                print("\033[0m", end="", file=sys.stdout)
            print(file=sys.stdout, flush=True)
            self._stream_kind = ""

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if self.as_json:
            self._emit_json(session_id, update)
            return
        kind = str(getattr(update, "session_update", ""))
        if kind == "agent_message_chunk":
            text = self._content_text(update)
            if self._stream_kind not in {"", "message"}:
                self._break_stream()
            if self.replaying and self._stream_kind != "message":
                print("assistant(history)> ", end="", flush=True)
            self._stream_kind = "message"
            self.last_message += text
            print(text, end="", flush=True)
        elif kind == "user_message_chunk" and self.replaying:
            self._break_stream()
            print(f"you(history)> {self._content_text(update)}", flush=True)
        elif kind == "agent_thought_chunk" and self.show_thinking:
            text = self._content_text(update)
            if self._stream_kind not in {"", "thinking"}:
                self._break_stream()
            if self._stream_kind != "thinking":
                print("\033[2mthinking> ", end="", flush=True)
            self._stream_kind = "thinking"
            print(text, end="", flush=True)
        elif kind in {"tool_call", "tool_call_update"}:
            self._break_stream()
            title = getattr(update, "title", None)
            status = getattr(update, "status", None)
            tool_call_id = str(getattr(update, "tool_call_id", ""))
            if title:
                if tool_call_id:
                    self._tool_titles[tool_call_id] = str(title)
                print(f"[{status or 'tool'}] {title}", file=sys.stderr, flush=True)
        elif kind == "plan":
            self._break_stream()
            entries = getattr(update, "entries", [])
            print("Plan:", file=sys.stderr)
            for entry in entries:
                print(
                    f"  [{getattr(entry, 'status', 'pending')}] {getattr(entry, 'content', '')}",
                    file=sys.stderr,
                )
        elif kind == "usage_update":
            self._break_stream()
            print(
                f"[context {getattr(update, 'used', 0):,}/{getattr(update, 'size', 0):,}]",
                file=sys.stderr,
            )

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **kwargs: Any
    ) -> RequestPermissionResponse:
        tool_call_id = str(getattr(tool_call, "tool_call_id", ""))
        title = (
            getattr(tool_call, "title", None)
            or self._tool_titles.get(tool_call_id)
            or "requested tool"
        )
        raw_input = getattr(tool_call, "raw_input", None)
        detail = self._permission_detail(raw_input)
        if not self.interactive:
            print(f"Permission denied (non-interactive): {title}{detail}", file=sys.stderr)
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        self._break_stream()
        try:
            answer = await asyncio.to_thread(self.input_fn, f"Allow {title}{detail}? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        allow = next((option for option in options if option.option_id == "allow"), None)
        if answer.strip().lower() in {"y", "yes"} and allow is not None:
            return RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", option_id=allow.option_id)
            )
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    @staticmethod
    def _permission_detail(raw_input: Any) -> str:
        if not isinstance(raw_input, dict):
            return ""
        secret_terms = ("key", "token", "secret", "password", "credential")
        safe: dict[str, Any] = {}
        for key, value in raw_input.items():
            normalized = str(key).lower()
            if any(term in normalized for term in secret_terms):
                safe[str(key)] = "[REDACTED]"
            elif normalized in {"content", "new_text", "old_text", "patch"}:
                safe[str(key)] = f"[{len(str(value))} characters]"
            elif isinstance(value, (str, int, float, bool)) or value is None:
                safe[str(key)] = TerminalClient._redact_permission_value(value)
            elif normalized in {"steps", "files", "paths"}:
                safe[str(key)] = f"[{len(value) if isinstance(value, list) else 'structured'}]"
        if not safe:
            return ""
        rendered = json.dumps(safe, ensure_ascii=False)
        return f" {rendered[:2000]}"

    @staticmethod
    def _redact_permission_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = re.sub(
            r"(?i)(\bauthorization\s*[:=]\s*)"
            r"(?:bearer\s+[^\s'\";]+|[^\s'\";]+)",
            r"\1[REDACTED]",
            value,
        )
        text = re.sub(
            r"(?i)(\b(?:api[_-]?key|token|secret|password|credential|authorization)\b"
            r"\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s]+)",
            r"\1[REDACTED]",
            text,
        )
        return re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer [REDACTED]", text)

    def finish_turn(self) -> str:
        self._break_stream()
        message = self.last_message
        self.last_message = ""
        return message


def _prompt_blocks(text: str, image_paths: list[str]) -> list[Any]:
    blocks: list[Any] = []
    if text:
        blocks.append(acp.text_block(text))
    for raw in image_paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Image does not exist: {path}")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError(f"Image exceeds 20 MiB: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not mime_type.startswith("image/"):
            raise ValueError(f"Not a recognized image: {path}")
        blocks.append(
            acp.image_block(base64.b64encode(path.read_bytes()).decode("ascii"), mime_type)
        )
    return blocks


async def _configure(agent: GlmAcpAgent, session_id: str, args: argparse.Namespace) -> None:
    values = [
        ("api_endpoint", args.api_endpoint),
        ("model", args.model),
        ("thought_level", args.thought_level),
        ("permission_mode", args.permission),
        ("generation_profile", args.generation_profile),
        ("auxiliary_model", args.auxiliary_model),
        ("mixture_mode", args.mixture_mode),
    ]
    for config_id, value in values:
        if value is not None:
            await agent.set_config_option(config_id=config_id, session_id=session_id, value=value)
    if args.mode:
        await agent.set_session_mode(mode_id=args.mode, session_id=session_id)


async def _handle_plain_command(
    text: str,
    agent: GlmAcpAgent,
    session_id: str,
    session: Any,
    pending_images: list[str],
) -> str | None:
    """Handle plain-mode slash commands.

    Returns:
      - ``"break"`` if the loop should exit (e.g. /exit, /quit)
      - ``"skip"`` if the line was handled locally and should NOT be sent to the model
      - ``None`` if the line (possibly mutated for /planmode) should be sent to the model
      - the (possibly rewritten) text itself if a slash command transformed it
    """
    stripped = text.strip()
    if stripped in {"/exit", "/quit"}:
        return "break"
    if text.startswith("/image "):
        pending_images.append(text.partition(" ")[2].strip())
        return "skip"
    if stripped == "/help":
        print(
            "Available commands in plain mode:\n"
            "  /exit, /quit          End the session\n"
            "  /image <path>         Queue an image for the next prompt\n"
            "  /max-iterations [N]   Show or set the per-turn tool-call cap\n"
            "                        (default 50, max 1000; e.g. /max-iterations 200)\n"
            "  /planmode <PRD>       Activate read-only Plan Mode\n"
            "  Anything else is sent to the model as a prompt.",
            file=sys.stderr,
        )
        return "skip"
    if stripped == "/max-iterations" or text.startswith("/max-iterations "):
        arg = text.partition(" ")[2].strip()
        current = getattr(session, "max_tool_iterations", 50)
        if not arg:
            print(
                f"Current tool-call iteration cap: {current} per turn "
                "(use /max-iterations <N> to set)",
                file=sys.stderr,
            )
            return "skip"
        try:
            new_cap = int(arg)
        except ValueError:
            print(f"Invalid value: {arg!r} — must be an integer", file=sys.stderr)
            return "skip"
        # set_config_option signature is (config_id, session_id, value).
        # It clamps to [1, 1000] and persists on session.
        await agent.set_config_option("max_tool_iterations", session_id, str(new_cap))
        actual = session.max_tool_iterations
        print(f"Tool-call iteration cap: {current} → {actual}", file=sys.stderr)
        return "skip"
    if text.startswith("/planmode "):
        prd = text.partition(" ")[2].strip()
        if prd:
            await agent.set_session_mode(mode_id="plan", session_id=session_id)
            print("Plan Mode activated — read-only research mode", file=sys.stderr)
            return prd  # the PRD becomes the next prompt
    # UX safeguard: catch the common "forgot the leading slash" typo. If the
    # user typed a known command name without ``/`` (e.g. ``max-iterations 200``
    # instead of ``/max-iterations 200``), show a short hint instead of
    # silently sending the text to the model as a chat prompt.
    suggestion = _suggest_command_if_close(stripped)
    if suggestion is not None:
        print(suggestion, file=sys.stderr)
        return "skip"
    return None


# Known plain-mode commands, used for the "did you mean?" typo hint.
_PLAIN_COMMAND_NAMES = (
    "/exit",
    "/quit",
    "/image",
    "/help",
    "/max-iterations",
    "/planmode",
)


def _suggest_command_if_close(text: str) -> str | None:
    """Return a ``did you mean?`` hint if ``text`` looks like a command
    that's missing its leading slash, else ``None``.

    Triggers on inputs that start with one of the known command names
    (with or without a leading slash, with or without arguments):
    ``max-iterations 200`` → hint to use ``/max-iterations 200``
    ``planmode build me an app`` → hint to use ``/planmode build me an app``
    ``help`` → hint to use ``/help``
    """
    if not text or text.startswith("/"):
        return None
    # Normalize: strip a stray leading slash if the user typed two, etc.
    first_word = text.split(None, 1)[0].lower()
    for name in _PLAIN_COMMAND_NAMES:
        bare = name.lstrip("/").lower()
        if first_word == bare:
            return (
                f"Unknown input. Did you mean '{name}'? "
                "(commands must start with a leading '/')."
            )
    return None


async def run_chat(args: argparse.Namespace) -> int:
    cwd = str(Path(args.cwd).expanduser().resolve())
    if not Path(cwd).is_dir():
        print(f"Workspace does not exist: {cwd}", file=sys.stderr)
        return 2
    if args.prompt is not None and args.stdin:
        print("--prompt and --stdin cannot be used together", file=sys.stderr)
        return 2
    one_shot = args.prompt is not None or args.stdin
    interactive = not one_shot and sys.stdin.isatty()
    if not one_shot and not interactive:
        print("Use --stdin when standard input is not a terminal.", file=sys.stderr)
        return 2

    client = TerminalClient(
        as_json=args.as_json,
        show_thinking=not args.no_thinking,
        interactive=interactive,
    )
    agent = GlmAcpAgent()
    agent.on_connect(client)
    try:
        await agent.initialize(protocol_version=1, client_info={"name": "glm-acp-chat"})
        if args.resume:
            client.replaying = interactive
            await agent.resume_session(
                cwd=cwd,
                session_id=args.resume,
                additional_directories=args.additional_dir,
            )
            session_id = args.resume
            client.replaying = False
            client.finish_turn()
        else:
            response = await agent.new_session(
                cwd=cwd,
                additional_directories=args.additional_dir,
            )
            session_id = response.session_id
        if one_shot and args.permission is None:
            args.permission = "ask"
        await _configure(agent, session_id, args)
        session = agent._sessions[session_id]
        if args.as_json:
            print(
                json.dumps(
                    {
                        "event": "session",
                        "sessionId": session_id,
                        "model": session.model,
                        "permission": session.permission_mode,
                    }
                ),
                flush=True,
            )
        else:
            print(
                f"Native GLM ACP session {session_id}\nWorkspace: {cwd}\n"
                f"Model: {session.model} · Permissions: {session.permission_mode}\n"
                "Type /help for harness commands (including /max-iterations and /planmode); "
                "/exit quits. Ctrl-C cancels a turn.",
                file=sys.stderr,
            )
            if session.permission_mode == "bypass":
                print("WARNING: Bypass mode allows tools without approval.", file=sys.stderr)

        async def submit(text: str, images: list[str] | None = None) -> None:
            client.last_message = ""
            task = asyncio.create_task(
                agent.prompt(
                    prompt=_prompt_blocks(text, images or []),
                    session_id=session_id,
                    message_id=str(uuid4()),
                )
            )
            try:
                await task
            except asyncio.CancelledError:
                await agent.cancel(session_id=session_id)
                raise
            finally:
                client.finish_turn()

        if one_shot:
            text = args.prompt if args.prompt is not None else sys.stdin.read()
            await submit(text, args.image)
            return 0

        pending_images = list(args.image)
        while True:
            try:
                text = await asyncio.to_thread(input, "you> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                print(file=sys.stderr)
                continue
            decision = await _handle_plain_command(
                text, agent, session_id, session, pending_images
            )
            if decision == "break":
                break
            if decision == "skip":
                continue
            if isinstance(decision, str):
                # /planmode rewrites the text to the PRD before submitting.
                text = decision
            try:
                await submit(text, pending_images)
                pending_images = []
            except KeyboardInterrupt:
                await agent.cancel(session_id=session_id)
                print("Turn cancelled.", file=sys.stderr)
                continue
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Native GLM ACP chat failed: {error}", file=sys.stderr)
        return 1
    finally:
        await agent.aclose()


def run_chat_command(args: argparse.Namespace) -> int:
    if (
        args.prompt is None
        and not args.stdin
        and not args.as_json
        and not args.plain
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    ):
        from .tui import run_tui_command

        args.cwd = str(Path(args.cwd).expanduser().resolve())
        return run_tui_command(args)
    try:
        return asyncio.run(run_chat(args))
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
