"""File system and shell tools exposed to the GLM model.

All file operations are sandboxed to the session's working directory and
any additional workspace roots.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory import append_memory, memory_path, read_memory

MAX_TOOL_OUTPUT_CHARS = 64_000
_COMMAND_STREAM_LIMIT = MAX_TOOL_OUTPUT_CHARS // 2
_ALWAYS_IGNORE = [
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".eggs",
]


def _bounded_output(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated at {limit} characters)"


def _run_rg(args: list[str], root: Path) -> str | None:
    """Run ripgrep without a shell, falling back when it is unavailable."""
    rg = shutil.which("rg")
    if not rg:
        return None
    try:
        result = subprocess.run(
            [rg, *args],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode not in {0, 1}:
        return None
    return _bounded_output(result.stdout.rstrip()) or "No matches found"


def _load_gitignore_patterns(root: Path) -> list[str]:
    """Load .gitignore patterns from the workspace root.

    Returns a list of glob patterns.  Simple implementation — handles
    directory names, file globs, and wildcards.
    """
    patterns: list[str] = []
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return patterns
    try:
        for line in gitignore.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    except OSError:
        pass
    return patterns


def _is_ignored(rel_path: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any gitignore pattern."""
    for pattern in patterns:
        # Normalize: remove leading /
        pat = pattern.lstrip("/")
        # If pattern ends with /, it matches directories and everything inside
        if pat.endswith("/"):
            pat = pat[:-1]
        # Direct exact match
        if rel_path == pat:
            return True
        # Match as a parent directory: pat is a prefix of rel_path
        if rel_path.startswith(pat + "/"):
            return True
        # Wildcard glob match on the full relative path
        if fnmatch.fnmatch(rel_path, pat):
            return True
        # Match any path component (e.g. */pat or pat/*)
        if fnmatch.fnmatch(rel_path, f"*/{pat}") or fnmatch.fnmatch(rel_path, f"{pat}/*"):
            return True
        if fnmatch.fnmatch(rel_path, f"*/{pat}/*"):
            return True
    return False


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file. Use absolute or relative paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {
                        "type": "integer",
                        "description": "1-based line to start reading from (optional)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-based line to end reading at (optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file. Creates the file if it does not exist, "
                "overwrites if it does."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Full content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace a specific block of text in a file. Both old_text and "
                "new_text must be exact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_text": {"type": "string", "description": "Exact text to find in the file"},
                    "new_text": {"type": "string", "description": "Text to replace it with"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a validated unified diff to one text file atomically. "
                "Read the file first and include exact context lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "patch": {"type": "string", "description": "Unified diff hunks"},
                },
                "required": ["path", "patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (defaults to cwd)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search for files by glob pattern (e.g. **/*.py). Returns matching paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Root directory (defaults to cwd)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents using a regular expression. Returns matching "
                "lines with file and line number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for",
                    },
                    "path": {"type": "string", "description": "Root directory (defaults to cwd)"},
                    "include": {"type": "string", "description": "Glob filter (e.g. *.py)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command in the working directory. Use for builds, tests, git, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": (
                "Update the task plan / todo list shown to the user. "
                "Call this at the start of complex multi-step tasks to lay out your plan, "
                "then update task statuses as you make progress. "
                "Each call replaces the entire plan — always send the COMPLETE list of tasks. "
                "Set the first task you're about to work on to 'in_progress', mark finished "
                "ones 'completed', and upcoming ones 'pending'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "The complete list of tasks. Replaces the previous plan.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Human-readable description of the task",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Current status of this task",
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": "Importance level (default: medium)",
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["tasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Read opt-in durable knowledge recorded for this project.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": (
                "Store one stable, reusable project fact or user preference. "
                "Do not store secrets, transient task state, or reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {"entry": {"type": "string"}},
                "required": ["entry"],
            },
        },
    },
]

TOOL_KINDS: dict[str, str] = {
    "read_file": "read",
    "write_file": "edit",
    "edit_file": "edit",
    "apply_patch": "edit",
    "list_directory": "read",
    "search_files": "search",
    "grep": "search",
    "run_command": "execute",
    "update_plan": "other",
    "recall_memory": "read",
    "store_memory": "edit",
}


class ToolError(Exception):
    pass


def _read_utf8_text(path: Path, action: str) -> str:
    """Read a UTF-8 text file with platform-independent binary detection."""
    data = path.read_bytes()
    if b"\x00" in data:
        raise ToolError(f"Cannot {action} binary file: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"Cannot {action} binary file: {path}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass
class ToolResult:
    """Structured result from a tool execution.

    Contains the output string, plus optional metadata for ACP tool call
    updates: the file path (for 'follow' in Zed) and old/new text (for
    inline diff rendering).
    """

    output: str
    # Relative or absolute path affected (enables Zed "follow" feature)
    file_path: str | None = None
    # Line number to scroll to (optional)
    line: int | None = None
    # For file edits: the old content before the change (for diff rendering)
    old_text: str | None = None
    # For file edits: the new content after the change (for diff rendering)
    new_text: str | None = None


class Sandbox:
    """Validates that paths stay within allowed workspace roots."""

    def __init__(self, cwd: str, additional_dirs: list[str] | None = None):
        self.roots = [Path(cwd).resolve()]
        if additional_dirs:
            self.roots += [Path(d).resolve() for d in additional_dirs]

    def resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.roots[0] / p
        p = p.resolve()
        for root in self.roots:
            try:
                p.relative_to(root)
                return p
            except ValueError:
                continue
        raise ToolError(f"Path '{path}' is outside the workspace roots")


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    sandbox: Sandbox,
    on_output: Any = None,
) -> ToolResult:
    """Execute a tool call and return a structured result."""
    try:
        if name == "read_file":
            return await _read_file(arguments, sandbox)
        elif name == "write_file":
            return await _write_file(arguments, sandbox)
        elif name == "edit_file":
            return await _edit_file(arguments, sandbox)
        elif name == "apply_patch":
            return await _apply_patch(arguments, sandbox)
        elif name == "list_directory":
            return await _list_directory(arguments, sandbox)
        elif name == "search_files":
            return await _search_files(arguments, sandbox)
        elif name == "grep":
            return await _grep(arguments, sandbox)
        elif name == "run_command":
            return await _run_command(arguments, sandbox, on_output=on_output)
        elif name == "recall_memory":
            return ToolResult(output=read_memory(str(sandbox.roots[0])))
        elif name == "store_memory":
            sandbox.resolve(str(memory_path(str(sandbox.roots[0]))))
            path = await asyncio.to_thread(
                append_memory, str(sandbox.roots[0]), str(arguments.get("entry", ""))
            )
            return ToolResult(output=f"Stored project memory in {path}", file_path=str(path))
        else:
            raise ToolError(f"Unknown tool: {name}")
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(str(e))


async def _read_file(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    return await asyncio.to_thread(_read_file_sync, args, sandbox)


def _read_file_sync(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    path = sandbox.resolve(args["path"])
    if not path.is_file():
        raise ToolError(f"File not found: {path}")
    start = args.get("start_line")
    end = args.get("end_line")
    try:
        start = int(start) if start else 1
        end = int(end) if end else None
    except (TypeError, ValueError):
        raise ToolError("start_line and end_line must be integers")
    start = max(start, 1)

    parts: list[str] = []
    chars = 0
    truncated_at: int | None = None
    with open(path, "rb") as fh:
        for line_no, raw_line in enumerate(fh, 1):
            if line_no < start:
                continue
            if end is not None and line_no > end:
                break
            if b"\x00" in raw_line:
                raise ToolError(f"Cannot read binary file: {path}")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                raise ToolError(f"Cannot read binary file: {path}")
            line = line.replace("\r\n", "\n").replace("\r", "\n")
            remaining = MAX_TOOL_OUTPUT_CHARS - chars
            if len(line) > remaining:
                if remaining > 0:
                    parts.append(line[:remaining])
                truncated_at = line_no + 1
                break
            parts.append(line)
            chars += len(line)

    text = "".join(parts)
    if truncated_at is not None:
        text += (
            f"\n... (truncated at {MAX_TOOL_OUTPUT_CHARS} characters; "
            f"continue with start_line={truncated_at})"
        )
    return ToolResult(output=text, file_path=str(path), line=start)


async def _write_file(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    return await asyncio.to_thread(_write_file_sync, args, sandbox)


def _write_file_sync(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    path = sandbox.resolve(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    old_text = path.read_text() if path.exists() else None
    path.write_text(args["content"], encoding="utf-8")
    return ToolResult(
        output=f"Wrote {len(args['content'])} bytes to {path}",
        file_path=str(path),
        old_text=old_text,
        new_text=args["content"],
    )


async def _edit_file(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    return await asyncio.to_thread(_edit_file_sync, args, sandbox)


def _edit_file_sync(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    path = sandbox.resolve(args["path"])
    if not path.is_file():
        raise ToolError(f"File not found: {path}")
    text = _read_utf8_text(path, "edit")
    old = args.get("old_text", "")
    new = args.get("new_text", "")
    if not old:
        raise ToolError("old_text is empty — cannot find a match")
    count = text.count(old)
    if count == 0:
        raise ToolError("old_text not found in file")
    if count > 1:
        raise ToolError(f"old_text appears {count} times — provide more context to make it unique")
    new_text = text.replace(old, new, 1)
    path.write_text(new_text, encoding="utf-8")
    return ToolResult(
        output=f"Edited {path}",
        file_path=str(path),
        old_text=old,
        new_text=new,
    )


async def _apply_patch(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    return await asyncio.to_thread(_apply_patch_sync, args, sandbox)


def _apply_patch_sync(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    """Apply unified-diff hunks only after every context line validates."""
    path = sandbox.resolve(args["path"])
    if not path.is_file():
        raise ToolError(f"File not found: {path}")
    old_text = _read_utf8_text(path, "patch")
    source = old_text.splitlines(keepends=True)
    patch_lines = str(args.get("patch", "")).splitlines(keepends=True)
    header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    output: list[str] = []
    source_index = 0
    hunks = 0
    i = 0
    while i < len(patch_lines):
        match = header.match(patch_lines[i].rstrip("\r\n"))
        if not match:
            i += 1
            continue
        hunks += 1
        target_index = int(match.group(1)) - 1
        if target_index < source_index or target_index > len(source):
            raise ToolError("Patch hunk has an invalid or overlapping source range")
        output.extend(source[source_index:target_index])
        source_index = target_index
        expected_old = int(match.group(2) or 1)
        expected_new = int(match.group(4) or 1)
        seen_old = 0
        seen_new = 0
        i += 1
        while i < len(patch_lines) and not patch_lines[i].startswith("@@ "):
            line = patch_lines[i]
            if line.startswith(("--- ", "+++ ")):
                i += 1
                continue
            if line.startswith("\\ No newline at end of file"):
                i += 1
                continue
            prefix = line[:1]
            payload = line[1:]
            if prefix in {" ", "-"}:
                if source_index >= len(source) or source[source_index] != payload:
                    raise ToolError(f"Patch context mismatch at source line {source_index + 1}")
                if prefix == " ":
                    output.append(source[source_index])
                    seen_new += 1
                source_index += 1
                seen_old += 1
            elif prefix == "+":
                output.append(payload)
                seen_new += 1
            else:
                raise ToolError(f"Unsupported patch line: {line.rstrip()}")
            i += 1
        if seen_old != expected_old or seen_new != expected_new:
            raise ToolError("Patch hunk line counts do not match its unified diff header")
    if not hunks:
        raise ToolError("Patch contains no unified diff hunks")
    output.extend(source[source_index:])
    new_text = "".join(output)
    path.write_text(new_text, encoding="utf-8")
    return ToolResult(
        output=f"Applied {hunks} patch hunk{'s' if hunks != 1 else ''} to {path}",
        file_path=str(path),
        old_text=old_text,
        new_text=new_text,
    )


async def _list_directory(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    return await asyncio.to_thread(_list_directory_sync, args, sandbox)


def _list_directory_sync(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    path = sandbox.resolve(args.get("path", "."))
    if not path.is_dir():
        raise ToolError(f"Not a directory: {path}")
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    lines = []
    for e in entries:
        prefix = "dir " if e.is_dir() else "    "
        lines.append(f"{prefix}{e.name}")
    output = "\n".join(lines) if lines else "(empty)"
    return ToolResult(output=_bounded_output(output), file_path=str(path))


async def _search_files(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    return await asyncio.to_thread(_search_files_sync, args, sandbox)


def _walk_files(root: Path, patterns: list[str]):
    """Yield project files while pruning ignored directories before descent."""
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        rel_dir = current_path.relative_to(root)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_ignored(
                str((rel_dir / name) if str(rel_dir) != "." else Path(name)),
                patterns,
            )
        ]
        for name in filenames:
            yield current_path / name


def _search_files_sync(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    pattern = args["pattern"]
    root = sandbox.resolve(args.get("path", "."))
    fast = _run_rg(["--files", "--glob", pattern], root)
    if fast is not None:
        return ToolResult(output=fast)
    gitignore_patterns = _load_gitignore_patterns(root)
    # Always ignore common non-project directories
    ignore_patterns = gitignore_patterns + _ALWAYS_IGNORE
    matches = []
    for p in _walk_files(root, ignore_patterns):
        rel = p.relative_to(root)
        rel_str = str(rel)
        if p.is_file() and fnmatch.fnmatch(rel_str, pattern):
            if _is_ignored(rel_str, ignore_patterns):
                continue
            matches.append(rel_str)
    matches.sort()
    output = "\n".join(matches[:200]) if matches else "No files found"
    return ToolResult(output=_bounded_output(output))


async def _grep(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    return await asyncio.to_thread(_grep_sync, args, sandbox)


def _grep_sync(args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
    pattern = args["pattern"]
    root = sandbox.resolve(args.get("path", "."))
    include = args.get("include")
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"Invalid regex pattern: {e}")
    rg_args = ["--line-number", "--color", "never", "--max-count", "500"]
    if include:
        rg_args.extend(["--glob", include])
    rg_args.append(pattern)
    fast = _run_rg(rg_args, root)
    if fast is not None:
        return ToolResult(output=fast)
    gitignore_patterns = _load_gitignore_patterns(root)
    ignore_patterns = gitignore_patterns + _ALWAYS_IGNORE
    results = []
    for p in _walk_files(root, ignore_patterns):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        rel_str = str(rel)
        if _is_ignored(rel_str, ignore_patterns):
            continue
        if include and not fnmatch.fnmatch(p.name, include):
            continue
        try:
            content = _read_utf8_text(p, "read")
        except (ToolError, OSError):
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                results.append(f"{rel}:{i}: {line.strip()}")
            if len(results) >= 500:
                results.append("... (truncated at 500 matches)")
                break
        if len(results) >= 500:
            break
    output = "\n".join(results) if results else "No matches found"
    return ToolResult(output=_bounded_output(output))


async def _run_command(args: dict[str, Any], sandbox: Sandbox, on_output: Any = None) -> ToolResult:
    command = args["command"]
    timeout = args.get("timeout", 120)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = 120
    process_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_kwargs["start_new_session"] = True
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(sandbox.roots[0]),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **process_kwargs,
    )

    async def collect(stream: asyncio.StreamReader | None, label: str) -> tuple[bytes, int]:
        if stream is None:
            return b"", 0
        head_limit = _COMMAND_STREAM_LIMIT // 2
        tail_limit = _COMMAND_STREAM_LIMIT - head_limit
        head = bytearray()
        tail = bytearray()
        total = 0
        while chunk := await stream.read(8192):
            total += len(chunk)
            if on_output:
                await on_output(label, chunk.decode(errors="replace"))
            if len(head) < head_limit:
                take = min(head_limit - len(head), len(chunk))
                head.extend(chunk[:take])
                chunk = chunk[take:]
            if chunk:
                tail.extend(chunk)
                if len(tail) > tail_limit:
                    del tail[:-tail_limit]
        if total <= _COMMAND_STREAM_LIMIT:
            return bytes(head + tail), total
        marker = (f"\n... (truncated {total - _COMMAND_STREAM_LIMIT} bytes) ...\n").encode()
        return bytes(head) + marker + bytes(tail), total

    stdout_task = asyncio.create_task(collect(proc.stdout, "stdout"))
    stderr_task = asyncio.create_task(collect(proc.stderr, "stderr"))
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        stdout_info, stderr_info = await asyncio.gather(stdout_task, stderr_task)
    except asyncio.TimeoutError:
        if os.name == "nt":
            proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await proc.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise ToolError(f"Command timed out after {timeout}s")
    stdout, _ = stdout_info
    stderr, _ = stderr_info
    stdout_text = stdout.decode(errors="replace").strip()
    stderr_text = stderr.decode(errors="replace").strip()
    sections = [f"Exit code: {proc.returncode}"]
    if stdout_text:
        sections.append(f"Stdout:\n{stdout_text}")
    if stderr_text:
        sections.append(f"Stderr:\n{stderr_text}")
    if not stdout_text and not stderr_text:
        sections.append("(no output)")
    return ToolResult(output="\n\n".join(sections))
