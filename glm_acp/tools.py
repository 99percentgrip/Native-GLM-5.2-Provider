"""File system and shell tools exposed to the GLM model.

All file operations are sandboxed to the session's working directory and
any additional workspace roots.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any

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
                    "start_line": {"type": "integer", "description": "1-based line to start reading from (optional)"},
                    "end_line": {"type": "integer", "description": "1-based line to end reading at (optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it does not exist, overwrites if it does.",
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
            "description": "Replace a specific block of text in a file. Both old_text and new_text must be exact.",
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
            "description": "Search for files by glob pattern (e.g. **/*.py). Returns matching paths.",
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
            "description": "Search file contents using a regular expression. Returns matching lines with file and line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for"},
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
            "description": "Execute a shell command in the working directory. Use for builds, tests, git, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
                },
                "required": ["command"],
            },
        },
    },
]

TOOL_KINDS: dict[str, str] = {
    "read_file": "read",
    "write_file": "edit",
    "edit_file": "edit",
    "list_directory": "read",
    "search_files": "search",
    "grep": "search",
    "run_command": "execute",
}


class ToolError(Exception):
    pass


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
) -> str:
    """Execute a tool call and return its output as a string."""
    try:
        if name == "read_file":
            return await _read_file(arguments, sandbox)
        elif name == "write_file":
            return await _write_file(arguments, sandbox)
        elif name == "edit_file":
            return await _edit_file(arguments, sandbox)
        elif name == "list_directory":
            return await _list_directory(arguments, sandbox)
        elif name == "search_files":
            return await _search_files(arguments, sandbox)
        elif name == "grep":
            return await _grep(arguments, sandbox)
        elif name == "run_command":
            return await _run_command(arguments, sandbox)
        else:
            raise ToolError(f"Unknown tool: {name}")
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(str(e))


async def _read_file(args: dict[str, Any], sandbox: Sandbox) -> str:
    path = sandbox.resolve(args["path"])
    if not path.is_file():
        raise ToolError(f"File not found: {path}")
    text = path.read_text()
    start = args.get("start_line")
    end = args.get("end_line")
    if start or end:
        lines = text.splitlines(keepends=True)
        s = (start - 1) if start else 0
        e = end if end else len(lines)
        text = "".join(lines[s:e])
    return text


async def _write_file(args: dict[str, Any], sandbox: Sandbox) -> str:
    path = sandbox.resolve(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"])
    return f"Wrote {len(args['content'])} bytes to {path}"


async def _edit_file(args: dict[str, Any], sandbox: Sandbox) -> str:
    path = sandbox.resolve(args["path"])
    if not path.is_file():
        raise ToolError(f"File not found: {path}")
    text = path.read_text()
    old = args["old_text"]
    new = args["new_text"]
    count = text.count(old)
    if count == 0:
        raise ToolError("old_text not found in file")
    if count > 1:
        raise ToolError(f"old_text appears {count} times — provide more context to make it unique")
    path.write_text(text.replace(old, new, 1))
    return f"Edited {path}"


async def _list_directory(args: dict[str, Any], sandbox: Sandbox) -> str:
    path = sandbox.resolve(args.get("path", "."))
    if not path.is_dir():
        raise ToolError(f"Not a directory: {path}")
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    lines = []
    for e in entries:
        prefix = "dir " if e.is_dir() else "    "
        lines.append(f"{prefix}{e.name}")
    return "\n".join(lines) if lines else "(empty)"


async def _search_files(args: dict[str, Any], sandbox: Sandbox) -> str:
    pattern = args["pattern"]
    root = sandbox.resolve(args.get("path", "."))
    matches = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if fnmatch.fnmatch(str(rel), pattern) and p.is_file():
            matches.append(str(rel))
    matches.sort()
    return "\n".join(matches[:200]) if matches else "No files found"


async def _grep(args: dict[str, Any], sandbox: Sandbox) -> str:
    pattern = args["pattern"]
    root = sandbox.resolve(args.get("path", "."))
    include = args.get("include")
    regex = re.compile(pattern)
    results = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if include and not fnmatch.fnmatch(p.name, include):
            continue
        try:
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if regex.search(line):
                    results.append(f"{rel}:{i}: {line.strip()}")
        except (UnicodeDecodeError, OSError):
            continue
        if len(results) >= 500:
            results.append("... (truncated at 500 matches)")
            break
    return "\n".join(results) if results else "No matches found"


async def _run_command(args: dict[str, Any], sandbox: Sandbox) -> str:
    command = args["command"]
    timeout = args.get("timeout", 120)
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(sandbox.roots[0]),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(f"Command timed out after {timeout}s")
    output = stdout.decode()
    if stderr:
        output += "\n" + stderr.decode()
    return output.strip() if output.strip() else "(no output)"
