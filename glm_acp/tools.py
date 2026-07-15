"""File system and shell tools exposed to the GLM model.

All file operations are sandboxed to the session's working directory and
any additional workspace roots.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory import (
    append_memory,
    append_user_profile,
    curate_learned_skills,
    discard_skill_evolution,
    draft_skill_evolution,
    forget_learned_skill,
    forget_memory,
    forget_skill_bundle,
    forget_user_profile,
    learned_skills_path,
    list_learned_skills,
    list_skill_bundles,
    manage_learned_skill,
    memory_path,
    promote_skill_evolution,
    propose_skill_evolution,
    read_learned_skill,
    read_memory,
    read_skill_bundle,
    read_user_profile,
    skill_curator_status,
    write_learned_skill,
    write_skill_bundle,
)

CRONJOB_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "cronjob",
        "description": (
            "Permission-gated management for persistent local scheduled tasks. Jobs run in "
            "fresh sessions and persist results locally. Scheduled runs cannot call this tool."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "update", "pause", "resume", "run", "remove"],
                },
                "job_id": {"type": "string"},
                "schedule": {
                    "type": "string",
                    "description": (
                        "Delay (30m), interval (every 2h), five-field cron, or aware ISO timestamp"
                    ),
                },
                "prompt": {"type": "string"},
                "name": {"type": "string"},
                "timezone": {"type": "string"},
                "repeat": {"type": "integer", "minimum": 1},
                "workdir": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "bundles": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "script": {"type": "string"},
                "no_agent": {"type": "boolean"},
                "include_disabled": {"type": "boolean"},
            },
            "required": ["action"],
        },
    },
}

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
_SENSITIVE_ENV_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_CREDENTIAL",
    "_PRIVATE_KEY",
    "_ACCESS_KEY",
)
_SENSITIVE_ENV_NAMES = {
    "ZAI_API_KEY",
    "Z_AI_API_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "SSH_AUTH_SOCK",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}


def _bounded_output(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated at {limit} characters)"


def _command_environment() -> dict[str, str]:
    """Build a useful shell environment without exposing inherited credentials."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _SENSITIVE_ENV_NAMES
        and not key.upper().endswith(_SENSITIVE_ENV_SUFFIXES)
    }


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
    CRONJOB_TOOL_DEFINITION,
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
                "Store one stable, reusable project fact. "
                "Do not store secrets, transient task state, or reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {"entry": {"type": "string"}},
                "required": ["entry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_user_profile",
            "description": "Read private cross-project facts and preferences approved by the user.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_user_profile",
            "description": (
                "Store one explicit durable user fact, preference, workflow, or environment "
                "detail across projects. Never infer sensitive traits or store secrets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["identity", "preference", "workflow", "environment"],
                    },
                },
                "required": ["entry", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "Remove one exact durable fact from project or user memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["project", "user"]},
                    "entry": {"type": "string"},
                },
                "required": ["scope", "entry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "session_search",
            "description": (
                "Search prior conversations when the user refers to earlier work. With no "
                "arguments, browse recent sessions. Use session_id plus around_ordinal to "
                "scroll around a previous match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "session_id": {"type": "string"},
                    "around_ordinal": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "window": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": (
                "List reusable project skills learned by this agent. Metadata only; "
                "use read_skill to load instructions when relevant."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Read one learned project's SKILL.md instructions on demand.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "learn_skill",
            "description": (
                "Create or refine a reusable project SKILL.md after a non-trivial task "
                "has passed verification. Store only concise procedures and pitfalls; "
                "never credentials, raw reasoning, transient state, or routine steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short lowercase/hyphen skill name",
                    },
                    "description": {
                        "type": "string",
                        "description": "What the skill does and when it should be used",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Concise reusable imperative workflow",
                    },
                    "environments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional project tags such as python, node, rust, or git",
                    },
                    "requires_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional core tool names required by this skill",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional semantic task tags that must match the request",
                    },
                },
                "required": ["name", "description", "instructions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_skill",
            "description": (
                "Remove one agent-learned project skill. This cannot remove user-authored "
                ".agents or .codex skills."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_skill",
            "description": (
                "Pin, unpin, archive, or restore one agent-learned project skill. "
                "Archiving is reversible and pinned skills cannot be archived."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["pin", "unpin", "archive", "restore"],
                    },
                },
                "required": ["name", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "curate_skills",
            "description": (
                "Run deterministic skill maintenance: mark skills stale after 30 idle days "
                "and archive unpinned skills after 90 idle days. Never deletes skills."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skill_bundles",
            "description": "List project-local groups of related learned skills.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill_bundle",
            "description": "Load every relevant learned skill in one project bundle.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_skill_bundle",
            "description": "Create or delete a project-local skill bundle with permission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "delete"]},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "instruction": {"type": "string"},
                },
                "required": ["action", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evolve_skill",
            "description": (
                "Draft a candidate from failed traces, then stage, promote, or discard it. "
                "Proposals require compatible completed baseline and candidate benchmark "
                "reports with no quality, latency, or token-cost regression."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["draft", "propose", "promote", "discard"],
                    },
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "instructions": {"type": "string"},
                    "baseline_report": {"type": "string"},
                    "candidate_report": {"type": "string"},
                    "failed_report": {"type": "string"},
                },
                "required": ["action", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                "Delegate one bounded read-only investigation or review to an independent "
                "auxiliary GLM worker. The worker cannot edit files or run commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "context": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["investigator", "reviewer", "test-analyst"],
                    },
                },
                "required": ["goal"],
            },
        },
    },
]

TOOL_KINDS: dict[str, str] = {
    "cronjob": "edit",
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
    "recall_user_profile": "read",
    "store_user_profile": "edit",
    "forget_memory": "edit",
    "session_search": "search",
    "list_skills": "read",
    "read_skill": "read",
    "learn_skill": "edit",
    "forget_skill": "edit",
    "manage_skill": "edit",
    "curate_skills": "edit",
    "list_skill_bundles": "read",
    "read_skill_bundle": "read",
    "manage_skill_bundle": "edit",
    "evolve_skill": "edit",
    "delegate_task": "other",
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
    # For commands: the process exit code, otherwise None
    exit_code: int | None = None


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
    cron_delivery: Any = None,
) -> ToolResult:
    """Execute a tool call and return a structured result."""
    try:
        if name == "cronjob":
            return await _cronjob(arguments, sandbox, delivery=cron_delivery)
        elif name == "read_file":
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
        elif name == "recall_user_profile":
            return ToolResult(output=read_user_profile())
        elif name == "store_user_profile":
            path = await asyncio.to_thread(
                append_user_profile,
                str(arguments.get("entry", "")),
                str(arguments.get("category", "preference")),
            )
            return ToolResult(output=f"Stored private user profile entry in {path}")
        elif name == "forget_memory":
            scope = str(arguments.get("scope", ""))
            entry = str(arguments.get("entry", ""))
            if scope == "project":
                sandbox.resolve(str(memory_path(str(sandbox.roots[0]))))
                path = await asyncio.to_thread(forget_memory, str(sandbox.roots[0]), entry)
            elif scope == "user":
                path = await asyncio.to_thread(forget_user_profile, entry)
            else:
                raise ToolError("Memory scope must be project or user")
            return ToolResult(output=f"Forgot {scope} memory entry from {path}")
        elif name == "list_skills":
            skills = await asyncio.to_thread(list_learned_skills, str(sandbox.roots[0]))
            if not skills:
                return ToolResult(output="No learned project skills have been recorded.")
            status = await asyncio.to_thread(skill_curator_status, str(sandbox.roots[0]))
            return ToolResult(
                output=(
                    f"Skills: {status['active']} active, {status['stale']} stale, "
                    f"{status['archived']} archived, {status['pinned']} pinned\n"
                    + "\n".join(
                        f"- {skill['name']} [{skill['state']}]"
                        f"{' [pinned]' if skill['pinned'] else ''}: "
                        f"{skill['description']} ({skill['path']}; uses={skill['use_count']}, "
                        f"revisions={skill['revision_count']})"
                        for skill in skills
                    )
                )
            )
        elif name == "read_skill":
            text = await asyncio.to_thread(
                read_learned_skill, str(sandbox.roots[0]), str(arguments.get("name", ""))
            )
            return ToolResult(output=text)
        elif name == "learn_skill":
            sandbox.resolve(str(learned_skills_path(str(sandbox.roots[0]))))
            path = await asyncio.to_thread(
                write_learned_skill,
                str(sandbox.roots[0]),
                str(arguments.get("name", "")),
                str(arguments.get("description", "")),
                str(arguments.get("instructions", "")),
                [str(value) for value in arguments.get("environments", [])],
                [str(value) for value in arguments.get("requires_tools", [])],
                [str(value) for value in arguments.get("tasks", [])],
            )
            return ToolResult(output=f"Learned project skill in {path}", file_path=str(path))
        elif name == "forget_skill":
            sandbox.resolve(str(learned_skills_path(str(sandbox.roots[0]))))
            path = await asyncio.to_thread(
                forget_learned_skill,
                str(sandbox.roots[0]),
                str(arguments.get("name", "")),
            )
            return ToolResult(output=f"Forgot learned project skill at {path}")
        elif name == "manage_skill":
            result = await asyncio.to_thread(
                manage_learned_skill,
                str(sandbox.roots[0]),
                str(arguments.get("name", "")),
                str(arguments.get("action", "")),
            )
            return ToolResult(output=json.dumps(result, ensure_ascii=False, indent=2))
        elif name == "curate_skills":
            result = await asyncio.to_thread(curate_learned_skills, str(sandbox.roots[0]))
            return ToolResult(output=json.dumps(result, ensure_ascii=False, indent=2))
        elif name == "list_skill_bundles":
            bundles = await asyncio.to_thread(list_skill_bundles, str(sandbox.roots[0]))
            return ToolResult(output=json.dumps(bundles, ensure_ascii=False, indent=2))
        elif name == "read_skill_bundle":
            text = await asyncio.to_thread(
                read_skill_bundle,
                str(sandbox.roots[0]),
                str(arguments.get("name", "")),
            )
            return ToolResult(output=text)
        elif name == "manage_skill_bundle":
            action = str(arguments.get("action", ""))
            if action == "create":
                path = await asyncio.to_thread(
                    write_skill_bundle,
                    str(sandbox.roots[0]),
                    str(arguments.get("name", "")),
                    str(arguments.get("description", "")),
                    [str(value) for value in arguments.get("skills", [])],
                    str(arguments.get("instruction", "")),
                )
                return ToolResult(output=f"Stored skill bundle in {path}", file_path=str(path))
            if action == "delete":
                path = await asyncio.to_thread(
                    forget_skill_bundle,
                    str(sandbox.roots[0]),
                    str(arguments.get("name", "")),
                )
                return ToolResult(output=f"Deleted skill bundle from {path}")
            raise ToolError("Bundle action must be create or delete")
        elif name == "evolve_skill":
            action = str(arguments.get("action", ""))
            skill_name = str(arguments.get("name", ""))
            if action == "draft":
                path = await asyncio.to_thread(
                    draft_skill_evolution,
                    str(sandbox.roots[0]),
                    skill_name,
                    str(arguments.get("failed_report", "")),
                )
                return ToolResult(output=f"Regression-derived skill draft staged in {path}")
            if action == "propose":
                path = await asyncio.to_thread(
                    propose_skill_evolution,
                    str(sandbox.roots[0]),
                    skill_name,
                    str(arguments.get("description", "")),
                    str(arguments.get("instructions", "")),
                    str(arguments.get("baseline_report", "")),
                    str(arguments.get("candidate_report", "")),
                )
                return ToolResult(output=f"Validated skill candidate staged in {path}")
            if action == "promote":
                path = await asyncio.to_thread(
                    promote_skill_evolution, str(sandbox.roots[0]), skill_name
                )
                return ToolResult(output=f"Promoted validated skill candidate to {path}")
            if action == "discard":
                path = await asyncio.to_thread(
                    discard_skill_evolution, str(sandbox.roots[0]), skill_name
                )
                return ToolResult(output=f"Discarded skill candidate {path}")
            raise ToolError("Evolution action must be draft, propose, promote, or discard")
        else:
            raise ToolError(f"Unknown tool: {name}")
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(str(e))


async def _cronjob(args: dict[str, Any], sandbox: Sandbox, *, delivery: Any = None) -> ToolResult:
    from .cron import (
        create_job,
        get_job,
        list_jobs,
        pause_job,
        remove_job,
        resume_job,
        update_job,
    )
    from .cron_scheduler import tick

    action = str(args.get("action", "")).lower()
    if action == "list":
        jobs = await asyncio.to_thread(
            list_jobs, include_disabled=bool(args.get("include_disabled", True))
        )
        return ToolResult(output=json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2))
    if action == "create":
        if not args.get("schedule"):
            raise ToolError("schedule is required for create")
        workdir = sandbox.resolve(str(args.get("workdir") or sandbox.roots[0]))
        job = await asyncio.to_thread(
            create_job,
            schedule=str(args["schedule"]),
            prompt=str(args.get("prompt", "")),
            name=args.get("name"),
            workspace_root=str(sandbox.roots[0]),
            workdir=str(workdir),
            timezone_name=str(args.get("timezone") or "UTC"),
            repeat=args.get("repeat"),
            skills=args.get("skills"),
            bundles=args.get("bundles"),
            script=args.get("script"),
            no_agent=bool(args.get("no_agent", False)),
            origin_session_id=args.get("_origin_session_id"),
        )
        return ToolResult(output=json.dumps({"job": job}, ensure_ascii=False, indent=2))
    job_id = str(args.get("job_id") or "")
    if not job_id:
        raise ToolError(f"job_id is required for {action}")
    if action == "pause":
        job = await asyncio.to_thread(pause_job, job_id)
    elif action == "resume":
        job = await asyncio.to_thread(resume_job, job_id)
    elif action == "remove":
        removed = await asyncio.to_thread(remove_job, job_id)
        if not removed:
            raise ToolError(f"Cron job not found: {job_id}")
        return ToolResult(output=json.dumps({"removed": job_id}))
    elif action == "run":
        if await asyncio.to_thread(get_job, job_id) is None:
            raise ToolError(f"Cron job not found: {job_id}")
        result = await tick(job_id=job_id, force=True, delivery=delivery)
        return ToolResult(output=json.dumps(result))
    elif action == "update":
        updates = {
            key: args[key]
            for key in (
                "schedule",
                "prompt",
                "name",
                "timezone",
                "repeat",
                "workdir",
                "skills",
                "bundles",
                "script",
                "no_agent",
            )
            if key in args
        }
        if "workdir" in updates:
            updates["workdir"] = str(sandbox.resolve(str(updates["workdir"])))
        job = await asyncio.to_thread(update_job, job_id, updates)
    else:
        raise ToolError(f"Unknown cron action: {action}")
    return ToolResult(output=json.dumps({"job": job}, ensure_ascii=False, indent=2))


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
        env=_command_environment(),
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
    return ToolResult(output="\n\n".join(sections), exit_code=proc.returncode)
