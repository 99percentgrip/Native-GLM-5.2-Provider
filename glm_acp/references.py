"""Bounded explicit context expansion with language-aware relevance ranking."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from shutil import which

from .security import wrap_untrusted_output
from .tools import Sandbox, ToolError

MAX_REFERENCES = 12
MAX_REFERENCE_CHARS = 48_000
MAX_FILE_CHARS = 16_000
MAX_FOLDER_FILES = 40
MAX_RANK_CANDIDATES = 600
_REFERENCE = re.compile(r"(?<!\w)@(file|folder|symbol):([^\s]+)|(?<!\w)@diff\b")
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]{2,}")
_IGNORED = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}
_SENSITIVE_NAMES = {".env", "credentials.json", "id_rsa", "id_ed25519", "known_hosts"}
_LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}
_DEFINITION_PATTERNS = {
    "python": r"(?m)^\s*(?:async\s+)?(?:def|class)\s+{name}\b",
    "javascript": r"(?m)^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+{name}\b",
    "typescript": (
        r"(?m)^\s*(?:export\s+)?(?:declare\s+)?(?:async\s+)?"
        r"(?:function|class|interface|type|enum|const|let|var)\s+{name}\b"
    ),
    "go": r"(?m)^\s*(?:func|type|var|const)\s+(?:\([^)]*\)\s*)?{name}\b",
    "rust": (
        r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
        r"(?:fn|struct|enum|trait|type|const|static|mod)\s+{name}\b"
    ),
    "java": (
        r"(?m)^\s*(?:public|protected|private|static|final|abstract|\s)*\b"
        r"(?:class|interface|enum|record)\s+{name}\b"
    ),
    "kotlin": (
        r"(?m)^\s*(?:public|private|internal|protected|data|sealed|\s)*"
        r"(?:fun|class|interface|object|typealias|val|var)\s+{name}\b"
    ),
    "ruby": r"(?m)^\s*(?:def|class|module)\s+{name}\b",
    "c": r"(?m)^.*\b{name}\s*\([^;]*\)\s*\{{",
    "cpp": r"(?m)^.*\b(?:class|struct|enum)?\s*{name}\b",
}


def _sensitive(path: Path) -> bool:
    return (
        any(part == ".ssh" for part in path.parts)
        or path.name in _SENSITIVE_NAMES
        or path.name.startswith(".env.")
    )


def _text(path: Path, limit: int = MAX_FILE_CHARS) -> str:
    data = path.read_bytes()
    if b"\0" in data:
        return "[binary file omitted]"
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError:
        return "[non-UTF-8 file omitted]"
    return value[:limit] + ("\n… [truncated]" if len(value) > limit else "")


def _query_terms(content: str) -> set[str]:
    without_refs = _REFERENCE.sub(" ", content)
    return {value.lower() for value in _IDENTIFIER.findall(without_refs) if len(value) > 2}


def _changed_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=5,
        check=False,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if result.returncode != 0:
        return set()
    changed: set[str] = set()
    for entry in result.stdout.split("\0"):
        if len(entry) >= 4:
            changed.add(entry[3:].split(" -> ")[-1])
    return changed


def _definition_regex(language: str, symbol: str) -> re.Pattern[str] | None:
    template = _DEFINITION_PATTERNS.get(language)
    if not template:
        return None
    return re.compile(template.format(name=re.escape(symbol)))


def _rank_file(
    path: Path,
    root: Path,
    terms: set[str],
    changed: set[str],
    symbol: str = "",
) -> tuple[int, str]:
    language = _LANGUAGES.get(path.suffix.lower(), "text")
    relative = path.relative_to(root).as_posix()
    score = 0
    reasons: list[str] = []
    if relative in changed:
        score += 30
        reasons.append("changed")
    name_terms = set(_IDENTIFIER.findall(path.stem.lower()))
    overlap = terms & name_terms
    if overlap:
        score += 18 + 3 * len(overlap)
        reasons.append("name-match")
    try:
        body = _text(path, 32_000)
    except OSError:
        return -1, "unreadable"
    lowered = body.lower()
    content_hits = sum(1 for term in terms if term in lowered)
    if content_hits:
        score += min(content_hits, 8) * 3
        reasons.append("task-term")
    if symbol:
        pattern = _definition_regex(language, symbol)
        if pattern and pattern.search(body):
            score += 100
            reasons.append("definition")
        elif re.search(rf"\b{re.escape(symbol)}\b", body):
            score += 25
            reasons.append("reference")
    if path.name.startswith("test_") or path.name.endswith((".test.ts", ".test.js", "_test.go")):
        score += 6
        reasons.append("test")
    if path.name in {"pyproject.toml", "package.json", "Cargo.toml", "go.mod"}:
        score += 4
        reasons.append("manifest")
    return score, ",".join(reasons) or language


def _candidate_files(path: Path) -> list[Path]:
    output: list[Path] = []
    for current, dirs, files in os.walk(path):
        dirs[:] = [name for name in sorted(dirs) if name not in _IGNORED]
        for name in sorted(files):
            candidate = Path(current) / name
            if candidate.is_symlink() or not candidate.is_file() or _sensitive(candidate):
                continue
            output.append(candidate)
            if len(output) >= MAX_RANK_CANDIDATES:
                return output
    return output


def _folder(path: Path, root: Path, terms: set[str]) -> str:
    if not path.is_dir():
        raise ToolError(f"Reference folder not found: {path}")
    changed = _changed_paths(root)
    candidates = _candidate_files(path)
    ranked = sorted(
        ((_rank_file(candidate, root, terms, changed), candidate) for candidate in candidates),
        key=lambda item: (-item[0][0], item[1].as_posix()),
    )[:MAX_FOLDER_FILES]
    output: list[str] = []
    for (score, reason), file_path in ranked:
        if score < 0:
            continue
        relative = file_path.relative_to(path)
        language = _LANGUAGES.get(file_path.suffix.lower(), "text")
        body = _text(file_path, 2_000)
        output.append(
            f"--- {relative.as_posix()} "
            f"[language={language}; relevance={score}; {reason}] ---\n{body}"
        )
    if len(candidates) > MAX_FOLDER_FILES:
        output.append("… [lower-ranked folder files omitted]")
    return "\n".join(output) or "[empty folder]"


def _snippet(body: str, symbol: str) -> str:
    lines = body.splitlines()
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    index = next((i for i, line in enumerate(lines) if pattern.search(line)), 0)
    start = max(0, index - 3)
    end = min(len(lines), index + 8)
    return "\n".join(f"{line_no + 1}: {lines[line_no]}" for line_no in range(start, end))


def _symbol(root: Path, symbol: str, terms: set[str]) -> tuple[str, list[str]]:
    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_.$:-]{0,199}", symbol):
        raise ToolError(f"Invalid symbol reference: {symbol}")
    leaf = symbol.split(".")[-1]
    rg = which("rg")
    if rg:
        result = subprocess.run(
            [
                rg,
                "-l",
                "--fixed-strings",
                "--glob",
                "!**/.git/**",
                "--glob",
                "!**/node_modules/**",
                leaf,
                ".",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
        matching_paths = [(root / relative).resolve() for relative in result.stdout.splitlines()]
    else:
        matching_paths = []
        for path in _candidate_files(root):
            try:
                if leaf in _text(path, 32_000):
                    matching_paths.append(path.resolve())
            except OSError:
                continue
    changed = _changed_paths(root)
    candidates: list[tuple[tuple[int, str], Path]] = []
    for path in matching_paths[:MAX_RANK_CANDIDATES]:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file() and not path.is_symlink() and not _sensitive(path):
            candidates.append((_rank_file(path, root, terms, changed, leaf), path))
    candidates.sort(key=lambda item: (-item[0][0], item[1].as_posix()))
    sections: list[str] = []
    targets: list[str] = []
    for (score, reason), path in candidates[:12]:
        body = _text(path, MAX_FILE_CHARS)
        language = _LANGUAGES.get(path.suffix.lower(), "text")
        sections.append(
            f"--- {path.relative_to(root).as_posix()} "
            f"[language={language}; relevance={score}; {reason}] ---\n{_snippet(body, leaf)}"
        )
        targets.append(str(path))
    if not sections:
        return f"No definition or reference match found for {symbol}", []
    return "\n".join(sections)[:MAX_FILE_CHARS], targets


def _diff(root: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=False,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "--"],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
            env={"PATH": os.environ.get("PATH", "")},
        )
    if result.returncode != 0:
        return "Git diff unavailable for this workspace"
    return (result.stdout or "[no tracked working-tree diff]")[:MAX_FILE_CHARS]


def expand_references(content: str, sandbox: Sandbox) -> tuple[str, list[str]]:
    """Append ranked, bounded, untrusted reference bodies and addressed paths."""
    matches = list(_REFERENCE.finditer(content))
    if not matches:
        return content, []
    if len(matches) > MAX_REFERENCES:
        raise ToolError(f"At most {MAX_REFERENCES} @ references are allowed per prompt")
    root = sandbox.roots[0]
    terms = _query_terms(content)
    sections: list[str] = []
    targets: list[str] = []
    total = 0
    for match in matches:
        kind = match.group(1) or "diff"
        value = (match.group(2) or "").rstrip(".,;)")
        if kind == "file":
            path = sandbox.resolve(value)
            if not path.is_file():
                raise ToolError(f"Reference file not found: {path}")
            if _sensitive(path):
                raise ToolError(f"Sensitive file cannot be expanded as a context reference: {path}")
            body = _text(path)
            targets.append(str(path))
        elif kind == "folder":
            path = sandbox.resolve(value)
            body = _folder(path, root, terms)
            targets.append(str(path))
        elif kind == "symbol":
            body, symbol_targets = _symbol(root, value, terms)
            targets.extend(symbol_targets)
        else:
            body = _diff(root)
        remaining = MAX_REFERENCE_CHARS - total
        if remaining <= 0:
            break
        body = body[:remaining]
        total += len(body)
        sections.append(f"Reference @{kind}{':' + value if value else ''}:\n{body}")
    expanded = "\n\n".join(sections)
    return content + "\n\nResolved explicit context references:\n" + wrap_untrusted_output(
        expanded, "explicit-references"
    ), list(dict.fromkeys(targets))[:100]
