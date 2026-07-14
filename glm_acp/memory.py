"""Opt-in, project-local instruction and durable memory support."""

from __future__ import annotations

from pathlib import Path

MAX_INSTRUCTION_CHARS = 24_000
MAX_MEMORY_CHARS = 32_000
INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md", "GLM.md")
MEMORY_RELATIVE_PATH = Path(".glm-acp") / "memory.md"


def _bounded_read(path: Path, limit: int) -> str:
    try:
        data = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return data[:limit]


def _safe_path(root: Path, path: Path) -> Path | None:
    """Resolve project knowledge paths without following links outside root."""
    try:
        resolved_root = root.resolve()
        resolved = path.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return resolved


def project_knowledge(cwd: str) -> str:
    """Load explicit root instructions and opt-in project memory."""
    root = Path(cwd)
    remaining = MAX_INSTRUCTION_CHARS
    sections: list[str] = []
    for name in INSTRUCTION_FILES:
        path = _safe_path(root, root / name)
        text = _bounded_read(path, remaining) if path is not None else ""
        if text:
            sections.append(f"### {name}\n{text}")
            remaining -= len(text)
        if remaining <= 0:
            break
    project_memory = _safe_path(root, root / MEMORY_RELATIVE_PATH)
    memory = _bounded_read(project_memory, MAX_MEMORY_CHARS) if project_memory is not None else ""
    if memory:
        sections.append(f"### Durable project memory\n{memory}")
    skills: list[str] = []
    for candidate in (root / ".agents" / "skills", root / ".codex" / "skills"):
        skill_root = _safe_path(root, candidate)
        if skill_root is None:
            continue
        try:
            is_directory = skill_root.is_dir()
        except OSError:
            is_directory = False
        if not is_directory:
            continue
        try:
            skill_files = sorted(skill_root.glob("*/SKILL.md"))[:50]
        except OSError:
            skill_files = []
        for skill_file in skill_files:
            safe_skill_file = _safe_path(root, skill_file)
            if safe_skill_file is None:
                continue
            text = _bounded_read(safe_skill_file, 4000)
            name = skill_file.parent.name
            description = ""
            if text.startswith("---"):
                for line in text.splitlines()[1:]:
                    if line == "---":
                        break
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"') or name
                    elif line.startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"')
            relative = safe_skill_file.relative_to(root.resolve())
            skills.append(f"- {name}: {description} ({relative})")
    if skills:
        sections.append(
            "### Available project skills\n"
            "Read the matching SKILL.md before using a skill.\n" + "\n".join(skills)
        )
    return "\n\n".join(sections)


def memory_path(cwd: str) -> Path:
    return Path(cwd) / MEMORY_RELATIVE_PATH


def read_memory(cwd: str) -> str:
    root = Path(cwd)
    path = _safe_path(root, memory_path(cwd))
    text = _bounded_read(path, MAX_MEMORY_CHARS) if path is not None else ""
    return text or "No durable project memory has been recorded."


def append_memory(cwd: str, entry: str) -> Path:
    """Append an explicit reusable fact while keeping the file bounded."""
    normalized = " ".join(entry.strip().split())
    if not normalized:
        raise ValueError("Memory entry cannot be empty")
    path = memory_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _bounded_read(path, MAX_MEMORY_CHARS)
    if normalized in existing:
        return path
    new_text = existing.rstrip() + ("\n" if existing.strip() else "") + f"- {normalized}\n"
    if len(new_text) > MAX_MEMORY_CHARS:
        raise ValueError("Project memory is full; consolidate it before adding entries")
    path.write_text(new_text, encoding="utf-8")
    return path
