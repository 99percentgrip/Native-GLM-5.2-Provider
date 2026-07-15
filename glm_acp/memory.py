"""Opt-in, project-local instruction and durable memory support."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config_dir

MAX_INSTRUCTION_CHARS = 24_000
MAX_MEMORY_CHARS = 32_000
INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md", "GLM.md")
MEMORY_RELATIVE_PATH = Path(".glm-acp") / "memory.md"
LEARNED_SKILLS_RELATIVE_PATH = Path(".glm-acp") / "skills"
MAX_LEARNED_SKILLS = 100
MAX_SKILL_DESCRIPTION_CHARS = 500
MAX_SKILL_INSTRUCTIONS_CHARS = 12_000
MAX_USER_PROFILE_CHARS = 16_000
USER_PROFILE_FILENAME = "user.md"
SKILL_USAGE_FILENAME = ".usage.json"
SKILL_ARCHIVE_DIRNAME = ".archive"
SKILL_STALE_AFTER_DAYS = 30
SKILL_ARCHIVE_AFTER_DAYS = 90

_SENSITIVE_LEARNING_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|token|secret|password|credential)\s*[:=]\s*[^\s]{8,}",
        re.IGNORECASE,
    ),
)


def _bounded_read(path: Path, limit: int) -> str:
    try:
        data = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return data[:limit]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o777)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
        os.replace(temporary, path)
        if private and os.name != "nt":
            path.parent.chmod(0o700)
            path.chmod(0o600)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


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
    for candidate in (
        root / LEARNED_SKILLS_RELATIVE_PATH,
        root / ".agents" / "skills",
        root / ".codex" / "skills",
    ):
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
    curator = skill_curator_status(cwd)
    if (
        curator["due_stale"]
        or curator["due_archive"]
        or curator["drifted"]
        or curator["overlap_candidates"]
    ):
        sections.append(
            "### Skill maintenance due\n"
            f"Stale candidates: {', '.join(curator['due_stale']) or 'none'}\n"
            f"Archive candidates: {', '.join(curator['due_archive']) or 'none'}\n"
            f"Manually changed skills: {', '.join(curator['drifted']) or 'none'}\n"
            f"Possible overlaps: {json.dumps(curator['overlap_candidates'])}\n"
            "Curation is reversible and requires user permission."
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
    clean_entry = _validate_learning_text(entry, "Memory entry", 2000)
    normalized = " ".join(clean_entry.split())
    path = memory_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _bounded_read(path, MAX_MEMORY_CHARS)
    if normalized in existing:
        return path
    new_text = existing.rstrip() + ("\n" if existing.strip() else "") + f"- {normalized}\n"
    if len(new_text) > MAX_MEMORY_CHARS:
        raise ValueError("Project memory is full; consolidate it before adding entries")
    _atomic_write(path, new_text)
    return path


def forget_memory(cwd: str, entry: str) -> Path:
    """Remove one exact normalized project-memory entry."""
    normalized = " ".join(entry.strip().split())
    if not normalized:
        raise ValueError("Memory entry cannot be empty")
    path = memory_path(cwd)
    safe = _safe_path(Path(cwd), path)
    if safe is None:
        raise ValueError("Project memory path escapes the workspace")
    lines = _bounded_read(safe, MAX_MEMORY_CHARS).splitlines()
    target = f"- {normalized}"
    remaining = [line for line in lines if line.strip() != target]
    if len(remaining) == len(lines):
        raise ValueError("Project memory entry not found")
    _atomic_write(safe, "\n".join(remaining).rstrip() + ("\n" if remaining else ""))
    return safe


def _validate_learning_text(value: str, label: str, limit: int) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{label} cannot be empty")
    if len(text) > limit:
        raise ValueError(f"{label} exceeds the {limit:,}-character limit")
    if any(pattern.search(text) for pattern in _SENSITIVE_LEARNING_PATTERNS):
        raise ValueError(f"{label} appears to contain a credential or secret")
    return text


def user_profile_path() -> Path:
    return config_dir() / USER_PROFILE_FILENAME


def _safe_user_profile_path() -> Path | None:
    root = config_dir()
    path = user_profile_path()
    try:
        if path.is_symlink():
            return None
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return path


def read_user_profile() -> str:
    path = _safe_user_profile_path()
    text = _bounded_read(path, MAX_USER_PROFILE_CHARS) if path is not None else ""
    return text or "No durable user profile has been recorded."


def user_knowledge() -> str:
    """Return only recorded cross-project user knowledge for prompt injection."""
    path = _safe_user_profile_path()
    return _bounded_read(path, MAX_USER_PROFILE_CHARS) if path is not None else ""


def append_user_profile(entry: str, category: str = "preference") -> Path:
    """Store one private, cross-project user fact with an explicit category."""
    clean = _validate_learning_text(entry, "User profile entry", 2000)
    normalized = " ".join(clean.split())
    normalized_category = re.sub(r"[^a-z-]", "", category.strip().lower())
    if normalized_category not in {"identity", "preference", "workflow", "environment"}:
        raise ValueError("User profile category is invalid")
    path = _safe_user_profile_path()
    if path is None:
        raise ValueError("User profile path is unsafe")
    existing = _bounded_read(path, MAX_USER_PROFILE_CHARS)
    line = f"- [{normalized_category}] {normalized}"
    if line in existing.splitlines():
        return path
    new_text = existing.rstrip() + ("\n" if existing.strip() else "") + line + "\n"
    if len(new_text) > MAX_USER_PROFILE_CHARS:
        raise ValueError("User profile is full; remove or consolidate entries first")
    _atomic_write(path, new_text, private=True)
    return path


def forget_user_profile(entry: str) -> Path:
    """Remove an exact private user-profile entry regardless of category."""
    normalized = " ".join(entry.strip().split())
    if not normalized:
        raise ValueError("User profile entry cannot be empty")
    path = _safe_user_profile_path()
    if path is None:
        raise ValueError("User profile path is unsafe")
    lines = _bounded_read(path, MAX_USER_PROFILE_CHARS).splitlines()
    pattern = re.compile(r"^- \[[a-z-]+\] (.*)$")
    remaining = [
        line
        for line in lines
        if not (match := pattern.match(line.strip())) or match.group(1) != normalized
    ]
    if len(remaining) == len(lines):
        raise ValueError("User profile entry not found")
    _atomic_write(path, "\n".join(remaining).rstrip() + ("\n" if remaining else ""), private=True)
    return path


def _skill_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    slug = slug[:64].rstrip("-")
    if not slug:
        raise ValueError("Skill name must contain letters or numbers")
    return slug


def learned_skills_path(cwd: str) -> Path:
    return Path(cwd) / LEARNED_SKILLS_RELATIVE_PATH


def _learned_skills_root(cwd: str, *, create: bool = False) -> Path:
    root = Path(cwd)
    path = _safe_path(root, learned_skills_path(cwd))
    if path is None:
        raise ValueError("Learned skills directory escapes the workspace")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _skill_usage_path(root: Path) -> Path:
    return root / SKILL_USAGE_FILENAME


def _load_skill_usage(root: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(_bounded_read(_skill_usage_path(root), 256_000))
    except json.JSONDecodeError:
        return {}
    skills = payload.get("skills") if isinstance(payload, dict) else None
    return skills if isinstance(skills, dict) else {}


def _save_skill_usage(root: Path, skills: dict[str, dict[str, Any]]) -> None:
    payload = {"version": 1, "updated_at": _now_iso(), "skills": skills}
    _atomic_write(
        _skill_usage_path(root),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _skill_record(skills: dict[str, dict[str, Any]], slug: str) -> dict[str, Any]:
    now = _now_iso()
    record = skills.setdefault(
        slug,
        {
            "created_at": now,
            "updated_at": now,
            "last_used_at": None,
            "view_count": 0,
            "use_count": 0,
            "revision_count": 0,
            "state": "active",
            "pinned": False,
        },
    )
    return record


def list_learned_skills(cwd: str) -> list[dict[str, Any]]:
    """Return bounded metadata for agent-owned learned skills."""
    try:
        root = _learned_skills_root(cwd)
    except ValueError:
        return []
    try:
        is_directory = root.is_dir()
    except OSError:
        return []
    if not is_directory:
        return []
    skills: list[dict[str, Any]] = []
    usage = _load_skill_usage(root)
    try:
        active = [
            path for path in sorted(root.glob("*/SKILL.md")) if path.parent.name != ".archive"
        ]
        archived = sorted((root / SKILL_ARCHIVE_DIRNAME).glob("*/SKILL.md"))
        candidates = [(path, "active") for path in active] + [
            (path, "archived") for path in archived
        ]
        candidates = candidates[:MAX_LEARNED_SKILLS]
    except OSError:
        return []
    workspace = Path(cwd).resolve()
    for candidate, location_state in candidates:
        safe = _safe_path(Path(cwd), candidate)
        if safe is None:
            continue
        text = _bounded_read(safe, MAX_SKILL_INSTRUCTIONS_CHARS + 2000)
        if not text.startswith("---"):
            continue
        name = candidate.parent.name
        description = ""
        for line in text.splitlines()[1:]:
            if line == "---":
                break
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip('"') or name
            elif line.startswith("description:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    description = str(json.loads(raw))
                except (json.JSONDecodeError, TypeError):
                    description = raw.strip('"')
        record = usage.get(candidate.parent.name, {})
        current_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        recorded_hash = record.get("content_sha256")
        skills.append(
            {
                "name": name,
                "description": description,
                "path": safe.relative_to(workspace).as_posix(),
                "state": "archived"
                if location_state == "archived"
                else record.get("state", "active"),
                "pinned": bool(record.get("pinned", False)),
                "use_count": int(record.get("use_count", 0)),
                "view_count": int(record.get("view_count", 0)),
                "revision_count": int(record.get("revision_count", 0)),
                "last_used_at": record.get("last_used_at"),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "drifted": bool(recorded_hash and recorded_hash != current_hash),
            }
        )
    return skills


def read_learned_skill(cwd: str, name: str) -> str:
    slug = _skill_slug(name)
    root = _learned_skills_root(cwd)
    path = _safe_path(Path(cwd), root / slug / "SKILL.md")
    text = _bounded_read(path, MAX_SKILL_INSTRUCTIONS_CHARS + 2000) if path else ""
    if not text:
        raise ValueError(f"Learned skill not found: {slug}")
    usage = _load_skill_usage(root)
    record = _skill_record(usage, slug)
    record["view_count"] = int(record.get("view_count", 0)) + 1
    record["use_count"] = int(record.get("use_count", 0)) + 1
    record["last_used_at"] = _now_iso()
    record["state"] = "active"
    _save_skill_usage(root, usage)
    return text


def write_learned_skill(cwd: str, name: str, description: str, instructions: str) -> Path:
    """Atomically create or refine a verified, agent-owned project skill."""
    slug = _skill_slug(name)
    clean_description = _validate_learning_text(
        description, "Skill description", MAX_SKILL_DESCRIPTION_CHARS
    )
    clean_instructions = _validate_learning_text(
        instructions, "Skill instructions", MAX_SKILL_INSTRUCTIONS_CHARS
    )
    root = _learned_skills_root(cwd, create=True)
    usage = _load_skill_usage(root)
    skill_dir = _safe_path(Path(cwd), root / slug)
    if skill_dir is None:
        raise ValueError("Learned skill path escapes the workspace")
    archived_path = _safe_path(Path(cwd), root / SKILL_ARCHIVE_DIRNAME / slug / "SKILL.md")
    if archived_path is not None and archived_path.is_file():
        raise ValueError("Restore the archived skill before refining it")
    if not skill_dir.exists() and len(list_learned_skills(cwd)) >= MAX_LEARNED_SKILLS:
        raise ValueError("Learned skill limit reached; remove or consolidate a skill first")
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = _safe_path(Path(cwd), skill_dir / "SKILL.md")
    if path is None:
        raise ValueError("Learned skill file escapes the workspace")
    title = " ".join(word.capitalize() for word in slug.split("-"))
    content = (
        "---\n"
        f"name: {slug}\n"
        f"description: {json.dumps(clean_description, ensure_ascii=False)}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{clean_instructions}\n\n"
        "## Provenance\n\n"
        "Learned by GLM ACP after successful task verification on "
        f"{datetime.now(timezone.utc).date().isoformat()}.\n"
    )
    _atomic_write(path, content)
    record = _skill_record(usage, slug)
    record["updated_at"] = _now_iso()
    record["revision_count"] = int(record.get("revision_count", 0)) + 1
    record["state"] = "active"
    record["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    _save_skill_usage(root, usage)
    return path


def forget_learned_skill(cwd: str, name: str) -> Path:
    """Remove only a skill owned by the GLM ACP learning directory."""
    slug = _skill_slug(name)
    root = _learned_skills_root(cwd)
    active_path = _safe_path(Path(cwd), root / slug / "SKILL.md")
    archived_path = _safe_path(Path(cwd), root / SKILL_ARCHIVE_DIRNAME / slug / "SKILL.md")
    path = active_path if active_path is not None and active_path.is_file() else archived_path
    if path is None or not path.is_file():
        raise ValueError(f"Learned skill not found: {slug}")
    shutil.rmtree(path.parent)
    usage = _load_skill_usage(root)
    usage.pop(slug, None)
    _save_skill_usage(root, usage)
    return path


def manage_learned_skill(cwd: str, name: str, action: str) -> dict[str, Any]:
    """Apply a reversible lifecycle action to one agent-owned skill."""
    slug = _skill_slug(name)
    normalized_action = action.strip().lower()
    if normalized_action not in {"pin", "unpin", "archive", "restore"}:
        raise ValueError("Skill action must be pin, unpin, archive, or restore")
    root = _learned_skills_root(cwd, create=True)
    usage = _load_skill_usage(root)
    record = _skill_record(usage, slug)
    active_dir = _safe_path(Path(cwd), root / slug)
    archive_root = _safe_path(Path(cwd), root / SKILL_ARCHIVE_DIRNAME)
    archived_dir = _safe_path(Path(cwd), root / SKILL_ARCHIVE_DIRNAME / slug)
    if active_dir is None or archive_root is None or archived_dir is None:
        raise ValueError("Skill lifecycle path escapes the workspace")
    active_exists = (active_dir / "SKILL.md").is_file()
    archived_exists = (archived_dir / "SKILL.md").is_file()
    if not active_exists and not archived_exists:
        raise ValueError(f"Learned skill not found: {slug}")

    if normalized_action == "pin":
        if not active_exists:
            raise ValueError(f"Active learned skill not found: {slug}")
        record["pinned"] = True
    elif normalized_action == "unpin":
        record["pinned"] = False
    elif normalized_action == "archive":
        if record.get("pinned"):
            raise ValueError("Pinned skills cannot be archived")
        if not active_exists:
            raise ValueError(f"Active learned skill not found: {slug}")
        archive_root.mkdir(parents=True, exist_ok=True)
        if archived_dir.exists():
            raise ValueError(f"Archived skill already exists: {slug}")
        os.replace(active_dir, archived_dir)
        record["state"] = "archived"
    elif normalized_action == "restore":
        if not archived_exists:
            raise ValueError(f"Archived learned skill not found: {slug}")
        if active_dir.exists():
            raise ValueError(f"Active learned skill already exists: {slug}")
        os.replace(archived_dir, active_dir)
        record["state"] = "active"
        record["last_used_at"] = _now_iso()
    record["updated_at"] = _now_iso()
    _save_skill_usage(root, usage)
    return {"name": slug, **record}


def curate_learned_skills(cwd: str, *, now: datetime | None = None) -> dict[str, list[str]]:
    """Deterministically mark idle skills stale and archive old unpinned skills."""
    root = _learned_skills_root(cwd, create=True)
    usage = _load_skill_usage(root)
    current = now or datetime.now(timezone.utc)
    result: dict[str, list[str]] = {
        "stale": [],
        "archived": [],
        "kept": [],
        "review": [],
    }
    for skill in list_learned_skills(cwd):
        if skill["state"] == "archived":
            continue
        slug = skill["name"]
        record = _skill_record(usage, slug)
        if skill["drifted"]:
            result["review"].append(slug)
            continue
        if record.get("pinned"):
            result["kept"].append(slug)
            continue
        timestamp = record.get("last_used_at") or record.get("updated_at") or record["created_at"]
        try:
            last_activity = datetime.fromisoformat(str(timestamp))
        except ValueError:
            last_activity = current
        if last_activity.tzinfo is None:
            last_activity = last_activity.replace(tzinfo=timezone.utc)
        idle_days = (current - last_activity).days
        if idle_days >= SKILL_ARCHIVE_AFTER_DAYS:
            _save_skill_usage(root, usage)
            manage_learned_skill(cwd, slug, "archive")
            usage = _load_skill_usage(root)
            result["archived"].append(slug)
        elif idle_days >= SKILL_STALE_AFTER_DAYS:
            record["state"] = "stale"
            result["stale"].append(slug)
        else:
            result["kept"].append(slug)
    _save_skill_usage(root, usage)
    return result


def skill_curator_status(cwd: str) -> dict[str, Any]:
    skills = list_learned_skills(cwd)
    current = datetime.now(timezone.utc)
    due_stale: list[str] = []
    due_archive: list[str] = []
    for skill in skills:
        if skill["state"] == "archived" or skill["pinned"]:
            continue
        timestamp = skill.get("last_used_at") or skill.get("updated_at") or skill.get("created_at")
        if not timestamp:
            continue
        try:
            last_activity = datetime.fromisoformat(str(timestamp))
        except ValueError:
            continue
        if last_activity.tzinfo is None:
            last_activity = last_activity.replace(tzinfo=timezone.utc)
        idle_days = (current - last_activity).days
        if idle_days >= SKILL_ARCHIVE_AFTER_DAYS:
            due_archive.append(skill["name"])
        elif idle_days >= SKILL_STALE_AFTER_DAYS:
            due_stale.append(skill["name"])
    overlap_candidates: list[dict[str, Any]] = []
    active_skills = [skill for skill in skills if skill["state"] != "archived"]
    for index, first in enumerate(active_skills):
        first_terms = set(re.findall(r"\b[a-z0-9-]{4,}\b", first["description"].lower()))
        for second in active_skills[index + 1 :]:
            second_terms = set(re.findall(r"\b[a-z0-9-]{4,}\b", second["description"].lower()))
            shared = first_terms & second_terms
            combined = first_terms | second_terms
            score = len(shared) / len(combined) if combined else 0.0
            if len(shared) >= 3 and score >= 0.35:
                overlap_candidates.append(
                    {
                        "skills": [first["name"], second["name"]],
                        "score": round(score, 2),
                    }
                )
    return {
        "total": len(skills),
        "active": sum(skill["state"] == "active" for skill in skills),
        "stale": sum(skill["state"] == "stale" for skill in skills),
        "archived": sum(skill["state"] == "archived" for skill in skills),
        "pinned": sum(bool(skill["pinned"]) for skill in skills),
        "due_stale": due_stale,
        "due_archive": due_archive,
        "drifted": [skill["name"] for skill in skills if skill["drifted"]],
        "overlap_candidates": overlap_candidates[:20],
        "skills": skills,
    }
