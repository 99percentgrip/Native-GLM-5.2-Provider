"""Persistent, cross-process-safe scheduled job management."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from .config import config_dir
from .security import scan_promptware

try:  # pragma: no cover - selected by platform
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
try:  # pragma: no cover - selected by platform
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None

STORE_VERSION = 1
MAX_JOBS = 500
MAX_PROMPT_CHARS = 32_000
MAX_NAME_CHARS = 120
MAX_HISTORY = 50
CLAIM_TTL_SECONDS = 1800
_DURATION_RE = re.compile(r"^(\d+)\s*(s|m|h|d)$", re.IGNORECASE)
_SECRET_RE = re.compile(r"(?i)(?:api[_ -]?key|token|password|secret|private[_ -]?key)\s*[:=]\s*\S+")
_thread_lock = threading.RLock()
_lock_state = threading.local()


class CronError(ValueError):
    """A safe, user-facing cron validation or persistence error."""


def cron_dir() -> Path:
    return config_dir() / "cron"


def jobs_path() -> Path:
    return cron_dir() / "jobs.json"


def results_dir() -> Path:
    return cron_dir() / "results"


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _secure_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def ensure_dirs() -> None:
    _secure_dir(cron_dir())
    _secure_dir(results_dir())


@contextlib.contextmanager
def store_lock() -> Iterator[None]:
    """Serialize complete read/modify/write transactions across processes."""
    depth = getattr(_lock_state, "depth", 0)
    if depth:
        _lock_state.depth = depth + 1
        try:
            yield
        finally:
            _lock_state.depth -= 1
        return
    with _thread_lock:
        ensure_dirs()
        handle = (cron_dir() / ".jobs.lock").open("a+b")
        _lock_state.depth = 1
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                handle.close()
                _lock_state.depth = 0


def _atomic_json(path: Path, payload: Any) -> None:
    _secure_dir(path.parent)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _secure_file(temporary)
        os.replace(temporary, path)
        _secure_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_unlocked() -> list[dict[str, Any]]:
    try:
        payload = json.loads(jobs_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as error:
        raise CronError(f"Cannot read cron store: {error}") from error
    if not isinstance(payload, dict) or payload.get("version") != STORE_VERSION:
        raise CronError("Unsupported or malformed cron store")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not all(isinstance(job, dict) for job in jobs):
        raise CronError("Malformed cron jobs list")
    return jobs


def _save_unlocked(jobs: list[dict[str, Any]]) -> None:
    _atomic_json(jobs_path(), {"version": STORE_VERSION, "jobs": jobs})


def _aware(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise CronError("Cron time calculations require timezone-aware datetimes")
    return value


def _duration_seconds(value: str) -> int:
    match = _DURATION_RE.fullmatch(value.strip())
    if not match:
        raise CronError("Duration must use Ns, Nm, Nh, or Nd (for example 30m)")
    amount = int(match.group(1))
    if amount <= 0:
        raise CronError("Duration must be greater than zero")
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2).lower()]
    seconds = amount * multiplier
    if seconds > 366 * 86400:
        raise CronError("Duration cannot exceed 366 days")
    return seconds


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise CronError(f"Unknown timezone: {name}") from error


def parse_schedule(
    value: str, *, timezone_name: str = "UTC", now: datetime | None = None
) -> dict[str, Any]:
    """Parse a delay, interval, strict five-field cron, or aware ISO one-shot."""
    raw = str(value or "").strip()
    if not raw:
        raise CronError("Schedule is required")
    current = _aware(now)
    lowered = raw.lower()
    if lowered.startswith("every "):
        seconds = _duration_seconds(raw[6:])
        return {"kind": "interval", "seconds": seconds, "display": f"every {raw[6:].strip()}"}
    delay = raw[3:].strip() if lowered.startswith("in ") else raw
    if _DURATION_RE.fullmatch(delay):
        seconds = _duration_seconds(delay)
        return {
            "kind": "once",
            "run_at": (current + timedelta(seconds=seconds)).isoformat(),
            "display": f"in {delay}",
        }
    fields = raw.split()
    if len(fields) == 5:
        if not croniter.is_valid(raw):
            raise CronError(f"Invalid five-field cron expression: {raw}")
        _zone(timezone_name)
        return {"kind": "cron", "expr": raw, "timezone": timezone_name, "display": raw}
    if len(fields) in {6, 7}:
        raise CronError("Only standard five-field cron expressions are supported")
    try:
        run_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise CronError(
            "Schedule must be a delay, interval, five-field cron, or ISO timestamp"
        ) from error
    if run_at.tzinfo is None or run_at.utcoffset() is None:
        raise CronError("ISO one-shot schedules must include a timezone offset")
    if run_at <= current:
        raise CronError("ISO one-shot schedule must be in the future")
    return {"kind": "once", "run_at": run_at.isoformat(), "display": raw}


def next_run(schedule: dict[str, Any], *, now: datetime | None = None) -> str | None:
    """Compute one future run; recurring schedules deliberately skip missed slots."""
    current = _aware(now)
    kind = schedule.get("kind")
    if kind == "once":
        return str(schedule.get("run_at") or "") or None
    if kind == "interval":
        return (current + timedelta(seconds=int(schedule["seconds"]))).isoformat()
    if kind == "cron":
        zone = _zone(str(schedule.get("timezone") or "UTC"))
        base = current.astimezone(zone)
        return croniter(str(schedule["expr"]), base).get_next(datetime).isoformat()
    raise CronError("Unknown schedule kind")


def _validate_text(value: Any, label: str, limit: int, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise CronError(f"{label} is required")
    if len(text) > limit:
        raise CronError(f"{label} exceeds {limit:,} characters")
    if _SECRET_RE.search(text):
        raise CronError(f"{label} appears to contain a credential and was not stored")
    findings = scan_promptware(text)
    if findings:
        raise CronError(
            f"{label} was blocked by promptware defense: "
            + ", ".join(finding.code for finding in findings)
        )
    return text


def _redact(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


def _contained(path: str | Path, root: Path, label: str, *, must_file: bool = False) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise CronError(f"{label} must stay inside the workspace root") from error
    if must_file and (not candidate.is_file() or candidate.is_symlink()):
        raise CronError(f"{label} must be a regular, non-symlink file")
    if not must_file and not candidate.is_dir():
        raise CronError(f"{label} must be an existing directory")
    return candidate


def create_job(
    *,
    schedule: str,
    prompt: str = "",
    name: str | None = None,
    workspace_root: str,
    workdir: str | None = None,
    timezone_name: str = "UTC",
    repeat: int | None = None,
    skills: list[str] | None = None,
    bundles: list[str] | None = None,
    script: str | None = None,
    no_agent: bool = False,
    origin_session_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise CronError("Workspace root must be an existing directory")
    working = _contained(workdir or root, root, "Workdir")
    parsed = parse_schedule(schedule, timezone_name=timezone_name, now=now)
    clean_prompt = _validate_text(prompt, "Prompt", MAX_PROMPT_CHARS, allow_empty=no_agent)
    if no_agent and not script:
        raise CronError("Script-only jobs require a script")
    script_path = str(_contained(script, working, "Script", must_file=True)) if script else None
    if repeat is not None and (not isinstance(repeat, int) or repeat <= 0 or repeat > 1_000_000):
        raise CronError("Repeat must be an integer between 1 and 1,000,000")
    recurring = parsed["kind"] in {"interval", "cron"}
    limit = repeat if repeat is not None else (None if recurring else 1)
    created = _aware(now).isoformat()
    job = {
        "id": uuid4().hex[:12],
        "name": _validate_text(
            name or clean_prompt[:50] or Path(script_path or "job").name, "Name", MAX_NAME_CHARS
        ),
        "prompt": clean_prompt,
        "schedule": parsed,
        "schedule_display": parsed["display"],
        "timezone": timezone_name,
        "workspace_root": str(root),
        "workdir": str(working),
        "skills": list(
            dict.fromkeys(str(item).strip() for item in (skills or []) if str(item).strip())
        )[:12],
        "bundles": list(
            dict.fromkeys(str(item).strip() for item in (bundles or []) if str(item).strip())
        )[:12],
        "script": script_path,
        "no_agent": bool(no_agent),
        "enabled": True,
        "state": "scheduled",
        "repeat_limit": limit,
        "run_count": 0,
        "success_count": 0,
        "created_at": created,
        "updated_at": created,
        "next_run_at": next_run(parsed, now=now),
        "last_run_at": None,
        "last_finished_at": None,
        "last_status": None,
        "last_error": None,
        "last_output_path": None,
        "claim": None,
        "history": [],
        "origin_session_id": origin_session_id,
    }
    with store_lock():
        jobs = _load_unlocked()
        if len(jobs) >= MAX_JOBS:
            raise CronError(f"Cron store is limited to {MAX_JOBS} jobs")
        jobs.append(job)
        _save_unlocked(jobs)
    return dict(job)


def list_jobs(*, include_disabled: bool = True) -> list[dict[str, Any]]:
    with store_lock():
        jobs = _load_unlocked()
    return [dict(job) for job in jobs if include_disabled or job.get("enabled")]


def get_job(job_id: str) -> dict[str, Any] | None:
    return next((job for job in list_jobs() if job.get("id") == job_id), None)


def _mutate(job_id: str, callback: Any) -> dict[str, Any]:
    with store_lock():
        jobs = _load_unlocked()
        for job in jobs:
            if job.get("id") == job_id:
                callback(job)
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_unlocked(jobs)
                return dict(job)
    raise CronError(f"Cron job not found: {job_id}")


def update_job(
    job_id: str, updates: dict[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    allowed = {
        "name",
        "prompt",
        "schedule",
        "timezone",
        "repeat",
        "skills",
        "bundles",
        "script",
        "no_agent",
        "workdir",
    }
    unknown = set(updates) - allowed
    if unknown:
        raise CronError(f"Unsupported update fields: {', '.join(sorted(unknown))}")
    current = get_job(job_id)
    if current is None:
        raise CronError(f"Cron job not found: {job_id}")
    if current.get("state") == "running":
        raise CronError("Running jobs cannot be updated; wait for the current run to finish")
    merged = dict(current)
    merged.update(updates)
    if isinstance(merged.get("schedule"), str):
        parsed = parse_schedule(
            str(merged["schedule"]),
            timezone_name=str(merged.get("timezone") or "UTC"),
            now=now,
        )
    elif "timezone" in updates and current["schedule"].get("kind") == "cron":
        parsed = parse_schedule(
            str(current["schedule"]["expr"]),
            timezone_name=str(merged.get("timezone") or "UTC"),
            now=now,
        )
    else:
        parsed = current["schedule"]
    root = Path(current["workspace_root"]).resolve()
    working = _contained(str(merged.get("workdir") or root), root, "Workdir")
    prompt = _validate_text(
        merged.get("prompt"), "Prompt", MAX_PROMPT_CHARS, allow_empty=bool(merged.get("no_agent"))
    )
    script = merged.get("script")
    script_path = (
        str(_contained(str(script), working, "Script", must_file=True)) if script else None
    )
    if merged.get("no_agent") and not script_path:
        raise CronError("Script-only jobs require a script")
    repeat = merged.get("repeat", current.get("repeat_limit"))
    if repeat is not None and (not isinstance(repeat, int) or repeat <= 0 or repeat > 1_000_000):
        raise CronError("Repeat must be an integer between 1 and 1,000,000")

    def apply(job: dict[str, Any]) -> None:
        if "name" in updates:
            job["name"] = _validate_text(merged["name"], "Name", MAX_NAME_CHARS)
        for field in ("skills", "bundles"):
            if field in updates:
                values = merged[field]
                if not isinstance(values, list):
                    raise CronError(f"{field} must be a list")
                job[field] = list(
                    dict.fromkeys(str(item).strip() for item in values if str(item).strip())
                )[:12]
        job.update(
            {
                "prompt": prompt,
                "workdir": str(working),
                "script": script_path,
                "no_agent": bool(merged.get("no_agent")),
                "repeat_limit": repeat,
            }
        )
        if "schedule" in updates or "timezone" in updates:
            job["schedule"] = parsed
            job["schedule_display"] = parsed["display"]
            job["timezone"] = str(merged.get("timezone") or "UTC")
            job["next_run_at"] = next_run(parsed, now=now)

    return _mutate(job_id, apply)


def pause_job(job_id: str) -> dict[str, Any]:
    def apply(job: dict[str, Any]) -> None:
        # Do not invalidate an in-flight claim. The current run finishes once,
        # then finish_job records it as paused and prevents the next dispatch.
        job["enabled"] = False
        if job.get("state") != "running":
            job["state"] = "paused"

    return _mutate(job_id, apply)


def resume_job(job_id: str, *, now: datetime | None = None) -> dict[str, Any]:
    def apply(job: dict[str, Any]) -> None:
        if job.get("state") == "running":
            raise CronError("Running jobs cannot be resumed")
        if job.get("state") == "completed":
            raise CronError("Completed jobs cannot be resumed; increase repeat or create a new job")
        job.update(enabled=True, state="scheduled", claim=None)
        job["next_run_at"] = next_run(job["schedule"], now=now)

    return _mutate(job_id, apply)


def remove_job(job_id: str) -> bool:
    with store_lock():
        jobs = _load_unlocked()
        for job in jobs:
            if job.get("id") == job_id and job.get("state") == "running":
                raise CronError("Running jobs cannot be removed; pause and wait for completion")
        remaining = [job for job in jobs if job.get("id") != job_id]
        if len(remaining) == len(jobs):
            return False
        _save_unlocked(remaining)
    return True


def claim_due(
    *, now: datetime | None = None, job_id: str | None = None, force: bool = False
) -> list[dict[str, Any]]:
    """Atomically claim due work and advance recurring schedules from now."""
    current = _aware(now)
    claimed: list[dict[str, Any]] = []
    with store_lock():
        jobs = _load_unlocked()
        changed = False
        for job in jobs:
            if job_id and job.get("id") != job_id:
                continue
            claim = job.get("claim") or {}
            expires = claim.get("expires_at")
            claimed_at = claim.get("claimed_at")
            claim_is_live = False
            if expires and claimed_at:
                try:
                    claim_is_live = (
                        datetime.fromisoformat(claimed_at)
                        <= current
                        < datetime.fromisoformat(expires)
                    )
                except ValueError:
                    pass
            if claim_is_live:
                continue
            recovering_once = (
                bool(claim)
                and job.get("state") == "running"
                and job.get("schedule", {}).get("kind") == "once"
            )
            if claim and not job.get("enabled"):
                job.update(claim=None, state="paused")
                changed = True
                continue
            if claim and not recovering_once:
                job.update(claim=None, state="scheduled")
                changed = True
            due_text = job.get("next_run_at")
            due = datetime.fromisoformat(due_text) if due_text else None
            if not job.get("enabled"):
                continue
            if not force and not recovering_once and (due is None or due > current):
                continue
            if force and job.get("state") == "completed":
                continue
            token = uuid4().hex
            job["claim"] = {
                "token": token,
                "claimed_at": current.isoformat(),
                "expires_at": (current + timedelta(seconds=CLAIM_TTL_SECONDS)).isoformat(),
            }
            job["state"] = "running"
            if job["schedule"]["kind"] == "once":
                job["next_run_at"] = None
            else:
                job["next_run_at"] = next_run(job["schedule"], now=current)
            job["last_run_at"] = current.isoformat()
            claimed.append(dict(job))
            changed = True
            if job_id:
                break
        if changed:
            _save_unlocked(jobs)
    return claimed


def renew_claim(job_id: str, token: str, *, now: datetime | None = None) -> bool:
    """Extend a live claim without reviving or stealing another runner's claim."""
    current = _aware(now)
    renewed = False

    def apply(job: dict[str, Any]) -> None:
        nonlocal renewed
        claim = job.get("claim") or {}
        if job.get("state") != "running" or claim.get("token") != token:
            return
        claim["expires_at"] = (current + timedelta(seconds=CLAIM_TTL_SECONDS)).isoformat()
        job["claim"] = claim
        renewed = True

    try:
        _mutate(job_id, apply)
    except CronError:
        return False
    return renewed


def finish_job(
    job_id: str,
    token: str,
    *,
    status: str,
    output: str = "",
    error: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _aware(now)
    if status not in {"ok", "error", "cancelled", "silent"}:
        raise CronError("Invalid run status")
    artifact = save_output(job_id, token, status=status, output=output, error=error, now=current)

    def apply(job: dict[str, Any]) -> None:
        claim = job.get("claim") or {}
        if claim.get("token") != token:
            raise CronError("Cron claim no longer belongs to this run")
        job["claim"] = None
        job["run_count"] = int(job.get("run_count") or 0) + 1
        if status in {"ok", "silent"}:
            job["success_count"] = int(job.get("success_count") or 0) + 1
        job["last_finished_at"] = current.isoformat()
        job["last_status"] = status
        job["last_error"] = _redact(str(error)[:2000]) if error else None
        job["last_output_path"] = str(artifact)
        entry = {
            "run_at": job.get("last_run_at"),
            "finished_at": current.isoformat(),
            "status": status,
            "error": job["last_error"],
            "output_path": str(artifact),
        }
        job["history"] = [*(job.get("history") or []), entry][-MAX_HISTORY:]
        limit = job.get("repeat_limit")
        if job["schedule"]["kind"] == "once" or (
            limit is not None and job["run_count"] >= int(limit)
        ):
            job.update(enabled=False, state="completed", next_run_at=None)
        elif job.get("enabled"):
            job["state"] = "scheduled"
        else:
            job["state"] = "paused"

    return _mutate(job_id, apply)


def save_output(
    job_id: str, token: str, *, status: str, output: str, error: str | None, now: datetime
) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", job_id) or not re.fullmatch(r"[a-f0-9]{32}", token):
        raise CronError("Unsafe cron artifact identifier")
    directory = results_dir() / job_id
    _secure_dir(directory)
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"{stamp}-{token[:8]}.json"
    _atomic_json(
        path,
        {
            "job_id": job_id,
            "run_token": token,
            "finished_at": now.isoformat(),
            "status": status,
            "output": _redact(output[:256_000]),
            "error": _redact(str(error)[:8000]) if error else None,
        },
    )
    return path


def status() -> dict[str, Any]:
    jobs = list_jobs()
    active = [job for job in jobs if job.get("enabled")]
    next_times = [job["next_run_at"] for job in active if job.get("next_run_at")]
    heartbeat = cron_dir() / "daemon-heartbeat"
    age = None
    try:
        age = max(0.0, datetime.now().timestamp() - float(heartbeat.read_text()))
    except (OSError, ValueError):
        pass
    return {
        "jobs": len(jobs),
        "active": len(active),
        "running": sum(job.get("state") == "running" for job in jobs),
        "next_run_at": min(next_times) if next_times else None,
        "heartbeat_age_seconds": age,
    }
