"""Persistent session storage.

Saves and loads session state (message history, model, mode, etc.) to
disk as JSON files so conversations survive agent-process restarts.

When Zed restarts and calls ``load_session`` with a previously-issued
``session_id``, the agent can rebuild the exact same conversation state
instead of starting from scratch.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("glm_acp")

# Directory for persisted sessions.  We use a hidden folder in the user's
# home directory so it is stable across process restarts (unlike /tmp which
# may be cleared) yet still easy to find/inspect.
SESSION_DIR = Path(os.path.expanduser("~/.glm-acp/sessions"))
SESSION_PERSISTENCE_ENV = "GLM_ACP_SESSION_PERSISTENCE"


def session_persistence_enabled() -> bool:
    return os.environ.get(SESSION_PERSISTENCE_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Save / load serialized session state to individual JSON files."""

    def __init__(self, base_dir: Path = SESSION_DIR) -> None:
        self._base_dir = base_dir

    def _path(self, session_id: str) -> Path:
        """Return the on-disk path for a session id.

        Sanitizes the session id to prevent path traversal — only
        alphanumeric, dash, and underscore are allowed in filenames.
        """
        import re

        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        return self._base_dir / f"{safe_id}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist *data* for *session_id* atomically.

        A ``saved_at`` timestamp is injected so ``list_sessions`` can sort
        by recency.
        """
        if not session_persistence_enabled():
            return
        data = {**data, "saved_at": _now_iso()}
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                os.chmod(self._base_dir, 0o700)
            path = self._path(session_id)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            # Atomic rename so a crash mid-write never leaves a corrupt file.
            os.replace(tmp, path)
            if os.name != "nt":
                os.chmod(path, 0o600)
            metadata = {
                "session_id": path.stem,
                "cwd": data.get("cwd", ""),
                "title": data.get("title"),
                "updated_at": data.get("saved_at"),
            }
            meta_path = path.with_suffix(".meta")
            meta_tmp = meta_path.with_suffix(".meta.tmp")
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, ensure_ascii=False)
            os.replace(meta_tmp, meta_path)
            if os.name != "nt":
                os.chmod(meta_path, 0o600)
        except OSError:
            logger.warning("Could not persist session %s", session_id, exc_info=True)

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Return the stored data for *session_id* or ``None`` if absent."""
        if not session_persistence_enabled():
            return None
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not load session %s", session_id, exc_info=True)
            return None

    def list(self) -> list[dict[str, Any]]:
        """Return metadata for all persisted sessions, most-recent first."""
        if not session_persistence_enabled():
            return []
        results: list[dict[str, Any]] = []
        if not self._base_dir.exists():
            return results
        indexed_ids: set[str] = set()
        for path in self._base_dir.glob("*.meta"):
            try:
                with open(path, encoding="utf-8") as fh:
                    metadata = json.load(fh)
                results.append(metadata)
                indexed_ids.add(path.stem)
            except (OSError, json.JSONDecodeError):
                continue
        # Backward compatibility for sessions written before metadata sidecars.
        for path in self._base_dir.glob("*.json"):
            if path.stem in indexed_ids:
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                session_id = path.stem
                results.append(
                    {
                        "session_id": session_id,
                        "cwd": data.get("cwd", ""),
                        "title": data.get("title"),
                        "updated_at": data.get("saved_at"),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
        # Sort by updated_at descending (most recent first)
        results.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return results

    def delete(self, session_id: str) -> None:
        """Remove the stored data for *session_id* (best-effort)."""
        path = self._path(session_id)
        try:
            path.unlink(missing_ok=True)
            path.with_suffix(".meta").unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete session %s", session_id, exc_info=True)
