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


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Save / load serialized session state to individual JSON files."""

    def __init__(self, base_dir: Path = SESSION_DIR) -> None:
        self._base_dir = base_dir

    def _path(self, session_id: str) -> Path:
        """Return the on-disk path for a session id."""
        return self._base_dir / f"{session_id}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist *data* for *session_id* atomically.

        A ``saved_at`` timestamp is injected so ``list_sessions`` can sort
        by recency.
        """
        data = {**data, "saved_at": _now_iso()}
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            path = self._path(session_id)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            # Atomic rename so a crash mid-write never leaves a corrupt file.
            os.replace(tmp, path)
        except OSError:
            logger.warning("Could not persist session %s", session_id, exc_info=True)

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Return the stored data for *session_id* or ``None`` if absent."""
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not load session %s", session_id, exc_info=True)
            return None

    def list(self) -> list[dict[str, Any]]:
        """Return metadata for all persisted sessions, most-recent first."""
        results: list[dict[str, Any]] = []
        if not self._base_dir.exists():
            return results
        for path in self._base_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                session_id = path.stem
                results.append({
                    "session_id": session_id,
                    "cwd": data.get("cwd", ""),
                    "title": data.get("title"),
                    "updated_at": data.get("saved_at"),
                })
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
        except OSError:
            logger.warning("Could not delete session %s", session_id, exc_info=True)
