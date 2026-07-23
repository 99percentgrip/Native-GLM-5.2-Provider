"""Metadata-only failure drafts and permission-gated runnable regression cases."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config_dir
from .security import scan_promptware

MAX_DRAFTS = 2_000
MAX_CASE_FILES = 24
MAX_CASE_BYTES = 256_000
_CASE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}")
_SECRET = re.compile(r"(?i)(?:api[_-]?key|token|secret|password|credential)\s*[:=]")


class FailureCorpusError(RuntimeError):
    pass


def _failure_kind(error: str) -> str:
    lowered = error.lower()
    for kind, markers in {
        "timeout": ("timed out", "timeout"),
        "permission": ("permission", "denied", "not allowed"),
        "conflict": ("conflict", "hash mismatch", "changed after"),
        "syntax": ("syntax", "parse", "invalid json"),
        "network": ("network", "connection", "http", "429", "503"),
        "verification": ("test", "assert", "exit code"),
        "tool_contract": ("argument", "schema", "required field", "unknown tool"),
    }.items():
        if any(marker in lowered for marker in markers):
            return kind
    return "other"


class FailureCorpus:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config_dir() / "failure-corpus" / "drafts.jsonl"

    def _read(self) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_bytes().splitlines()[-MAX_DRAFTS:]
        except OSError:
            return []
        output: list[dict[str, Any]] = []
        for line in lines:
            if len(line) > 4096:
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("schema") == 1:
                output.append(value)
        return output

    def record_draft(
        self, workspace: str, tool: str, error: str, paths: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Record only a normalized failure class and non-identifying fingerprints."""
        kind = _failure_kind(error)
        extensions = sorted(
            {Path(value).suffix.lower() for value in paths or [] if Path(value).suffix}
        )
        project = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]
        fingerprint = hashlib.sha256(
            json.dumps([project, tool[:100], kind, extensions], separators=(",", ":")).encode()
        ).hexdigest()[:24]
        existing = self._read()
        if any(item.get("fingerprint") == fingerprint for item in existing[-200:]):
            return None
        payload = {
            "schema": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "project": project,
            "tool": tool[:100],
            "failure_kind": kind,
            "extensions": extensions[:20],
            "status": "draft",
        }
        raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, raw)
                if os.name != "nt":
                    os.chmod(self.path, 0o600)
            finally:
                os.close(descriptor)
        except OSError:
            return None
        return payload

    def list(self) -> list[dict[str, Any]]:
        return list(reversed(self._read()))[:200]

    def project_metadata(self, workspace: str) -> list[dict[str, Any]]:
        """Return bounded non-identifying failure classes for one repository."""
        project = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]
        return [
            {
                "failure_kind": str(item.get("failure_kind", "other"))[:40],
                "tool": str(item.get("tool", ""))[:100],
            }
            for item in self._read()
            if item.get("project") == project
        ][-200:]

    def discard(self, fingerprint: str) -> bool:
        values = self._read()
        kept = [item for item in values if item.get("fingerprint") != fingerprint]
        if len(kept) == len(values):
            return False
        self._write_drafts(kept)
        return True

    def _write_drafts(self, values: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            temporary.write_text(
                "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values),
                encoding="utf-8",
            )
            if os.name != "nt":
                os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def promote(self, workspace: str, fingerprint: str, case: dict[str, Any]) -> Path:
        """Promote a reviewed draft into an outcome-based benchmark case."""
        draft = next(
            (item for item in self._read() if item.get("fingerprint") == fingerprint), None
        )
        if draft is None:
            raise FailureCorpusError("Failure draft was not found")
        case_id = str(case.get("id", ""))
        prompt = str(case.get("prompt", ""))
        files = case.get("files")
        verify = case.get("verify")
        timeout = case.get("timeout", 180)
        if not _CASE_ID.fullmatch(case_id):
            raise FailureCorpusError("Regression case id is invalid")
        if not prompt or len(prompt) > 4_000 or scan_promptware(prompt) or _SECRET.search(prompt):
            raise FailureCorpusError("Regression prompt is empty, unsafe, or too large")
        if not isinstance(files, dict) or not 1 <= len(files) <= MAX_CASE_FILES:
            raise FailureCorpusError(f"Regression case requires 1-{MAX_CASE_FILES} fixture files")
        normalized_files: dict[str, str] = {}
        total = 0
        for relative, content in files.items():
            if not isinstance(relative, str) or not isinstance(content, str):
                raise FailureCorpusError("Regression fixture files must be string pairs")
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.name.startswith(".env"):
                raise FailureCorpusError(f"Unsafe regression fixture path: {relative}")
            if scan_promptware(content) or _SECRET.search(content):
                raise FailureCorpusError(f"Unsafe regression fixture content: {relative}")
            total += len(content.encode("utf-8"))
            if total > MAX_CASE_BYTES:
                raise FailureCorpusError("Regression fixtures exceed 256 KiB")
            normalized_files[path.as_posix()] = content
        if (
            not isinstance(verify, list)
            or not 1 <= len(verify) <= 16
            or not all(isinstance(value, str) and 0 < len(value) <= 500 for value in verify)
        ):
            raise FailureCorpusError("Regression verification must be a bounded argv list")
        if _SECRET.search(" ".join(verify)):
            raise FailureCorpusError("Regression verification appears to contain a secret")
        if not isinstance(timeout, int) or not 30 <= timeout <= 900:
            raise FailureCorpusError("Regression timeout must be between 30 and 900 seconds")
        root = Path(workspace).resolve()
        target = root / ".glm-acp" / "evaluation" / "failure-cases.json"
        try:
            existing = json.loads(target.read_text(encoding="utf-8")) if target.exists() else []
        except (OSError, json.JSONDecodeError) as error:
            raise FailureCorpusError(f"Existing failure corpus is invalid: {error}") from error
        if not isinstance(existing, list):
            raise FailureCorpusError("Existing failure corpus must contain a JSON array")
        if any(item.get("id") == case_id for item in existing if isinstance(item, dict)):
            raise FailureCorpusError(f"Regression case id already exists: {case_id}")
        existing.append(
            {
                "id": case_id,
                "prompt": prompt,
                "files": normalized_files,
                "verify": verify,
                "timeout": timeout,
                "source_failure": fingerprint,
                "failure_kind": draft.get("failure_kind", "other"),
            }
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            temporary.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self.discard(fingerprint)
        return target
