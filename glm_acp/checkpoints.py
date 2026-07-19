"""Bounded workspace checkpoints with conflict-aware rollback."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import config_dir

DEFAULT_CHECKPOINT_MAX_FILES = 20_000
DEFAULT_CHECKPOINT_MAX_MIB = 250
HARD_CHECKPOINT_MAX_FILES = 1_000_000
HARD_CHECKPOINT_MAX_MIB = 10_240
CHECKPOINT_MAX_FILES_ENV = "GLM_ACP_CHECKPOINT_MAX_FILES"
CHECKPOINT_MAX_MIB_ENV = "GLM_ACP_CHECKPOINT_MAX_MIB"
CHECKPOINT_LIMITS_FILENAME = "checkpoint-limits.json"
_IGNORED = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}
_SENSITIVE_NAMES = {".env", "credentials.json", "id_rsa", "id_ed25519"}
_SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


def _sensitive(path: Path) -> bool:
    return (
        ".ssh" in path.parts
        or path.name in _SENSITIVE_NAMES
        or path.name.startswith(".env.")
        or path.suffix.lower() in _SENSITIVE_SUFFIXES
    )


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointLimits:
    max_files: int
    max_mib: int
    files_source: str = "default"
    mib_source: str = "default"

    @property
    def max_bytes(self) -> int:
        return self.max_mib * 1024 * 1024


def _limit(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise CheckpointError(f"{name} must be a whole number from 1 to {maximum}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise CheckpointError(f"{name} must be a whole number from 1 to {maximum}") from error
    if str(value).strip() != str(parsed) or not 1 <= parsed <= maximum:
        raise CheckpointError(f"{name} must be a whole number from 1 to {maximum}")
    return parsed


class CheckpointManager:
    """Store baseline bytes and the exact agent-produced hashes needed for safe rollback."""

    def __init__(self, base_dir: Path | None = None, limits_path: Path | None = None) -> None:
        self.base_dir = base_dir or config_dir() / "checkpoints"
        self.limits_path = limits_path or config_dir() / CHECKPOINT_LIMITS_FILENAME

    def limits(self) -> CheckpointLimits:
        max_files = DEFAULT_CHECKPOINT_MAX_FILES
        max_mib = DEFAULT_CHECKPOINT_MAX_MIB
        files_source = "default"
        mib_source = "default"
        try:
            payload = json.loads(self.limits_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = None
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(f"Cannot read checkpoint limits: {error}") from error
        if payload is not None:
            if not isinstance(payload, dict) or payload.get("schema") != 1:
                raise CheckpointError("Checkpoint limits file must be a schema-1 JSON object")
            max_files = _limit(
                payload.get("max_files"), "Checkpoint max files", HARD_CHECKPOINT_MAX_FILES
            )
            max_mib = _limit(payload.get("max_mib"), "Checkpoint max MiB", HARD_CHECKPOINT_MAX_MIB)
            files_source = mib_source = "profile"
        if CHECKPOINT_MAX_FILES_ENV in os.environ:
            max_files = _limit(
                os.environ[CHECKPOINT_MAX_FILES_ENV],
                CHECKPOINT_MAX_FILES_ENV,
                HARD_CHECKPOINT_MAX_FILES,
            )
            files_source = "environment"
        if CHECKPOINT_MAX_MIB_ENV in os.environ:
            max_mib = _limit(
                os.environ[CHECKPOINT_MAX_MIB_ENV],
                CHECKPOINT_MAX_MIB_ENV,
                HARD_CHECKPOINT_MAX_MIB,
            )
            mib_source = "environment"
        return CheckpointLimits(max_files, max_mib, files_source, mib_source)

    def configure_limits(self, max_files: object, max_mib: object) -> CheckpointLimits:
        files = _limit(max_files, "Checkpoint max files", HARD_CHECKPOINT_MAX_FILES)
        mib = _limit(max_mib, "Checkpoint max MiB", HARD_CHECKPOINT_MAX_MIB)
        parent = self.limits_path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            os.chmod(parent, 0o700)
        temporary = parent / f".{self.limits_path.name}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps({"schema": 1, "max_files": files, "max_mib": mib}, indent=2) + "\n",
                encoding="utf-8",
            )
            if os.name != "nt":
                os.chmod(temporary, 0o600)
            os.replace(temporary, self.limits_path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.limits()

    def reset_limits(self) -> CheckpointLimits:
        self.limits_path.unlink(missing_ok=True)
        return self.limits()

    def _workspace_dir(self, root: Path) -> Path:
        identity = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:20]
        return self.base_dir / identity

    @staticmethod
    def _files(root: Path):
        for current, dirs, files in os.walk(root):
            dirs[:] = [
                name
                for name in dirs
                if name not in _IGNORED and not name.startswith(".glm-acp-images")
            ]
            current_path = Path(current)
            for name in files:
                path = current_path / name
                if path.is_symlink() or not path.is_file():
                    continue
                yield path

    def create(self, cwd: str, label: str = "automatic") -> dict[str, object]:
        root = Path(cwd).resolve()
        limits = self.limits()
        checkpoint_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        target = self._workspace_dir(root) / checkpoint_id
        files_dir = target / "files"
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            os.chmod(self.base_dir, 0o700)
        manifest: dict[str, object] = {
            "schema": 1,
            "id": checkpoint_id,
            "cwd": str(root),
            "label": label[:200],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": {},
            "agent_hashes": {},
            "excluded_sensitive_paths": [],
        }
        total = 0
        count = 0
        try:
            for path in self._files(root):
                if _sensitive(path):
                    manifest["excluded_sensitive_paths"].append(  # type: ignore[union-attr]
                        path.relative_to(root).as_posix()
                    )
                    continue
                data = path.read_bytes()
                count += 1
                total += len(data)
                if count > limits.max_files or total > limits.max_bytes:
                    raise CheckpointError(
                        f"Workspace checkpoint exceeds {limits.max_files} files or "
                        f"{limits.max_mib} MiB. Change it with "
                        f"`/checkpoint limits <files> <MiB>` or the "
                        f"{CHECKPOINT_MAX_FILES_ENV}/{CHECKPOINT_MAX_MIB_ENV} environment variables"
                    )
                relative = path.relative_to(root)
                saved = files_dir / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(data)
                if os.name != "nt":
                    os.chmod(saved, 0o600)
                manifest["files"][relative.as_posix()] = _hash(data)  # type: ignore[index]
            target.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                os.chmod(target, 0o700)
            (target / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if os.name != "nt":
                os.chmod(target / "manifest.json", 0o600)
            return {
                "id": checkpoint_id,
                "files": count,
                "bytes": total,
                "label": label[:200],
                "excluded_sensitive_paths": len(manifest["excluded_sensitive_paths"]),  # type: ignore[arg-type]
            }
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise

    def _load(self, cwd: str, checkpoint_id: str) -> tuple[Path, dict[str, object]]:
        if not checkpoint_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for ch in checkpoint_id
        ):
            raise CheckpointError("Invalid checkpoint id")
        root = Path(cwd).resolve()
        target = self._workspace_dir(root) / checkpoint_id
        try:
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(f"Checkpoint not found: {checkpoint_id}") from error
        if not isinstance(manifest, dict) or manifest.get("cwd") != str(root):
            raise CheckpointError("Checkpoint workspace does not match the current workspace")
        return target, manifest

    def note_changes(self, cwd: str, checkpoint_id: str, paths: list[str]) -> None:
        target, manifest = self._load(cwd, checkpoint_id)
        root = Path(cwd).resolve()
        hashes = manifest.get("agent_hashes")
        if not isinstance(hashes, dict):
            hashes = {}
            manifest["agent_hashes"] = hashes
        for raw in paths[:100]:
            path = Path(raw).resolve()
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as error:
                raise CheckpointError(
                    f"Changed path is outside checkpoint workspace: {path}"
                ) from error
            hashes[relative] = _hash(path.read_bytes()) if path.is_file() else None
        temporary = target / "manifest.tmp"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target / "manifest.json")

    def note_workspace_changes(self, cwd: str, checkpoint_id: str) -> list[str]:
        """Record every current path that differs from the checkpoint baseline."""
        limits = self.limits()
        _, manifest = self._load(cwd, checkpoint_id)
        root = Path(cwd).resolve()
        baseline = manifest.get("files", {})
        if not isinstance(baseline, dict):
            raise CheckpointError("Checkpoint manifest is malformed")
        current: dict[str, str] = {}
        count = 0
        total = 0
        for path in self._files(root):
            if _sensitive(path):
                continue
            data = path.read_bytes()
            count += 1
            total += len(data)
            if count > limits.max_files or total > limits.max_bytes:
                raise CheckpointError(
                    "Workspace grew beyond checkpoint tracking limits "
                    f"({limits.max_files} files/{limits.max_mib} MiB)"
                )
            current[path.relative_to(root).as_posix()] = _hash(data)
        changed = sorted(
            relative
            for relative in set(baseline) | set(current)
            if baseline.get(relative) != current.get(relative)
        )
        self.note_changes(cwd, checkpoint_id, [str(root / relative) for relative in changed])
        return changed

    def list(self, cwd: str) -> list[dict[str, object]]:
        root_dir = self._workspace_dir(Path(cwd).resolve())
        results = []
        if not root_dir.exists():
            return results
        for path in sorted(root_dir.iterdir(), reverse=True):
            try:
                payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results.append({key: payload.get(key) for key in ("id", "label", "created_at")})
        return results[:20]

    def rollback(self, cwd: str, checkpoint_id: str) -> dict[str, object]:
        target, manifest = self._load(cwd, checkpoint_id)
        root = Path(cwd).resolve()
        baseline = manifest.get("files", {})
        agent_hashes = manifest.get("agent_hashes", {})
        if not isinstance(baseline, dict) or not isinstance(agent_hashes, dict):
            raise CheckpointError("Checkpoint manifest is malformed")
        conflicts: list[str] = []
        for relative, expected in agent_hashes.items():
            path = root / str(relative)
            try:
                path.resolve().relative_to(root)
            except ValueError:
                conflicts.append(str(relative))
                continue
            if path.is_symlink():
                conflicts.append(str(relative))
                continue
            current = _hash(path.read_bytes()) if path.is_file() else None
            if current != expected:
                conflicts.append(str(relative))
        if conflicts:
            return {"rolled_back": False, "conflicts": conflicts[:100], "restored": []}
        operations: list[tuple[Path, str, bytes | None, bytes | None]] = []
        for relative in agent_hashes:
            path = root / str(relative)
            current = path.read_bytes() if path.is_file() else None
            baseline_data: bytes | None = None
            if relative in baseline:
                saved = target / "files" / str(relative)
                baseline_data = saved.read_bytes()
                if _hash(baseline_data) != baseline[relative]:
                    raise CheckpointError(f"Checkpoint content integrity failure: {relative}")
            operations.append((path, str(relative), current, baseline_data))
        restored: list[str] = []
        committed: list[tuple[Path, bytes | None]] = []
        try:
            for path, relative, current, baseline_data in operations:
                if baseline_data is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(baseline_data)
                elif path.exists():
                    if path.is_dir() or path.is_symlink():
                        raise OSError(f"Refusing to delete non-file rollback target: {relative}")
                    path.unlink()
                committed.append((path, current))
                restored.append(relative)
        except OSError as error:
            recovery_errors: list[str] = []
            for path, previous in reversed(committed):
                try:
                    if previous is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(previous)
                except OSError:
                    recovery_errors.append(str(path))
            detail = (
                f"; recovery failed for {', '.join(recovery_errors)}" if recovery_errors else ""
            )
            raise CheckpointError(
                f"Rollback commit failed and was recovered{detail}: {error}"
            ) from error
        return {"rolled_back": True, "conflicts": [], "restored": restored}
