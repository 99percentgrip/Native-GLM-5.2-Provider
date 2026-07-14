"""Safe self-uninstallation for public frozen-binary installs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .config import config_dir, credentials_path

COMMAND_NAMES = ("glm-acp", "native-glm-acp")
WINDOWS_COMMAND_NAMES = ("glm-acp.exe", "native-glm-acp.exe")
PROFILE_MARKER = "# Native GLM ACP"


class UninstallError(RuntimeError):
    """Raised when a safe public-install uninstall cannot be performed."""


@dataclass(frozen=True)
class UninstallResult:
    commands: tuple[Path, ...]
    scheduled: bool
    profile_updated: bool
    zed_settings: Path | None
    zed_backup: Path | None
    credentials_removed: bool


class _Token(NamedTuple):
    kind: str
    value: str
    start: int
    end: int


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def _default_install_dir(platform_name: str) -> Path:
    override = os.environ.get("GLM_ACP_INSTALL_DIR")
    if override:
        return Path(override).expanduser()
    if platform_name == "windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "Programs" / "NativeGLMAcp"
        return Path.home() / "AppData" / "Local" / "Programs" / "NativeGLMAcp"
    return Path(os.environ.get("XDG_BIN_HOME", Path.home() / ".local" / "bin"))


def _tokenize_jsonc(text: str) -> list[_Token]:
    """Return JSON string and punctuation tokens while ignoring JSONC comments."""
    tokens: list[_Token] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            closing = text.find("*/", index + 2)
            if closing < 0:
                raise UninstallError("Zed settings contain an unterminated comment")
            index = closing + 2
            continue
        if char == '"':
            start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    raw = text[start:index]
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError as error:
                        raise UninstallError("Zed settings contain an invalid string") from error
                    tokens.append(_Token("string", value, start, index))
                    break
                else:
                    index += 1
            else:
                raise UninstallError("Zed settings contain an unterminated string")
            continue
        if char in "{}[]:,":
            tokens.append(_Token("punct", char, index, index + 1))
        index += 1
    return tokens


def _matching_close(tokens: list[_Token], opening: int) -> int:
    pairs = {"{": "}", "[": "]"}
    expected = pairs.get(tokens[opening].value)
    if expected is None:
        raise UninstallError("Expected an object or array in Zed settings")
    depth = 0
    for index in range(opening + 1, len(tokens)):
        value = tokens[index].value
        if value == tokens[opening].value:
            depth += 1
        elif value == expected:
            if depth == 0:
                return index
            depth -= 1
    raise UninstallError("Zed settings contain an unterminated object")


def _object_member(tokens: list[_Token], opening: int, name: str) -> tuple[int, int] | None:
    closing = _matching_close(tokens, opening)
    depth = 0
    index = opening + 1
    while index < closing:
        token = tokens[index]
        if token.value in {"{", "["}:
            depth += 1
        elif token.value in {"}", "]"}:
            depth -= 1
        elif (
            depth == 0
            and token.kind == "string"
            and token.value == name
            and index + 2 < closing
            and tokens[index + 1].value == ":"
        ):
            return index, index + 2
        index += 1
    return None


def _member_string(tokens: list[_Token], opening: int, name: str) -> str | None:
    member = _object_member(tokens, opening, name)
    if member is None:
        return None
    value = tokens[member[1]]
    return value.value if value.kind == "string" else None


def _matching_custom_command(command: str | None, install_dir: Path) -> bool:
    if not command:
        return False
    command_path = Path(command).expanduser()
    if command_path.name not in {*COMMAND_NAMES, *WINDOWS_COMMAND_NAMES}:
        return False
    if not command_path.is_absolute():
        return True
    return command_path.parent.resolve() == install_dir.resolve()


def _member_removal_span(
    text: str, tokens: list[_Token], key: int, closing: int
) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, tokens[key].start) + 1
    start = line_start if not text[line_start : tokens[key].start].strip() else tokens[key].start
    end = tokens[closing].end
    next_token = closing + 1
    if next_token < len(tokens) and tokens[next_token].value == ",":
        end = tokens[next_token].end
        newline = text.find("\n", end)
        if newline >= 0 and not text[end:newline].strip():
            end = newline + 1
    else:
        previous = key - 1
        if previous >= 0 and tokens[previous].value == ",":
            start = tokens[previous].start
    return start, end


def _next_backup_path(settings: Path) -> Path:
    base = settings.with_name(f"{settings.name}.backup-before-glm-acp-uninstall")
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = base.with_name(f"{base.name}.{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _atomic_replace(path: Path, content: str) -> None:
    mode = path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_zed_custom_agent(settings: Path, install_dir: Path) -> Path | None:
    """Remove only a matching custom glm-acp entry, preserving JSONC formatting."""
    try:
        text = settings.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    tokens = _tokenize_jsonc(text)
    try:
        root = next(index for index, token in enumerate(tokens) if token.value == "{")
    except StopIteration as error:
        raise UninstallError("Zed settings do not contain a root object") from error
    servers = _object_member(tokens, root, "agent_servers")
    if servers is None or tokens[servers[1]].value != "{":
        return None
    agent = _object_member(tokens, servers[1], "glm-acp")
    if agent is None or tokens[agent[1]].value != "{":
        return None
    agent_open = agent[1]
    if _member_string(tokens, agent_open, "type") != "custom" or not _matching_custom_command(
        _member_string(tokens, agent_open, "command"), install_dir
    ):
        return None
    agent_close = _matching_close(tokens, agent_open)
    start, end = _member_removal_span(text, tokens, agent[0], agent_close)
    backup = _next_backup_path(settings)
    shutil.copy2(settings, backup)
    _atomic_replace(settings, text[:start] + text[end:])
    return backup


def _default_zed_settings(home: Path, platform_name: str) -> Path:
    override = os.environ.get("GLM_ACP_ZED_SETTINGS")
    if override:
        return Path(override).expanduser()
    if platform_name == "windows" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "Zed" / "settings.json"
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "zed" / "settings.json"
    return home / ".config" / "zed" / "settings.json"


def _remove_unix_profile_path(home: Path, install_dir: Path) -> bool:
    marker = f'{PROFILE_MARKER}\nexport PATH="{install_dir}:$PATH"\n'
    changed = False
    profiles = [home / ".profile", home / ".zprofile"]
    if override := os.environ.get("GLM_ACP_SHELL_PROFILE"):
        profiles.insert(0, Path(override).expanduser())
    for profile in dict.fromkeys(profiles):
        try:
            content = profile.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        updated = content.replace(f"\n{marker}", "\n").replace(marker, "")
        if updated != content:
            _atomic_replace(profile, updated)
            changed = True
    return changed


def _without_windows_path(value: str | None, install_dir: Path) -> str:
    entries = [entry for entry in (value or "").split(";") if entry]
    target = os.path.normcase(os.path.normpath(str(install_dir)))
    return ";".join(
        entry for entry in entries if os.path.normcase(os.path.normpath(entry)) != target
    )


def _remove_windows_user_path(install_dir: Path) -> bool:
    import winreg

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Environment",
            0,
            winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE,
        )
    except FileNotFoundError:
        return False
    with key:
        try:
            current, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return False
        updated = _without_windows_path(current, install_dir)
        if updated == current:
            return False
        winreg.SetValueEx(key, "Path", 0, kind, updated)
        return True


def _schedule_windows_removal(commands: tuple[Path, ...]) -> None:
    descriptor, script_name = tempfile.mkstemp(prefix="glm-acp-uninstall-", suffix=".cmd")
    script = Path(script_name)
    lines = ["@echo off", "set tries=0", ":retry"]
    for command in commands:
        lines.append(f'del /f /q "{command}" >nul 2>&1')
    missing = " ".join(f'if not exist "{command}"' for command in commands)
    lines.extend(
        [
            f"{missing} goto done",
            "set /a tries+=1",
            "if %tries% GEQ 10 goto done",
            "timeout /t 1 /nobreak >nul",
            "goto retry",
            ":done",
            'del /f /q "%~f0" >nul 2>&1',
        ]
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\r\n") as handle:
        handle.write("\n".join(lines) + "\n")
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    subprocess.Popen(
        ["cmd.exe", "/d", "/c", str(script)],
        close_fds=True,
        creationflags=flags,
    )


def uninstall_release(
    *,
    purge: bool = False,
    executable: Path | None = None,
    install_dir: Path | None = None,
    frozen: bool | None = None,
    platform_name: str | None = None,
    home: Path | None = None,
    zed_settings: Path | None = None,
) -> UninstallResult:
    """Remove a public frozen-binary install without touching source or Registry installs."""
    platform_name = platform_name or ("windows" if os.name == "nt" else "unix")
    frozen = _is_frozen() if frozen is None else frozen
    if not frozen:
        raise UninstallError(
            "This is a source installation. Use your package manager to uninstall it "
            "(for example: uv pip uninstall glm-acp)."
        )
    executable = (executable or Path(sys.executable)).resolve()
    expected_dir = (install_dir or _default_install_dir(platform_name)).expanduser().resolve()
    if executable.parent != expected_dir or executable.name not in {
        *COMMAND_NAMES,
        *WINDOWS_COMMAND_NAMES,
    }:
        raise UninstallError(
            "This executable is not a public installer-owned copy. "
            "Registry agents must be removed from Zed."
        )

    home = home or Path.home()
    settings = zed_settings or _default_zed_settings(home, platform_name)
    zed_backup = remove_zed_custom_agent(settings, expected_dir)

    names = WINDOWS_COMMAND_NAMES if platform_name == "windows" else COMMAND_NAMES
    commands = tuple(expected_dir / name for name in names)
    if platform_name == "windows":
        try:
            profile_updated = _remove_windows_user_path(expected_dir)
            _schedule_windows_removal(commands)
        except OSError as error:
            raise UninstallError(f"Could not schedule command removal: {error}") from error
        scheduled = True
    else:
        try:
            profile_updated = _remove_unix_profile_path(home, expected_dir)
            for command in commands:
                command.unlink(missing_ok=True)
        except OSError as error:
            raise UninstallError(f"Could not remove installed commands: {error}") from error
        scheduled = False

    credentials_removed = False
    if purge:
        credential_file = credentials_path()
        try:
            if credential_file.exists():
                credential_file.unlink()
                credentials_removed = True
        except OSError as error:
            raise UninstallError(f"Could not remove stored credentials: {error}") from error
        try:
            config_dir().rmdir()
        except OSError:
            pass

    return UninstallResult(
        commands=commands,
        scheduled=scheduled,
        profile_updated=profile_updated,
        zed_settings=settings if zed_backup else None,
        zed_backup=zed_backup,
        credentials_removed=credentials_removed,
    )
