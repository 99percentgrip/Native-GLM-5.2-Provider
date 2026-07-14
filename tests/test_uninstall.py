"""Tests for safe public-install removal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glm_acp.config import CONFIG_DIR_ENV, credentials_path, store_api_key
from glm_acp.uninstall import (
    UninstallError,
    _default_zed_settings,
    _without_windows_path,
    remove_zed_custom_agent,
    uninstall_release,
)


def _public_install(tmp_path: Path) -> tuple[Path, Path]:
    install_dir = tmp_path / ".local" / "bin"
    install_dir.mkdir(parents=True)
    native = install_dir / "native-glm-acp"
    native.write_text("binary", encoding="utf-8")
    alias = install_dir / "glm-acp"
    alias.write_text("binary", encoding="utf-8")
    return install_dir, native


def _zed_settings(home: Path, command: Path, agent_type: str = "custom") -> Path:
    settings = home / ".config" / "zed" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    command_json = json.dumps(str(command))
    settings.write_text(
        f"""{{
  // Preserve unrelated JSONC settings.
  "theme": "One Dark",
  "agent_servers": {{
    "glm-acp": {{
      "default_config_options": {{"model": "glm-5.2",}},
      "type": "{agent_type}",
      "command": {command_json},
      "args": [],
    }},
    "codex-acp": {{"type": "registry",}},
  }},
}}
""",
        encoding="utf-8",
    )
    return settings


def test_uninstall_removes_public_commands_path_and_custom_zed_entry(tmp_path):
    home = tmp_path / "home"
    install_dir, native = _public_install(home)
    settings = _zed_settings(home, install_dir / "glm-acp")
    original_settings = settings.read_text(encoding="utf-8")
    profile = home / ".profile"
    profile.write_text(
        f'keep-this\n\n# Native GLM ACP\nexport PATH="{install_dir}:$PATH"\n',
        encoding="utf-8",
    )

    result = uninstall_release(
        executable=native,
        install_dir=install_dir,
        frozen=True,
        platform_name="unix",
        home=home,
        zed_settings=settings,
    )

    assert not native.exists()
    assert not (install_dir / "glm-acp").exists()
    assert result.profile_updated is True
    assert profile.read_text(encoding="utf-8") == "keep-this\n\n"
    updated_settings = settings.read_text(encoding="utf-8")
    assert '"glm-acp"' not in updated_settings
    assert '"codex-acp"' in updated_settings
    assert result.zed_backup is not None
    assert result.zed_backup.read_text(encoding="utf-8") == original_settings


def test_uninstall_preserves_credentials_by_default(monkeypatch, tmp_path):
    home = tmp_path / "home"
    install_dir, native = _public_install(home)
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "credentials"))
    store_api_key("secret-value")

    result = uninstall_release(
        executable=native,
        install_dir=install_dir,
        frozen=True,
        platform_name="unix",
        home=home,
        zed_settings=tmp_path / "missing-settings.json",
    )

    assert result.credentials_removed is False
    assert credentials_path().exists()


def test_uninstall_purge_removes_only_stored_credential(monkeypatch, tmp_path):
    home = tmp_path / "home"
    install_dir, native = _public_install(home)
    config = tmp_path / "credentials"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config))
    store_api_key("secret-value")
    retained = config / "sessions"
    retained.mkdir()

    result = uninstall_release(
        purge=True,
        executable=native,
        install_dir=install_dir,
        frozen=True,
        platform_name="unix",
        home=home,
        zed_settings=tmp_path / "missing-settings.json",
    )

    assert result.credentials_removed is True
    assert not credentials_path().exists()
    assert retained.exists()


def test_source_and_registry_managed_installs_refuse_self_deletion(tmp_path):
    install_dir, native = _public_install(tmp_path)
    with pytest.raises(UninstallError, match="source installation"):
        uninstall_release(
            executable=native,
            install_dir=install_dir,
            frozen=False,
            platform_name="unix",
        )

    other = tmp_path / "registry" / "native-glm-acp"
    other.parent.mkdir()
    other.write_text("binary", encoding="utf-8")
    with pytest.raises(UninstallError, match="not a public installer-owned copy"):
        uninstall_release(
            executable=other,
            install_dir=install_dir,
            frozen=True,
            platform_name="unix",
        )
    assert other.exists()


def test_zed_registry_or_unrelated_custom_entry_is_preserved(tmp_path):
    install_dir, _ = _public_install(tmp_path)
    registry = _zed_settings(tmp_path, install_dir / "glm-acp", agent_type="registry")
    assert remove_zed_custom_agent(registry, install_dir) is None
    assert '"glm-acp"' in registry.read_text(encoding="utf-8")

    unrelated = _zed_settings(tmp_path, (tmp_path / "other" / "bin" / "glm-acp").resolve())
    assert remove_zed_custom_agent(unrelated, install_dir) is None
    assert '"glm-acp"' in unrelated.read_text(encoding="utf-8")


def test_windows_path_removal_preserves_other_entries(tmp_path):
    install_dir = tmp_path / "NativeGLMAcp"
    value = f"C:\\Windows;{install_dir};C:\\Tools"
    assert _without_windows_path(value, install_dir) == "C:\\Windows;C:\\Tools"


def test_zed_settings_honors_xdg_config_home(monkeypatch, tmp_path):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert _default_zed_settings(tmp_path / "home", "unix") == xdg / "zed" / "settings.json"
