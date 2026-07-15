"""Offline checks for runtime-free public installers."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fake_unix_release(tmp_path: Path, checksum: str | None = None) -> Path:
    download = tmp_path / "releases" / "latest" / "download"
    download.mkdir(parents=True)
    executable = tmp_path / "native-glm-acp"
    executable.write_text("#!/bin/sh\nprintf '0.9.0\\n'\n", encoding="utf-8")
    executable.chmod(0o755)
    system = platform.system().lower()
    machine = platform.machine().lower()
    architecture = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
    asset = download / f"native-glm-acp-{system}-{architecture}.tar.gz"
    with tarfile.open(asset, "w:gz") as archive:
        archive.add(executable, arcname="native-glm-acp")
    digest = checksum or hashlib.sha256(asset.read_bytes()).hexdigest()
    asset.with_suffix(asset.suffix + ".sha256").write_text(
        f"{digest}  {asset.name}\n", encoding="utf-8"
    )
    return download


@pytest.mark.skipif(os.name == "nt", reason="Unix installer integration test")
def test_unix_installer_verifies_and_installs_both_commands(tmp_path):
    _fake_unix_release(tmp_path)
    install_dir = tmp_path / "bin"
    env = os.environ.copy()
    env.update(
        {
            "GLM_ACP_RELEASE_BASE_URL": (tmp_path / "releases").as_uri(),
            "GLM_ACP_INSTALL_DIR": str(install_dir),
            "PATH": f"{install_dir}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        ["sh", str(ROOT / "scripts" / "install.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (install_dir / "native-glm-acp").stat().st_mode & 0o111
    assert (install_dir / "glm-acp").is_symlink()
    assert os.readlink(install_dir / "glm-acp") == "native-glm-acp"
    assert subprocess.check_output([install_dir / "glm-acp", "--version"], text=True).strip() == (
        "0.9.0"
    )
    assert "Next: glm-acp --setup" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="Unix installer integration test")
def test_unix_installer_persists_user_path_once(tmp_path):
    _fake_unix_release(tmp_path)
    install_dir = tmp_path / "bin"
    profile = tmp_path / ".profile"
    env = os.environ.copy()
    env.update(
        {
            "GLM_ACP_RELEASE_BASE_URL": (tmp_path / "releases").as_uri(),
            "GLM_ACP_INSTALL_DIR": str(install_dir),
            "GLM_ACP_SHELL_PROFILE": str(profile),
        }
    )

    for _ in range(2):
        result = subprocess.run(
            ["sh", str(ROOT / "scripts" / "install.sh")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    path_line = f'export PATH="{install_dir}:$PATH"'
    assert profile.read_text(encoding="utf-8").count(path_line) == 1


@pytest.mark.skipif(os.name == "nt", reason="Unix installer integration test")
def test_unix_installer_rejects_bad_checksum(tmp_path):
    _fake_unix_release(tmp_path, checksum="0" * 64)
    install_dir = tmp_path / "bin"
    env = os.environ.copy()
    env.update(
        {
            "GLM_ACP_RELEASE_BASE_URL": (tmp_path / "releases").as_uri(),
            "GLM_ACP_INSTALL_DIR": str(install_dir),
        }
    )

    result = subprocess.run(
        ["sh", str(ROOT / "scripts" / "install.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not (install_dir / "native-glm-acp").exists()


def test_installer_release_contract():
    unix = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    windows = (ROOT / "scripts" / "install.ps1").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "sha256sum -c" in unix
    assert "ln -sf native-glm-acp" in unix
    assert "Get-FileHash -Algorithm SHA256" in windows
    assert '"native-glm-acp.exe"' in windows
    assert '"glm-acp.exe"' in windows
    assert "scripts/install.sh" in workflow
    assert "scripts/install.ps1" in workflow


def test_unix_installer_has_valid_shell_syntax():
    shell = shutil.which("bash")
    if shell is None:
        pytest.skip("bash is unavailable")
    result = subprocess.run(
        [shell, "-n", str(ROOT / "scripts" / "install.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
