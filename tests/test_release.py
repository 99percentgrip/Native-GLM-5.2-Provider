"""Tests for glm_acp.release — version detection, bumping, and file updates."""

from __future__ import annotations

from pathlib import Path

import pytest

from glm_acp.release import (
    bump_version,
    detect_version,
    find_version_files,
    update_version_in_files,
)


def test_bump_patch():
    assert bump_version("1.9.2", "patch") == "1.9.3"


def test_bump_minor():
    assert bump_version("1.9.2", "minor") == "1.10.0"


def test_bump_major():
    assert bump_version("1.9.2", "major") == "2.0.0"


def test_bump_zero_versions():
    assert bump_version("0.0.0", "patch") == "0.0.1"
    assert bump_version("0.1.0", "minor") == "0.2.0"
    assert bump_version("1.0.0", "major") == "2.0.0"


def test_bump_invalid_version():
    with pytest.raises(ValueError, match="Cannot parse"):
        bump_version("not-a-version", "patch")


def test_bump_invalid_type():
    with pytest.raises(ValueError, match="Unknown release type"):
        bump_version("1.0.0", "bogus")


def test_detect_version_from_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test"\nversion = "3.7.1"\n', encoding="utf-8"
    )
    assert detect_version(str(tmp_path)) == "3.7.1"


def test_detect_version_from_init(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "2.4.0"\n', encoding="utf-8")
    assert detect_version(str(tmp_path)) == "2.4.0"


def test_detect_version_returns_none(tmp_path: Path):
    (tmp_path / "README.md").write_text("No version here", encoding="utf-8")
    assert detect_version(str(tmp_path)) is None


def test_find_version_files(tmp_path: Path):
    (tmp_path / "app.py").write_text('VERSION = "1.5.0"\n', encoding="utf-8")
    (tmp_path / "docs.md").write_text("Version 1.5.0 released\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("unrelated content\n", encoding="utf-8")
    matches = find_version_files(str(tmp_path), "1.5.0")
    names = {p.name for p in matches}
    assert "app.py" in names
    assert "docs.md" in names
    assert "other.txt" not in names


def test_find_version_files_skips_venv(tmp_path: Path):
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "site.py").write_text('version = "1.5.0"\n', encoding="utf-8")
    matches = find_version_files(str(tmp_path), "1.5.0")
    assert all(".venv" not in p.parts for p in matches)


def test_update_version_in_files(tmp_path: Path):
    (tmp_path / "app.py").write_text('VERSION = "1.5.0"\n', encoding="utf-8")
    (tmp_path / "docs.md").write_text("# Version 1.5.0\n", encoding="utf-8")
    updated = update_version_in_files(str(tmp_path), "1.5.0", "1.6.0")
    assert len(updated) == 2
    assert "1.6.0" in (tmp_path / "app.py").read_text()
    assert "1.6.0" in (tmp_path / "docs.md").read_text()
    assert "1.5.0" not in (tmp_path / "app.py").read_text()


@pytest.mark.asyncio
async def test_version_info(tmp_path: Path):
    from glm_acp.release import version_info

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "4.2.0"\n', encoding="utf-8"
    )
    result = version_info(str(tmp_path))
    assert "4.2.0" in result
