"""Tests for the installed command and terminal authentication setup."""

from __future__ import annotations

from glm_acp import __version__
from glm_acp.cli import configure_credentials, main
from glm_acp.config import CONFIG_DIR_ENV, load_stored_api_key


def test_version_has_single_source(capsys):
    try:
        main(["--version"])
    except SystemExit as error:
        assert error.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_setup_stores_prompted_key_without_printing_it(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("Z_AI_API_KEY", raising=False)

    assert configure_credentials(lambda _: "top-secret-value") == 0

    output = capsys.readouterr().out
    assert "top-secret-value" not in output
    assert "Credentials saved" in output
    assert load_stored_api_key() == "top-secret-value"


def test_check_auth_reports_only_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("ZAI_API_KEY", "top-secret-value")

    assert main(["--check-auth"]) == 0

    output = capsys.readouterr().out
    assert "configured" in output
    assert "top-secret-value" not in output
