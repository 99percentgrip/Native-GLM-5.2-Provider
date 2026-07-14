"""Offline checks for the ACP Registry submission assets."""

from __future__ import annotations

import json
from pathlib import Path

from glm_acp import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_registry_manifest_matches_release_contract():
    manifest = json.loads((ROOT / "registry" / "agent.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "native-glm-acp"
    assert manifest["version"] == __version__
    assert manifest["authors"] == ["Aleksejs Kozlitins"]
    assert manifest["license"] == "Apache-2.0"
    assert manifest["repository"] == ("https://github.com/99percentgrip/Native-GLM-5.2-Provider")
    binaries = manifest["distribution"]["binary"]
    assert set(binaries) == {
        "darwin-x86_64",
        "linux-x86_64",
        "windows-x86_64",
    }
    for target, distribution in binaries.items():
        assert f"/v{__version__}/" in distribution["archive"]
        assert target in distribution["archive"]
        assert distribution["cmd"].startswith("./native-glm-acp")


def test_registry_icon_is_small_monochrome_svg():
    icon = (ROOT / "registry" / "icon.svg").read_text(encoding="utf-8")
    assert 'viewBox="0 0 16 16"' in icon
    assert "currentColor" in icon
    assert len(icon.encode()) < 4096
