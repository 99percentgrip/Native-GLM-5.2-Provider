"""Project-agnostic release automation for the TUI.

Provides ``/version``, ``/release``, and ``/ci`` commands that auto-detect
the project structure and execute the full release pipeline:

  1. Detect version from ``__init__.py`` or ``pyproject.toml``
  2. Bump semver (patch / minor / major)
  3. Update all files containing the old version string
  4. Verify with ruff + pytest (if available)
  5. Commit, tag, push
  6. Report CI status

Works for any Python project — not just GLM ACP.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


# ---------------------------------------------------------------------------
# Version detection and bumping
# ---------------------------------------------------------------------------


def detect_version(cwd: str = ".") -> str | None:
    """Detect the project version from common Python locations."""
    root = Path(cwd)

    # 1. Try pyproject.toml [project] version or [tool.hatch.version]
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
        if m:
            return m.group(1)

    # 2. Try scanning for __version__ in __init__.py files
    for init_file in root.rglob("__init__.py"):
        if ".venv" in init_file.parts or "site-packages" in init_file.parts:
            continue
        text = init_file.read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
        if m and _VERSION_RE.match(m.group(1)):
            return m.group(1)

    # 3. Try setup.cfg
    setup_cfg = root / "setup.cfg"
    if setup_cfg.exists():
        text = setup_cfg.read_text(encoding="utf-8")
        m = re.search(r"^version\s*=\s*(.+)$", text, re.MULTILINE)
        if m:
            return m.group(1).strip()

    return None


def bump_version(version: str, release_type: str) -> str:
    """Bump a semver string.

    >>> bump_version("1.9.2", "patch")
    '1.9.3'
    >>> bump_version("1.9.2", "minor")
    '1.10.0'
    >>> bump_version("1.9.2", "major")
    '2.0.0'
    """
    m = _VERSION_RE.match(version)
    if not m:
        raise ValueError(f"Cannot parse version: {version}")
    if release_type not in ("patch", "minor", "major"):
        raise ValueError(f"Unknown release type: {release_type}. Use: patch, minor, or major.")
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if release_type == "major":
        return f"{major + 1}.0.0"
    if release_type == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def find_version_files(cwd: str, old_version: str) -> list[Path]:
    """Find files containing the old version string (excluding venv, git, dist)."""
    root = Path(cwd)
    skip_dirs = {".venv", ".git", "dist", "build", "__pycache__", "node_modules", ".mypy_cache"}
    matches: list[Path] = []
    for path in root.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix in {".pyc", ".so", ".dylib", ".png", ".jpg", ".svg", ".ico"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if old_version in text:
            matches.append(path)
    return matches


def update_version_in_files(cwd: str, old_version: str, new_version: str) -> list[str]:
    """Replace old version with new version in all matching files. Returns file list."""
    updated: list[str] = []
    for path in find_version_files(cwd, old_version):
        try:
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace(old_version, new_version), encoding="utf-8")
            updated.append(str(path.relative_to(cwd)))
        except (OSError, UnicodeDecodeError):
            pass
    return updated


# ---------------------------------------------------------------------------
# Shell command runner
# ---------------------------------------------------------------------------


async def _run(cmd: list[str], cwd: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out_text = stdout.decode(errors="replace")
        err_text = stderr.decode(errors="replace")
        return proc.returncode or 0, out_text, err_text
    except (OSError, asyncio.TimeoutExpired) as e:
        return 1, "", str(e)


def _has_tool(tool: str) -> bool:
    try:
        subprocess.run(["which", tool], capture_output=True, timeout=3, check=False)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Public API: /version, /release, /ci
# ---------------------------------------------------------------------------


def version_info(cwd: str = ".") -> str:
    """Return current version for /version command."""
    ver = detect_version(cwd)
    if not ver:
        return (
            "Could not detect project version.\n\n"
            "Looked in: pyproject.toml, __init__.py, setup.cfg"
        )
    return f"**Current version: `{ver}`**\n\nUse `/release patch|minor|major` to cut a release."


async def cut_release(cwd: str, release_type: str = "patch") -> str:
    """Execute the full release pipeline. Returns a formatted summary."""
    if release_type not in ("patch", "minor", "major"):
        return f"Unknown release type `{release_type}`. Use: `/release patch|minor|major`"

    old_version = detect_version(cwd)
    if not old_version:
        return "Cannot detect project version. Aborting."

    try:
        new_version = bump_version(old_version, release_type)
    except ValueError as e:
        return str(e)

    lines = [f"## Release `{old_version}` → `{new_version}` ({release_type})\n"]

    # Step 1: Update version files
    updated = update_version_in_files(cwd, old_version, new_version)
    lines.append(f"**Updated {len(updated)} files:**")
    for f in updated[:15]:
        lines.append(f"- `{f}`")
    if len(updated) > 15:
        lines.append(f"- ... and {len(updated) - 15} more")
    lines.append("")

    # Step 2: Verify
    lines.append("**Verification:**")
    verify_ok = True

    if _has_tool("ruff"):
        rc, out, err = await _run(["ruff", "check", "."], cwd, timeout=60)
        if rc == 0:
            lines.append("- `ruff check .` — passed")
        else:
            lines.append("- `ruff check .` — **FAILED**")
            lines.append(f"```\n{err[:500]}\n```")
            verify_ok = False
    else:
        lines.append("- `ruff` not found — skipping")

    test_dir = Path(cwd) / "tests"
    if test_dir.exists():
        rc, out, err = await _run(
            ["python", "-m", "pytest", "tests/", "-q", "--tb=line"],
            cwd,
            timeout=180,
        )
        last_line = (out + err).strip().splitlines()[-1] if (out + err).strip() else "no output"
        if rc == 0:
            lines.append(f"- `pytest` — {last_line}")
        else:
            lines.append(f"- `pytest` — **FAILED**: {last_line}")
            verify_ok = False
    else:
        lines.append("- No `tests/` directory — skipping")

    if not verify_ok:
        lines.append("\n**Verification failed. Reverting version changes.**")
        # Revert
        await _run(["git", "checkout", "."], cwd, timeout=10)
        return "\n".join(lines)

    lines.append("")

    # Step 3: Commit, tag, push
    lines.append("**Git operations:**")
    tag = f"v{new_version}"

    rc, _, err = await _run(["git", "add", "-A"], cwd, timeout=10)
    rc, _, err = await _run(
        ["git", "commit", "-m", f"Release {tag}"], cwd, timeout=15
    )
    if rc != 0:
        lines.append(f"- `git commit` — **FAILED**: {err[:200]}")
        return "\n".join(lines)
    lines.append("- `git commit` — done")

    rc, _, err = await _run(
        ["git", "tag", "-a", tag, "-m", f"Release {tag}"], cwd, timeout=10
    )
    if rc != 0:
        lines.append(f"- `git tag {tag}` — **FAILED**: {err[:200]}")
        return "\n".join(lines)
    lines.append(f"- `git tag {tag}` — done")

    rc, _, err = await _run(["git", "push", "origin", "HEAD"], cwd, timeout=30)
    if rc != 0:
        lines.append(f"- `git push` — **FAILED**: {err[:200]}")
        return "\n".join(lines)
    lines.append("- `git push main` — done")

    rc, _, err = await _run(["git", "push", "origin", tag], cwd, timeout=30)
    if rc != 0:
        lines.append(f"- `git push {tag}` — **FAILED**: {err[:200]}")
    else:
        lines.append(f"- `git push {tag}` — done")

    lines.append(f"\n**Release `{tag}` pushed.** Use `/ci` to check build status.")

    return "\n".join(lines)


async def ci_status(cwd: str = ".") -> str:
    """Show CI run status via gh CLI."""
    if not _has_tool("gh"):
        return "`gh` CLI not found. Install from https://cli.github.com/"

    rc, out, err = await _run(
        ["gh", "run", "list", "--limit", "5"],
        cwd,
        timeout=15,
    )
    if rc != 0:
        return f"Failed to get CI status:\n```\n{err[:500]}\n```"

    lines = ["## Recent CI runs\n"]
    lines.append("```")
    lines.append(out.strip())
    lines.append("```")
    lines.append("\nUse `/ci` again to refresh.")

    return "\n".join(lines)
