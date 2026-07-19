"""Tests for checkpoints, references, policy, sandbox, workflows, profiles, and plugins."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from glm_acp.agent import GlmAcpAgent, Session
from glm_acp.checkpoints import CheckpointError, CheckpointManager
from glm_acp.config import config_dir
from glm_acp.os_sandbox import command_prefix
from glm_acp.plugins import PluginError, PluginRegistry
from glm_acp.policy import PolicyEngine
from glm_acp.profiles import active_profile
from glm_acp.references import expand_references
from glm_acp.tools import Sandbox, ToolError, execute_tool
from glm_acp.workflows import ordered_steps
from glm_acp.worktrees import WorktreeError, WorktreeManager


def test_checkpoint_rolls_back_only_agent_recorded_hashes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tracked = workspace / "tracked.txt"
    tracked.write_text("before")
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.create(str(workspace), "test")

    tracked.write_text("after")
    created = workspace / "created.txt"
    created.write_text("new")
    manager.note_workspace_changes(str(workspace), str(checkpoint["id"]))

    result = manager.rollback(str(workspace), str(checkpoint["id"]))
    assert result["rolled_back"] is True
    assert tracked.read_text() == "before"
    assert not created.exists()


def test_checkpoint_stops_on_conflicting_later_edit(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "file.txt"
    path.write_text("baseline")
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.create(str(workspace))
    path.write_text("agent")
    manager.note_workspace_changes(str(workspace), str(checkpoint["id"]))
    path.write_text("user-later")

    result = manager.rollback(str(workspace), str(checkpoint["id"]))
    assert result == {"rolled_back": False, "conflicts": ["file.txt"], "restored": []}
    assert path.read_text() == "user-later"


def test_checkpoint_never_copies_sensitive_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("SECRET=value")
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.create(str(workspace))
    stored = list((tmp_path / "checkpoints").rglob("*"))
    assert checkpoint["excluded_sensitive_paths"] == 1
    assert not any(path.name == ".env" for path in stored)
    assert "SECRET=value" not in "".join(
        path.read_text(errors="ignore") for path in stored if path.is_file()
    )


def test_checkpoint_limits_are_persistent_bounded_and_overridable(tmp_path, monkeypatch):
    limits_path = tmp_path / "profile" / "checkpoint-limits.json"
    manager = CheckpointManager(tmp_path / "checkpoints", limits_path)
    assert (manager.limits().max_files, manager.limits().max_mib) == (20_000, 250)

    configured = manager.configure_limits(1, 1)
    assert configured.files_source == configured.mib_source == "profile"
    restored = CheckpointManager(tmp_path / "other-checkpoints", limits_path).limits()
    assert (restored.max_files, restored.max_mib) == (1, 1)

    workspace = tmp_path / "workspace-limits"
    workspace.mkdir()
    (workspace / "one.txt").write_text("one")
    (workspace / "two.txt").write_text("two")
    with pytest.raises(CheckpointError, match="/checkpoint limits"):
        manager.create(str(workspace))

    monkeypatch.setenv("GLM_ACP_CHECKPOINT_MAX_FILES", "2")
    monkeypatch.setenv("GLM_ACP_CHECKPOINT_MAX_MIB", "2")
    effective = manager.limits()
    assert (effective.max_files, effective.max_mib) == (2, 2)
    assert effective.files_source == effective.mib_source == "environment"
    assert manager.create(str(workspace))["files"] == 2

    with pytest.raises(CheckpointError, match="1 to 1000000"):
        manager.configure_limits(1_000_001, 1)
    monkeypatch.setenv("GLM_ACP_CHECKPOINT_MAX_FILES", "invalid")
    with pytest.raises(CheckpointError, match="GLM_ACP_CHECKPOINT_MAX_FILES"):
        manager.limits()


@pytest.mark.asyncio
async def test_checkpoint_limits_command_is_easy_to_change_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_ACP_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("GLM_ACP_CHECKPOINT_MAX_FILES", raising=False)
    monkeypatch.delenv("GLM_ACP_CHECKPOINT_MAX_MIB", raising=False)
    agent = GlmAcpAgent()
    agent._save_session = AsyncMock()
    limits_path = tmp_path / "profile" / "checkpoint-limits.json"
    agent._checkpoints = CheckpointManager(tmp_path / "checkpoints", limits_path)
    session = Session("checkpoint-limits", str(tmp_path / "workspace"))

    saved = await agent._handle_command(session, "/checkpoint limits 100000 1024")
    assert "100,000 files" in saved
    assert "1,024 MiB" in saved
    shown = await agent._handle_command(session, "/checkpoint limits")
    assert "100,000" in shown and "profile" in shown
    reset = await agent._handle_command(session, "/checkpoint limits reset")
    assert "20,000 files / 250 MiB" in reset
    assert not limits_path.exists()
    await agent.aclose()


def test_explicit_references_are_bounded_and_untrusted(tmp_path):
    (tmp_path / "a.py").write_text("def useful_symbol():\n    return 1\n")
    expanded, targets = expand_references(
        "Review @file:a.py @folder:. @symbol:useful_symbol @diff", Sandbox(str(tmp_path))
    )
    assert "<untrusted_context" in expanded
    assert "def useful_symbol" in expanded
    assert str(tmp_path / "a.py") in targets

    with pytest.raises(ToolError, match="outside the workspace"):
        expand_references("Read @file:../secret", Sandbox(str(tmp_path)))
    (tmp_path / ".env").write_text("SECRET=value")
    with pytest.raises(ToolError, match="Sensitive file"):
        expand_references("Read @file:.env", Sandbox(str(tmp_path)))


def test_policy_is_ordered_and_invalid_policy_fails_closed(tmp_path):
    policy_dir = tmp_path / ".glm-acp"
    policy_dir.mkdir()
    path = policy_dir / "policy.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "effect": "deny",
                        "tools": ["run_command"],
                        "command_regex": "rm\\s",
                        "reason": "no removal",
                    },
                    {"effect": "allow", "tools": ["run_command"]},
                ],
            }
        )
    )
    engine = PolicyEngine(str(tmp_path))
    assert engine.evaluate("run_command", {"command": "rm file"}, [])[0] == "deny"
    assert engine.evaluate("run_command", {"command": "pytest"}, [])[0] == "allow"

    path.write_text("{")
    effect, reason = engine.evaluate("read_file", {"path": "x"}, ["x"])
    assert effect == "deny"
    assert "Invalid policy" in reason


@pytest.mark.asyncio
async def test_policy_cannot_override_read_only_and_checks_workflow_steps(tmp_path, monkeypatch):
    policy_dir = tmp_path / ".glm-acp"
    policy_dir.mkdir()
    path = policy_dir / "policy.json"
    agent = GlmAcpAgent()
    session = Session("session", str(tmp_path))

    async def send_message(*args, **kwargs):
        return None

    monkeypatch.setattr(agent, "_send_message", send_message)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [{"effect": "allow", "tools": ["write_file"]}],
            }
        )
    )
    session.permission_mode = "read"
    permitted, _ = await agent._check_permission(
        session, "call", "write_file", {"path": "x", "content": "x"}
    )
    assert permitted is False

    class PermissionClient:
        called = False

        async def request_permission(self, **kwargs):
            self.called = True
            return SimpleNamespace(outcome=SimpleNamespace(outcome="selected", option_id="allow"))

    permission_client = PermissionClient()
    agent._conn = permission_client
    session.permission_mode = "ask"
    permitted, _ = await agent._check_permission(
        session, "call", "write_file", {"path": "x", "content": "x"}
    )
    assert permitted is True
    assert permission_client.called is True

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "effect": "deny",
                        "tools": ["run_command"],
                        "command_regex": "deploy",
                    },
                    {"effect": "allow", "tools": ["run_workflow"]},
                ],
            }
        )
    )
    session.permission_mode = "bypass"
    permitted, reason = await agent._check_permission(
        session,
        "call",
        "run_workflow",
        {
            "steps": [
                {
                    "id": "deploy",
                    "tool": "run_command",
                    "arguments": {"command": "deploy production"},
                }
            ]
        },
    )
    assert permitted is False
    assert "Workflow step deploy" in reason


def test_profile_paths_preserve_default_and_isolate_named_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_ACP_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("GLM_ACP_PROFILE", raising=False)
    assert active_profile() == "default"
    assert config_dir() == tmp_path / "config"

    monkeypatch.setenv("GLM_ACP_PROFILE", "client_a")
    assert config_dir() == tmp_path / "config" / "profiles" / "client_a"
    monkeypatch.setenv("GLM_ACP_PROFILE", "../escape")
    with pytest.raises(ValueError):
        active_profile()


def test_os_sandbox_required_uses_bubblewrap_or_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_ACP_OS_SANDBOX", "required")
    try:
        prefix, backend = command_prefix([tmp_path], network_override=False)
    except RuntimeError as error:
        assert "required" in str(error)
        return
    if backend == "bubblewrap":
        assert "--ro-bind" in prefix
        assert "--unshare-net" in prefix
        bind_index = prefix.index("--bind")
        assert prefix[bind_index : bind_index + 3] == ["--bind", str(tmp_path), str(tmp_path)]
    else:  # pragma: no cover - future supported backend
        assert backend not in {"disabled", "workspace-only"}


def test_declarative_workflow_orders_dependencies_and_stops_on_failure(tmp_path):
    steps = ordered_steps(
        [
            {
                "id": "verify",
                "tool": "run_command",
                "arguments": {"command": "false"},
                "needs": ["write"],
            },
            {"id": "write", "tool": "write_file", "arguments": {"path": "x.txt", "content": "x"}},
            {
                "id": "never",
                "tool": "write_file",
                "arguments": {"path": "never", "content": "x"},
                "needs": ["verify"],
            },
        ]
    )
    assert [step["id"] for step in steps] == ["write", "verify", "never"]


@pytest.mark.asyncio
async def test_declarative_workflow_executes_static_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_ACP_OS_SANDBOX", "off")
    result = await execute_tool(
        "run_workflow",
        {
            "steps": [
                {
                    "id": "write",
                    "tool": "write_file",
                    "arguments": {"path": "x.txt", "content": "x"},
                },
                {
                    "id": "fail",
                    "tool": "run_command",
                    "arguments": {"command": "exit 4"},
                    "needs": ["write"],
                },
                {
                    "id": "skip",
                    "tool": "write_file",
                    "arguments": {"path": "skip", "content": "x"},
                    "needs": ["fail"],
                },
            ]
        },
        Sandbox(str(tmp_path)),
    )
    assert result.exit_code == 4
    assert (tmp_path / "x.txt").read_text() == "x"
    assert not (tmp_path / "skip").exists()


def test_plugin_package_is_hash_pinned_and_data_only(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    prompt = source / "guidance.md"
    prompt.write_text("Use the repository verifier.")
    digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
    manifest = source / "plugin.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "id": "quality_rules",
                "version": "1.0.0",
                "permissions": ["prompt_context"],
                "files": {"guidance.md": digest},
                "prompt_files": ["guidance.md"],
            }
        )
    )
    registry = PluginRegistry(tmp_path / "installed")
    assert registry.install(manifest)["files"] == 1
    assert registry.verify("quality_rules")["verified"] is True
    assert "repository verifier" in registry.prompt_fragments()

    installed_manifest = tmp_path / "installed" / "quality_rules" / "plugin.json"
    original_manifest = installed_manifest.read_text()
    installed_manifest.write_text(original_manifest.replace("1.0.0", "1.0.1"))
    with pytest.raises(PluginError, match="manifest hash mismatch"):
        registry.verify("quality_rules")
    installed_manifest.write_text(original_manifest)

    (tmp_path / "installed" / "quality_rules" / "guidance.md").write_text("tampered")
    with pytest.raises(PluginError, match="hash mismatch"):
        registry.verify("quality_rules")

    executable = source / "plugin.py"
    executable.write_text("print('unsafe')")
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "id": "unsafe_code",
                "permissions": [],
                "files": {"plugin.py": hashlib.sha256(executable.read_bytes()).hexdigest()},
            }
        )
    )
    with pytest.raises(PluginError, match="Executable plugin content"):
        registry.install(manifest)


def test_worktree_manager_preserves_dirty_worker_and_removes_clean_one(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "file.txt").write_text("base")
    subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    manager = WorktreeManager(tmp_path / "workers")
    state = manager.create(str(repo))
    worker_file = Path(state["path"]) / "file.txt"
    worker_file.write_text("changed")
    assert "changed" in manager.diff(state["path"], state["base_ref"])
    with pytest.raises(WorktreeError, match="dirty"):
        manager.remove_if_clean(state["repo"], state["path"])

    worker_file.write_text("base")
    manager.remove_if_clean(state["repo"], state["path"])
    assert not Path(state["path"]).exists()
