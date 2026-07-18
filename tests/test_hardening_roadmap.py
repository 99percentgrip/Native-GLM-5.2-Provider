"""Cross-platform sandbox, promotion, trust, observability, and resilience tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.eval import load_cases
from glm_acp.agent import GlmAcpAgent, Session
from glm_acp.failure_corpus import FailureCorpus
from glm_acp.observability import observability_snapshot, render_observability
from glm_acp.os_sandbox import command_prefix
from glm_acp.plugins import (
    PluginError,
    PluginRegistry,
    generate_signing_key,
    read_public_key,
    sign_plugin_manifest,
)
from glm_acp.references import expand_references
from glm_acp.resilience import run_hardening_checks
from glm_acp.telemetry import TrajectoryRecorder
from glm_acp.tools import Sandbox
from glm_acp.worktrees import WorktreeError, WorktreeManager


def _repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "value.txt").write_text("baseline\n")
    subprocess.run(["git", "add", "value.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=path, check=True)
    return path


def test_platform_sandbox_capabilities_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("glm_acp.os_sandbox.sys.platform", "darwin")
    monkeypatch.setattr("glm_acp.os_sandbox.shutil.which", lambda name: "/usr/bin/sandbox-exec")
    prefix, backend = command_prefix([tmp_path], "required", False)
    assert backend == "macos-seatbelt"
    assert prefix[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "(deny default)" in prefix[2]
    assert "(deny network*)" in prefix[2]
    assert str(tmp_path) in prefix[2]

    monkeypatch.setattr("glm_acp.os_sandbox.sys.platform", "win32")
    with pytest.raises(RuntimeError, match="do not provide network isolation"):
        command_prefix([tmp_path], "required", False)
    assert command_prefix([tmp_path], "auto", True) == ([], "windows-job")


def test_worktree_promotion_is_hash_pinned_conflict_aware_and_reversible(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    manager = WorktreeManager(tmp_path / "workers")
    state = manager.create(str(repo))
    (Path(state["path"]) / "value.txt").write_text("worker\n")
    inspected = manager.inspect(str(repo), state["path"], state["base_ref"])
    assert inspected["paths"] == ["value.txt"]
    assert len(inspected["diff_sha256"]) == 64

    with pytest.raises(WorktreeError, match="changed after review"):
        manager.promote(str(repo), state["path"], state["base_ref"], "0" * 64)
    assert (repo / "value.txt").read_text() == "baseline\n"

    monkeypatch.setattr(
        manager,
        "_after_apply",
        lambda *_: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(WorktreeError, match="rolled back"):
        manager.promote(str(repo), state["path"], state["base_ref"], inspected["diff_sha256"])
    assert (repo / "value.txt").read_text() == "baseline\n"

    monkeypatch.setattr(manager, "_after_apply", lambda *_: None)
    promoted = manager.promote(
        str(repo), state["path"], state["base_ref"], inspected["diff_sha256"]
    )
    assert promoted["promoted"] is True
    assert (repo / "value.txt").read_text() == "worker\n"
    assert Path(state["path"]).exists()


def test_worktree_promotion_stops_on_primary_conflict(tmp_path):
    repo = _repo(tmp_path / "repo")
    manager = WorktreeManager(tmp_path / "workers")
    state = manager.create(str(repo))
    (Path(state["path"]) / "value.txt").write_text("worker\n")
    inspected = manager.inspect(str(repo), state["path"], state["base_ref"])
    (repo / "value.txt").write_text("user-later\n")
    with pytest.raises(WorktreeError, match="conflicts"):
        manager.promote(str(repo), state["path"], state["base_ref"], inspected["diff_sha256"])
    assert (repo / "value.txt").read_text() == "user-later\n"


@pytest.mark.asyncio
async def test_agent_worker_promotion_requires_fresh_isolated_verification(tmp_path, monkeypatch):
    if not Path("/usr/bin/bwrap").exists():
        pytest.skip("Bubblewrap is required for isolated worker promotion")
    monkeypatch.setenv("GLM_ACP_CONFIG_DIR", str(tmp_path / "config"))
    repo = _repo(tmp_path / "repo")
    manager = WorktreeManager(tmp_path / "workers")
    state = manager.create(str(repo))
    (Path(state["path"]) / "value.txt").write_text("worker\n")
    inspected = manager.inspect(str(repo), state["path"], state["base_ref"])
    agent = GlmAcpAgent()
    agent._worktrees = manager
    session = Session("session", str(repo))
    result = await agent._worktree_worker(
        session,
        {
            "action": "promote",
            "worker_path": state["path"],
            "base_ref": state["base_ref"],
            "diff_sha256": inspected["diff_sha256"],
            "verification_command": "test -f value.txt",
        },
    )
    assert result.changed_paths == [str(repo / "value.txt")]
    assert (repo / "value.txt").read_text() == "worker\n"
    await agent.aclose()


def test_language_aware_references_rank_definitions_and_task_files(tmp_path, monkeypatch):
    monkeypatch.setattr("glm_acp.references.which", lambda _name: None)
    (tmp_path / "unrelated.py").write_text("def noise():\n    return 1\n")
    (tmp_path / "caller.py").write_text(
        "from service import PaymentService\n\ndef use():\n    return PaymentService()\n"
    )
    (tmp_path / "service.py").write_text(
        "class PaymentService:\n    def calculate_invoice(self):\n        return 1\n"
    )
    expanded, targets = expand_references(
        "Fix PaymentService invoice behavior @folder:. @symbol:PaymentService",
        Sandbox(tmp_path),
    )
    assert expanded.index("service.py") < expanded.index("unrelated.py")
    symbol_section = expanded.split("Reference @symbol:PaymentService:", 1)[1]
    assert symbol_section.index("service.py") < symbol_section.index("caller.py")
    assert str(tmp_path / "service.py") in targets
    assert "language=python" in expanded


def test_failure_drafts_are_metadata_only_and_promote_to_runnable_cases(tmp_path):
    corpus = FailureCorpus(tmp_path / "private" / "drafts.jsonl")
    draft = corpus.record_draft(
        str(tmp_path),
        "run_command",
        "pytest failed token=super-secret-value",
        [str(tmp_path / "module.py")],
    )
    assert draft is not None
    raw = corpus.path.read_text()
    assert "super-secret-value" not in raw
    assert str(tmp_path) not in raw
    target = corpus.promote(
        str(tmp_path),
        draft["fingerprint"],
        {
            "id": "python-regression",
            "prompt": "Correct add so the outcome test passes.",
            "files": {"maths.py": "def add(a, b): return a - b\n"},
            "verify": ["python", "-m", "pytest", "-q"],
            "timeout": 180,
        },
    )
    cases = load_cases(target)
    assert cases[0]["source_failure"] == draft["fingerprint"]
    assert corpus.list() == []


def test_observability_aggregates_metadata_and_ignores_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_ACP_TELEMETRY", "1")
    path = tmp_path / "trajectory.jsonl"
    recorder = TrajectoryRecorder(path)
    recorder.record(
        "tool_call", "session", tool="read_file", success=True, duration_ms=10, paths=["/secret/x"]
    )
    recorder.record("tool_call", "session", tool="run_command", success=False, duration_ms=30)
    recorder.record(
        "llm_call",
        "session",
        input_tokens=100,
        output_tokens=20,
        cached_tokens=25,
        duration_ms=40,
    )
    with path.open("ab") as stream:
        stream.write(b"not-json\n")
    raw = path.read_text()
    assert "/secret/x" not in raw
    snapshot = observability_snapshot(path)
    assert snapshot["tools"]["success_rate"] == 0.5
    assert snapshot["llm"]["cache_hit_ratio"] == 0.25
    assert "Local Observability" in render_observability(snapshot)


def test_ed25519_signed_plugins_require_trusted_exact_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    guidance = source / "guidance.md"
    guidance.write_text("Use the repository verifier.")
    manifest = source / "plugin.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "id": "signed_quality",
                "version": "1.0.0",
                "permissions": ["prompt_context"],
                "prompt_files": ["guidance.md"],
                "files": {"guidance.md": hashlib.sha256(guidance.read_bytes()).hexdigest()},
            }
        )
    )
    private_key = tmp_path / "publisher.private.json"
    public_key = tmp_path / "publisher.public.json"
    generate_signing_key(private_key, public_key, "example.publisher")
    sign_plugin_manifest(manifest, private_key)
    registry = PluginRegistry(tmp_path / "installed")
    publisher, key = read_public_key(public_key)
    registry.trust_publisher(publisher, key)
    installed = registry.install(manifest)
    assert installed["signed"] is True
    assert installed["publisher"] == "example.publisher"
    assert registry.verify("signed_quality")["trust"] == "trusted-publisher"

    payload = json.loads(manifest.read_text())
    payload["version"] = "2.0.0"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(PluginError, match="signature verification failed"):
        registry.install(manifest)

    unsigned = source / "unsigned.json"
    payload.pop("signature")
    unsigned.write_text(json.dumps(payload))
    monkeypatch.setenv("GLM_ACP_REQUIRE_SIGNED_PLUGINS", "1")
    with pytest.raises(PluginError, match="Unsigned plugins"):
        registry.install(unsigned)

    if os.name != "nt":
        assert private_key.stat().st_mode & 0o777 == 0o600


def test_bounded_offline_fuzz_and_fault_injection_harness():
    result = run_hardening_checks(iterations=20, seed=7)
    assert result["passed"] is True
    assert result["checks"]["malformed_telemetry"] == 20
    assert result["checks"]["promotion_rollback_fault"] is True
