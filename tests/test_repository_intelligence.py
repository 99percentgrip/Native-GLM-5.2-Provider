"""Repository intelligence and safe metacognitive-learning contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from glm_acp.agent import GlmAcpAgent, Session
from glm_acp.awareness import EpistemicLedger
from glm_acp.cli import main
from glm_acp.meta_learning import (
    SafeMetacognitiveLearning,
    built_in_evaluation_cases,
    evaluate_metacognitive_reports,
)
from glm_acp.metacognition import CapabilityProfiles, MetacognitiveController
from glm_acp.observability import observability_snapshot, render_observability
from glm_acp.project_context import detect_project_facts
from glm_acp.repository_intelligence import MAX_EDGES, MAX_NODES, RepositoryIntelligence


def _assessment(root: Path, task: str):
    return MetacognitiveController(profiles=CapabilityProfiles()).assess(
        task=task,
        facts=detect_project_facts(root),
        ledger=EpistemicLedger(),
        permission_mode="bypass",
        session_mode="code",
        changed_paths=[],
        fresh_verification=False,
        persistent_goal=False,
    )


def test_lazy_world_combines_repository_signals_and_compares_impact(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "AGENTS.md").write_text("# Rules\n")
    (tmp_path / "CODEOWNERS").write_text("app/** @runtime-team\n")
    package = tmp_path / "app"
    package.mkdir()
    source = package / "service.py"
    source.write_text("from app.helper import run\n")
    helper = package / "helper.py"
    helper.write_text("def run():\n    return 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    test_file = tests / "test_service.py"
    test_file.write_text("def test_service():\n    assert True\n")
    intelligence = RepositoryIntelligence()
    task = "Implement and release service changes safely"
    intelligence.prepare(
        task=task,
        facts=detect_project_facts(tmp_path),
        targets=[str(source)],
        changed_paths=[],
        assessment=_assessment(tmp_path, task),
        edit_generation=0,
        failure_drafts=[{"failure_kind": "verification"}],
    )

    assert "pyproject.toml" in intelligence.nodes
    assert "AGENTS.md" in intelligence.nodes
    assert "app/service.py" in intelligence.nodes
    assert "app/helper.py" in intelligence.nodes
    assert intelligence.failure_kinds == {"verification": 1}
    assert intelligence.ownership["app/service.py"] == ("@runtime-team",)
    assert intelligence.prediction is not None
    assert intelligence.premortem
    predicted_before_edit = list(intelligence.prediction.files)
    intelligence.prepare(
        task=task,
        facts=detect_project_facts(tmp_path),
        targets=[str(source), str(test_file)],
        changed_paths=[str(source)],
        assessment=_assessment(tmp_path, task),
        edit_generation=1,
        failure_drafts=[{"failure_kind": "verification"}],
    )
    assert intelligence.prediction.edit_generation == 0
    assert intelligence.prediction.files == predicted_before_edit
    intelligence.compare(str(tmp_path), [str(source), str(test_file)], ["pytest"])
    assert intelligence.prediction.compared
    assert "tests/test_service.py" in intelligence.prediction.observed_files
    assert len(intelligence.nodes) <= MAX_NODES
    assert len(intelligence.edges) <= MAX_EDGES


def test_direct_small_task_has_no_repository_overhead(tmp_path: Path) -> None:
    intelligence = RepositoryIntelligence()
    intelligence.prepare(
        task="hello",
        facts=detect_project_facts(tmp_path),
        targets=[],
        changed_paths=[],
        assessment=_assessment(tmp_path, "hello"),
        edit_generation=0,
    )
    assert intelligence.nodes == {}
    assert intelligence.prediction is None
    assert intelligence.premortem == []


def test_world_state_round_trips_without_file_bodies(tmp_path: Path) -> None:
    secret = "body-that-must-not-be-persisted"
    source = tmp_path / "module.py"
    source.write_text(f"VALUE = '{secret}'\n")
    task = "implement module"
    original = RepositoryIntelligence()
    original.prepare(
        task=task,
        facts=detect_project_facts(tmp_path),
        targets=[str(source)],
        changed_paths=[],
        assessment=_assessment(tmp_path, task),
        edit_generation=0,
    )
    serialized = json.dumps(original.to_dict())
    restored = RepositoryIntelligence(json.loads(serialized))
    assert secret not in serialized
    assert restored.nodes.keys() == original.nodes.keys()
    corrupt = RepositoryIntelligence(
        {
            "builds": "bad",
            "nodes": {},
            "failure_kinds": [],
            "ownership": [],
            "prediction": {"files": "not-a-list", "edit_generation": "bad"},
        }
    )
    assert corrupt.builds == 0
    assert corrupt.prediction is not None
    assert corrupt.prediction.files == []


def _reports() -> tuple[dict, dict]:
    baseline = {
        "cases": [
            {
                "id": "known",
                "split": "train",
                "passed": True,
                "confidence": 0.8,
                "latency_ms": 100,
                "tokens": 100,
                "evidence_freshness": 0.7,
            },
            {
                "id": "fresh",
                "split": "fresh",
                "passed": False,
                "confidence": 0.7,
                "latency_ms": 100,
                "tokens": 100,
                "false_completion": 1,
                "unsupported_claims": 1,
                "evidence_freshness": 0.3,
            },
            {
                "id": "mutation",
                "split": "fresh",
                "mutation_of": "known",
                "passed": False,
                "confidence": 0.7,
                "latency_ms": 100,
                "tokens": 100,
                "false_completion": 1,
                "unsupported_claims": 1,
                "evidence_freshness": 0.3,
            },
        ]
    }
    candidate = {
        "cases": [
            {
                "id": "known",
                "split": "train",
                "passed": True,
                "confidence": 0.9,
                "latency_ms": 102,
                "tokens": 102,
                "evidence_freshness": 0.8,
            },
            {
                "id": "fresh",
                "split": "fresh",
                "passed": True,
                "confidence": 0.9,
                "latency_ms": 102,
                "tokens": 102,
                "correct_clarification": 1,
                "evidence_freshness": 0.9,
            },
            {
                "id": "mutation",
                "split": "fresh",
                "mutation_of": "known",
                "passed": True,
                "confidence": 0.9,
                "latency_ms": 102,
                "tokens": 102,
                "evidence_freshness": 0.9,
            },
        ]
    }
    return baseline, candidate


def test_causal_attribution_drafts_but_does_not_auto_promote() -> None:
    learning = SafeMetacognitiveLearning()
    for generation, evidence in ((0, "ev1"), (1, "ev2")):
        learning.observe(
            tool="run_command",
            success=False,
            output="tests failed",
            evidence_ids=[],
            edit_generation=generation,
        )
        attribution = learning.observe(
            tool="run_command",
            success=True,
            output="tests passed",
            evidence_ids=[evidence],
            edit_generation=generation,
        )
        assert attribution is not None
        assert attribution.cause == "verification_failure"
        assert attribution.intervention == "invoke_verifier"
    strategy = learning.candidates["invoke_verifier_after_edits"]
    assert strategy.status == "draft"
    baseline, candidate = _reports()
    gate = learning.evaluate(baseline, candidate)
    assert gate.passed
    promoted = learning.promote("invoke_verifier_after_edits", gate)
    assert promoted.status == "promoted"
    assert gate.digest in learning.model_context()


def test_missing_context_can_attribute_lsp_intervention() -> None:
    learning = SafeMetacognitiveLearning()
    learning.observe(
        tool="read_file",
        success=False,
        output="target not found",
        evidence_ids=[],
        edit_generation=0,
    )
    attribution = learning.observe(
        tool="semantic_code",
        success=True,
        output="definition",
        evidence_ids=["ev9"],
        edit_generation=0,
    )
    assert attribution is not None
    assert attribution.cause == "missing_context"
    assert "use_lsp_for_code_navigation" in learning.candidates


def test_fresh_mutated_gate_rejects_cost_safety_and_static_only() -> None:
    baseline, candidate = _reports()
    assert evaluate_metacognitive_reports(baseline, candidate).passed
    candidate["cases"][1]["unsafe_actions"] = 1
    candidate["cases"][1]["latency_ms"] = 1_000
    gate = evaluate_metacognitive_reports(baseline, candidate)
    assert not gate.passed
    assert any("unsafe" in reason for reason in gate.reasons)
    with pytest.raises(ValueError, match="time-split fresh"):
        evaluate_metacognitive_reports(
            {"cases": [{"id": "known", "split": "train", "passed": False}]},
            {"cases": [{"id": "known", "split": "train", "passed": True}]},
        )


def test_evaluation_corpus_covers_requested_cases_and_mutations() -> None:
    cases = built_in_evaluation_cases()
    names = {item["id"] for item in cases}
    assert len(cases) == 22
    assert "decisive-ambiguity" in names
    assert "tool-unavailable" in names
    assert "wrong-first-diagnosis" in names
    assert "stale-test-after-edit" in names
    assert "contradictory-tools" in names
    assert "unverified-worker" in names
    assert "high-risk-independent-verifier" in names
    assert "trivial-no-reflection" in names
    assert "malicious-belief-output" in names
    assert "compaction-keeps-uncertainty" in names
    assert "cannot-establish-safely" in names
    assert sum(bool(item.get("mutation_of")) for item in cases) == 11


@pytest.mark.asyncio
async def test_commands_and_session_persistence_are_inspectable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("GLM_ACP_CONFIG_DIR", str(tmp_path / "config"))
    session = Session("intelligence", str(tmp_path))
    session.refresh_system_prompt("release package")
    agent = GlmAcpAgent()
    agent._save_session = AsyncMock()
    repository = await agent._handle_command(session, "/repository")
    learning = await agent._handle_command(session, "/meta-learning")
    restored = Session.from_dict(session.to_dict(), "restored")
    assert "Repository Intelligence" in repository
    assert "Safe Metacognitive Learning" in learning
    assert restored.repository_intelligence.objective_hash
    assert "repository_intelligence" in session.to_dict()
    assert "meta_learning" in session.to_dict()
    await agent.aclose()


def test_cli_exposes_cases_and_evaluation_gate(tmp_path: Path, capsys) -> None:
    baseline, candidate = _reports()
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    assert main(["meta-cases", "--json"]) == 0
    assert "mutated-decisive-ambiguity" in capsys.readouterr().out
    assert main(["meta-eval", str(baseline_path), str(candidate_path)]) == 0
    assert '"passed": true' in capsys.readouterr().out


def test_observability_aggregates_intelligence_without_bodies(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    events = [
        {"schema": 1, "event": "repository_prediction", "session": "hash", "premortem_items": 3},
        {
            "schema": 1,
            "event": "repository_impact",
            "session": "hash",
            "compared": True,
            "observed_files": 2,
            "unexpected_files": 1,
            "observed_checks": 1,
        },
        {
            "schema": 1,
            "event": "causal_attribution",
            "session": "hash",
            "cause": "verification_failure",
            "intervention": "invoke_verifier",
            "corrected": True,
        },
        {
            "schema": 1,
            "event": "metacognitive_evaluation",
            "session": "hash",
            "passed": True,
            "fresh_gain": 1,
            "mutated_gain": 1,
        },
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in events))
    snapshot = observability_snapshot(path)
    assert snapshot["repository_intelligence"]["unexpected_files"] == 1
    assert snapshot["safe_meta_learning"]["corrected"] == 1
    assert snapshot["safe_meta_learning"]["fresh_gain"] == 1
    rendered = render_observability(snapshot)
    assert "Repository intelligence" in rendered
    assert "Safe metacognitive learning" in rendered
    assert "verification_failure" not in json.dumps(snapshot)
