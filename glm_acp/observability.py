"""Secret-safe local observability summaries over metadata-only trajectories."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .telemetry import trajectory_path

MAX_OBSERVABILITY_EVENTS = 50_000
MAX_OBSERVABILITY_BYTES = 20 * 1024 * 1024


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return int(ordered[index])


def _safe_int(value: Any) -> int:
    """Treat malformed metadata as zero instead of breaking the dashboard."""
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _events(path: Path, max_events: int) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size > MAX_OBSERVABILITY_BYTES:
        with path.open("rb") as stream:
            stream.seek(-MAX_OBSERVABILITY_BYTES, 2)
            stream.readline()
            raw_lines = stream.readlines()
    else:
        raw_lines = path.read_bytes().splitlines()
    output: list[dict[str, Any]] = []
    for raw in raw_lines[-max_events:]:
        if len(raw) > 4096:
            continue
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema") == 1:
            output.append(value)
    return output


def observability_snapshot(
    path: Path | None = None, max_events: int = MAX_OBSERVABILITY_EVENTS
) -> dict[str, Any]:
    """Aggregate bounded local metrics without exposing event bodies or session ids."""
    events = _events(path or trajectory_path(), min(max(1, max_events), MAX_OBSERVABILITY_EVENTS))
    tools = [event for event in events if event.get("event") == "tool_call"]
    llm = [event for event in events if event.get("event") == "llm_call"]
    turns = [event for event in events if event.get("event") == "turn_complete"]
    certificates = [event for event in events if event.get("event") == "completion_certificate"]
    capability = [event for event in events if event.get("event") == "capability_outcome"]
    critics = [event for event in events if event.get("event") == "evidence_critic"]
    hypothesis_sets = [event for event in events if event.get("event") == "hypothesis_set"]
    hypothesis_tests = [event for event in events if event.get("event") == "hypothesis_test"]
    voi = [event for event in events if event.get("event") == "voi_selection"]
    repository_predictions = [
        event for event in events if event.get("event") == "repository_prediction"
    ]
    repository_impacts = [event for event in events if event.get("event") == "repository_impact"]
    attributions = [event for event in events if event.get("event") == "causal_attribution"]
    meta_evaluations = [
        event for event in events if event.get("event") == "metacognitive_evaluation"
    ]
    meta_candidates = [event for event in events if event.get("event") == "metacognitive_candidate"]
    durations = [_safe_int(event.get("duration_ms")) for event in tools]
    llm_durations = [_safe_int(event.get("duration_ms")) for event in llm]
    tool_counts = Counter(str(event.get("tool", "unknown")) for event in tools)
    tool_failures = Counter(
        str(event.get("tool", "unknown")) for event in tools if not event.get("success", False)
    )
    input_tokens = sum(_safe_int(event.get("input_tokens")) for event in llm)
    output_tokens = sum(_safe_int(event.get("output_tokens")) for event in llm)
    cached_tokens = sum(_safe_int(event.get("cached_tokens")) for event in llm)
    sessions = {str(event.get("session", "")) for event in events if event.get("session")}
    modes = Counter(str(event.get("execution_mode", "unknown")) for event in capability)
    families = Counter(str(event.get("task_family", "unknown")) for event in capability)
    return {
        "schema": 1,
        "events": len(events),
        "sessions": len(sessions),
        "window": {
            "first": str(events[0].get("timestamp", "")) if events else "",
            "last": str(events[-1].get("timestamp", "")) if events else "",
        },
        "turns": {
            "completed": len(turns),
            "freshly_verified": sum(bool(event.get("fresh_verification")) for event in turns),
            "changed_files": sum(_safe_int(event.get("changed_files")) for event in turns),
        },
        "awareness": {
            "certificates": len(certificates),
            "complete": sum(bool(event.get("complete")) for event in certificates),
            "prevented_false_completion": sum(
                bool(event.get("prevented")) for event in certificates
            ),
            "mean_evidence_coverage": round(
                sum(float(event.get("coverage", 0.0) or 0.0) for event in certificates)
                / max(len(certificates), 1),
                4,
            ),
            "active_contradictions": sum(
                _safe_int(event.get("contradictions")) for event in certificates
            ),
            "stale_evidence": sum(_safe_int(event.get("stale_evidence")) for event in certificates),
        },
        "metacognition": {
            "outcomes": len(capability),
            "success_rate": round(
                sum(bool(event.get("success")) for event in capability) / max(len(capability), 1),
                4,
            ),
            "verified_rate": round(
                sum(
                    str(event.get("verification_strength", "none")) in {"targeted", "full"}
                    for event in capability
                )
                / max(len(capability), 1),
                4,
            ),
            "mean_tokens": (
                sum(
                    _safe_int(event.get("input_tokens")) + _safe_int(event.get("output_tokens"))
                    for event in capability
                )
                // max(len(capability), 1)
            ),
            "by_mode": dict(modes.most_common(4)),
            "by_task_family": dict(families.most_common(8)),
        },
        "grounded_deliberation": {
            "critic_reviews": len(critics),
            "critic_approved": sum(event.get("outcome") == "approve" for event in critics),
            "critic_revisions": sum(event.get("outcome") == "revise" for event in critics),
            "critic_unavailable": sum(event.get("outcome") == "unavailable" for event in critics),
            "hypothesis_sets": len(hypothesis_sets),
            "hypotheses_generated": sum(_safe_int(event.get("count")) for event in hypothesis_sets),
            "hypotheses_tested": len(hypothesis_tests),
            "voi_selections": len(voi),
            "voi_match_rate": round(
                sum(bool(event.get("matched")) for event in voi) / max(len(voi), 1),
                4,
            ),
        },
        "repository_intelligence": {
            "predictions": len(repository_predictions),
            "pre_mortems": sum(
                _safe_int(event.get("premortem_items")) for event in repository_predictions
            ),
            "comparisons": sum(bool(event.get("compared")) for event in repository_impacts),
            "observed_files": sum(
                _safe_int(event.get("observed_files")) for event in repository_impacts
            ),
            "unexpected_files": sum(
                _safe_int(event.get("unexpected_files")) for event in repository_impacts
            ),
            "observed_checks": sum(
                _safe_int(event.get("observed_checks")) for event in repository_impacts
            ),
        },
        "safe_meta_learning": {
            "causal_attributions": len(attributions),
            "corrected": sum(bool(event.get("corrected")) for event in attributions),
            "evaluations": len(meta_evaluations),
            "evaluations_passed": sum(bool(event.get("passed")) for event in meta_evaluations),
            "fresh_gain": sum(_safe_int(event.get("fresh_gain")) for event in meta_evaluations),
            "mutated_gain": sum(_safe_int(event.get("mutated_gain")) for event in meta_evaluations),
            "promotions": sum(event.get("status") == "promoted" for event in meta_candidates),
        },
        "llm": {
            "calls": len(llm),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_ratio": round(cached_tokens / max(input_tokens, 1), 4),
            "latency_ms_p50": int(statistics.median(llm_durations)) if llm_durations else 0,
            "latency_ms_p95": _percentile(llm_durations, 0.95),
        },
        "tools": {
            "calls": len(tools),
            "failures": sum(tool_failures.values()),
            "success_rate": round(
                sum(bool(event.get("success")) for event in tools) / max(len(tools), 1), 4
            ),
            "latency_ms_p50": int(statistics.median(durations)) if durations else 0,
            "latency_ms_p95": _percentile(durations, 0.95),
            "by_tool": [
                {"tool": tool, "calls": count, "failures": tool_failures[tool]}
                for tool, count in tool_counts.most_common(20)
            ],
        },
        "safety": {
            "rollbacks": sum(event.get("event") == "rollback" for event in events),
            "rollback_conflicts": sum(
                event.get("event") == "rollback" and not event.get("success", False)
                for event in events
            ),
            "worker_promotions": sum(
                event.get("event") == "worker_promotion" and event.get("success", False)
                for event in events
            ),
        },
    }


def render_observability(snapshot: dict[str, Any]) -> str:
    tools = snapshot["tools"]
    llm = snapshot["llm"]
    turns = snapshot["turns"]
    safety = snapshot["safety"]
    awareness = snapshot["awareness"]
    metacognition = snapshot["metacognition"]
    deliberation = snapshot["grounded_deliberation"]
    repository = snapshot["repository_intelligence"]
    meta_learning = snapshot["safe_meta_learning"]
    by_tool = (
        "\n".join(
            f"- `{item['tool']}`: {item['calls']} calls, {item['failures']} failures"
            for item in tools["by_tool"]
        )
        or "- No tool activity recorded."
    )
    return (
        "📈 **Local Observability**\n"
        f"- Window: {snapshot['window']['first'] or 'empty'} → "
        f"{snapshot['window']['last'] or 'empty'}\n"
        f"- Sessions: {snapshot['sessions']} · completed turns: {turns['completed']} · "
        f"freshly verified: {turns['freshly_verified']}\n"
        f"- LLM: {llm['calls']} calls · {llm['input_tokens']:,} input · "
        f"{llm['output_tokens']:,} output · {llm['cache_hit_ratio']:.1%} cache hit · "
        f"p95 {llm['latency_ms_p95']} ms\n"
        f"- Tools: {tools['calls']} calls · {tools['success_rate']:.1%} success · "
        f"p95 {tools['latency_ms_p95']} ms\n"
        f"- Safety: {safety['rollbacks']} rollbacks ({safety['rollback_conflicts']} conflicts) · "
        f"{safety['worker_promotions']} worker promotions\n\n"
        f"- Awareness: {awareness['complete']}/{awareness['certificates']} certificates complete · "
        f"{awareness['mean_evidence_coverage']:.1%} mean evidence coverage · "
        f"{awareness['prevented_false_completion']} unsupported completions prevented\n\n"
        f"- Metacognition: {metacognition['outcomes']} empirical outcomes · "
        f"{metacognition['success_rate']:.1%} success · "
        f"{metacognition['verified_rate']:.1%} verified · "
        f"{metacognition['mean_tokens']:,} mean tokens\n"
        f"- Adaptive modes: {json.dumps(metacognition['by_mode'], sort_keys=True)}\n\n"
        f"- Grounded deliberation: {deliberation['critic_reviews']} critic reviews · "
        f"{deliberation['critic_revisions']} revisions requested · "
        f"{deliberation['hypotheses_tested']}/"
        f"{deliberation['hypotheses_generated']} hypotheses tested · "
        f"{deliberation['voi_match_rate']:.1%} VOI match\n\n"
        f"- Repository intelligence: {repository['predictions']} predictions · "
        f"{repository['comparisons']} observed comparisons · "
        f"{repository['unexpected_files']} unexpected files · "
        f"{repository['pre_mortems']} pre-mortem risks\n"
        f"- Safe metacognitive learning: {meta_learning['causal_attributions']} attributions · "
        f"{meta_learning['evaluations_passed']}/{meta_learning['evaluations']} gates passed · "
        f"{meta_learning['promotions']} explicit promotions\n\n"
        "**Most-used tools**\n" + by_tool
    )
