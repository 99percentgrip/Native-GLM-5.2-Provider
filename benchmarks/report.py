#!/usr/bin/env python3
"""Create a compact Markdown leaderboard from benchmark JSON reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or not isinstance(report.get("results"), list):
        raise ValueError(f"unsupported benchmark report: {path}")
    return report


def row(report: dict[str, Any]) -> list[str]:
    elapsed = [
        float(item["elapsed_seconds"])
        for item in report["results"]
        if isinstance(item.get("elapsed_seconds"), (int, float))
        and not item.get("verification", {}).get("skipped")
    ]
    input_tokens = sum(int(item.get("input_tokens", 0)) for item in report["results"])
    output_tokens = sum(int(item.get("output_tokens", 0)) for item in report["results"])
    median = f"{statistics.median(elapsed):.2f}" if elapsed else "—"
    return [
        str(report.get("label", report.get("runner", "unknown"))),
        f"{100 * float(report.get('pass_rate', 0)):.1f}%",
        f"{report.get('passed', 0)}/{report.get('total', 0)}",
        str(report.get("skipped", 0)),
        median,
        f"{input_tokens:,}",
        f"{output_tokens:,}",
    ]


def case_cell(report: dict[str, Any], case_id: str) -> str:
    attempts = [item for item in report["results"] if item.get("id") == case_id]
    scored = [item for item in attempts if not item.get("verification", {}).get("skipped")]
    if not scored:
        return "skipped"
    passed = sum(bool(item.get("verification", {}).get("passed")) for item in scored)
    elapsed = [
        float(item["elapsed_seconds"])
        for item in scored
        if isinstance(item.get("elapsed_seconds"), (int, float))
    ]
    timing = f" · {statistics.median(elapsed):.2f}s" if elapsed else ""
    return f"{passed}/{len(scored)}{timing}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [load_report(path) for path in args.reports]
    reports.sort(key=lambda item: (-float(item.get("pass_rate", 0)), str(item.get("label", ""))))
    lines = [
        "# Native GLM ACP quality report",
        "",
        "Outcome-based, isolated coding tasks. Lower median time is better after pass rate.",
        "",
        "| Runner | Pass rate | Passed | Skipped | Median seconds | Input tokens | Output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append("| " + " | ".join(row(report)) + " |")
    case_ids = sorted(
        {str(item.get("id")) for report in reports for item in report["results"] if item.get("id")}
    )
    labels = [str(report.get("label", report.get("runner", "unknown"))) for report in reports]
    lines.extend(
        [
            "",
            "## Per-case outcomes",
            "",
            "Each cell is passed attempts / scored attempts and median elapsed time.",
            "",
            "| Case | " + " | ".join(labels) + " |",
            "|---|" + "---:|" * len(labels),
        ]
    )
    for case_id in case_ids:
        lines.append(
            "| "
            + case_id
            + " | "
            + " | ".join(case_cell(report, case_id) for report in reports)
            + " |"
        )
    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
