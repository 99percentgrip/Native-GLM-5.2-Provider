#!/usr/bin/env python3
"""Run isolated, outcome-based coding-agent benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REQUIRED_CASE_FIELDS = {"id", "prompt", "files", "verify", "timeout"}


def load_cases() -> list[dict[str, Any]]:
    cases = json.loads((Path(__file__).with_name("cases.json")).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases.json must contain a non-empty list")
    seen: set[str] = set()
    for case in cases:
        missing = REQUIRED_CASE_FIELDS - set(case)
        if missing:
            raise ValueError(f"benchmark case is missing fields: {sorted(missing)}")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"benchmark case id is invalid or duplicated: {case_id!r}")
        seen.add(case_id)
        if not isinstance(case["files"], dict) or not case["files"]:
            raise ValueError(f"benchmark case {case_id} must define fixture files")
        if not isinstance(case["verify"], list) or not case["verify"]:
            raise ValueError(f"benchmark case {case_id} must define a verification command")
    return cases


def prepare(case: dict[str, Any], root: Path) -> None:
    for relative, content in case["files"].items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"benchmark fixture escapes workspace: {relative}") from error
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


async def run_native(case: dict[str, Any], root: Path) -> dict[str, Any]:
    from glm_acp.agent import GlmAcpAgent, Session

    timing: dict[str, float | None] = {"started": None, "first": None}

    class QuietClient:
        async def session_update(self, **_: Any) -> None:
            if timing["started"] is not None and timing["first"] is None:
                timing["first"] = time.perf_counter()
            return None

        async def request_permission(self, **_: Any) -> Any:
            raise RuntimeError("benchmark unexpectedly requested permission")

    agent = GlmAcpAgent()
    agent.on_connect(QuietClient())
    session = Session(f"benchmark-{case['id']}", str(root))
    session.permission_mode = "bypass"
    session.messages.append({"role": "user", "content": case["prompt"]})
    agent._sessions[session.id] = session
    started = time.perf_counter()
    timing["started"] = started
    try:
        try:
            stop_reason = await asyncio.wait_for(
                agent._run_turn(session), timeout=float(case["timeout"])
            )
        except asyncio.TimeoutError:
            return {"stop_reason": "timeout", "elapsed_seconds": float(case["timeout"])}
        except Exception as error:
            return {
                "stop_reason": "runner_error",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(error).__name__,
            }
        return {
            "stop_reason": stop_reason,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "first_delta_seconds": (
                round(timing["first"] - started, 3) if timing["first"] is not None else None
            ),
            "input_tokens": session.total_input_tokens,
            "output_tokens": session.total_output_tokens,
            "cached_tokens": session.total_cached_tokens,
        }
    finally:
        await agent.aclose()


async def run_external(command: list[str], case: dict[str, Any], root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=root,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert process.stdin is not None
    try:
        process.stdin.write(case["prompt"].encode())
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        process.stdin.close()
    deadline = time.monotonic() + float(case["timeout"])
    while process.returncode is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    if process.returncode is None:
        process.kill()
        while process.returncode is None:
            await asyncio.sleep(0.05)
        return {"stop_reason": "timeout", "elapsed_seconds": case["timeout"]}
    return {
        "stop_reason": "completed" if process.returncode == 0 else "runner_error",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runner_exit_code": process.returncode,
    }


def verify(case: dict[str, Any], root: Path) -> dict[str, Any]:
    command = list(case["verify"])
    if command and command[0] == "python":
        command[0] = sys.executable
    elif command and shutil.which(command[0]) is None:
        return {
            "passed": False,
            "skipped": True,
            "exit_code": None,
            "summary": f"required executable is unavailable: {command[0]}",
        }
    try:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "exit_code": None,
            "summary": "verification timed out after 60 seconds",
        }
    return {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "summary": redact((result.stdout + result.stderr)[-1000:]),
    }


def redact(text: str) -> str:
    """Remove known credential values from any persisted benchmark output."""
    redacted = text
    for key, value in os.environ.items():
        upper = key.upper()
        if (
            value
            and len(value) >= 8
            and any(
                marker in upper for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
            )
        ):
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def candidate_metadata(runner: str) -> dict[str, str]:
    if runner != "native":
        return {}
    from glm_acp import __version__
    from glm_acp.agent import SYSTEM_PROMPT_TEMPLATE
    from glm_acp.config import DEFAULT_MODEL

    return {
        "package_version": __version__,
        "model": DEFAULT_MODEL,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
    }


def build_report(
    *,
    label: str,
    runner: str,
    repeat: int,
    planned_total: int,
    status: str,
    candidate: dict[str, str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    scored = [item for item in results if not item["verification"].get("skipped")]
    passed = sum(bool(item["verification"]["passed"]) for item in scored)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed": len(results),
        "planned_total": planned_total,
        "label": label,
        "runner": runner,
        "candidate": candidate,
        "repeat": repeat,
        "passed": passed,
        "total": len(scored),
        "skipped": len(results) - len(scored),
        "pass_rate": round(passed / len(scored), 4) if scored else 0.0,
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "results": results,
    }


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def persist_report(report: dict[str, Any], output: Path | None, markdown: Path | None) -> str:
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if output:
        write_atomic(output, rendered + "\n")
    if markdown:
        from benchmarks.report import render_reports

        write_atomic(markdown, render_reports([report]))
    return rendered


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--runner", choices=("native", "external"), default="native")
    parser.add_argument("--external-command", nargs=argparse.REMAINDER)
    parser.add_argument("--case", action="append", dest="selected")
    parser.add_argument("--label", default="native-glm-acp")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    cases = load_cases()
    if args.validate:
        print(f"Validated {len(cases)} benchmark cases")
        return 0
    if args.repeat < 1 or args.repeat > 20:
        parser.error("--repeat must be between 1 and 20")
    if args.list:
        for case in cases:
            print(case["id"])
        return 0
    if args.runner == "external" and not args.external_command:
        parser.error("--external-command is required for an external runner")
    if args.runner == "native":
        from glm_acp.config import has_api_key

        if not has_api_key():
            print(
                "Native benchmark requires configured Z.ai credentials. "
                "Run `glm-acp --setup` or set ZAI_API_KEY.",
                file=sys.stderr,
            )
            return 2
    selected = set(args.selected or [])
    known = {str(case["id"]) for case in cases}
    unknown = selected - known
    if unknown:
        parser.error(f"unknown benchmark case(s): {', '.join(sorted(unknown))}")
    selected_cases = [case for case in cases if not selected or case["id"] in selected]
    planned_total = len(selected_cases) * args.repeat
    candidate = candidate_metadata(args.runner)
    results: list[dict[str, Any]] = []
    for attempt in range(1, args.repeat + 1):
        for case in selected_cases:
            ordinal = len(results) + 1
            print(
                f"[{ordinal}/{planned_total}] Running {case['id']} (attempt {attempt})...",
                file=sys.stderr,
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix=f"glm-eval-{case['id']}-") as temp:
                workspace = Path(temp)
                prepare(case, workspace)
                if args.runner == "native":
                    run = await run_native(case, workspace)
                else:
                    run = await run_external(args.external_command, case, workspace)
                verification = verify(case, workspace)
                results.append(
                    {"id": case["id"], "attempt": attempt, **run, "verification": verification}
                )
            outcome = (
                "SKIP"
                if verification.get("skipped")
                else ("PASS" if verification["passed"] else "FAIL")
            )
            print(
                f"[{ordinal}/{planned_total}] {outcome} {case['id']} "
                f"({run.get('elapsed_seconds', 0):.2f}s)",
                file=sys.stderr,
                flush=True,
            )
            partial = build_report(
                label=args.label,
                runner=args.runner,
                repeat=args.repeat,
                planned_total=planned_total,
                status="running",
                candidate=candidate,
                results=results,
            )
            persist_report(partial, args.output, args.markdown_output)
    report = build_report(
        label=args.label,
        runner=args.runner,
        repeat=args.repeat,
        planned_total=planned_total,
        status="completed",
        candidate=candidate,
        results=results,
    )
    rendered = persist_report(report, args.output, args.markdown_output)
    print(rendered)
    return 0 if report["passed"] == report["total"] and report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
