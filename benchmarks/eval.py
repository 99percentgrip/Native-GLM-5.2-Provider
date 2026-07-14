#!/usr/bin/env python3
"""Run isolated, outcome-based coding-agent benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cases() -> list[dict[str, Any]]:
    return json.loads((Path(__file__).with_name("cases.json")).read_text(encoding="utf-8"))


def prepare(case: dict[str, Any], root: Path) -> None:
    for relative, content in case["files"].items():
        path = root / relative
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
        stop_reason = await asyncio.wait_for(
            agent._run_turn(session), timeout=float(case["timeout"])
        )
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
    try:
        await asyncio.wait_for(
            process.communicate(case["prompt"].encode()), timeout=float(case["timeout"])
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
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
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, timeout=60, check=False
    )
    return {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "summary": (result.stdout + result.stderr)[-1000:],
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--runner", choices=("native", "external"), default="native")
    parser.add_argument("--external-command", nargs=argparse.REMAINDER)
    parser.add_argument("--case", action="append", dest="selected")
    args = parser.parse_args()
    cases = load_cases()
    if args.list:
        for case in cases:
            print(case["id"])
        return 0
    if args.runner == "external" and not args.external_command:
        parser.error("--external-command is required for an external runner")
    selected = set(args.selected or [])
    results = []
    for case in cases:
        if selected and case["id"] not in selected:
            continue
        with tempfile.TemporaryDirectory(prefix=f"glm-eval-{case['id']}-") as temp:
            workspace = Path(temp)
            prepare(case, workspace)
            if args.runner == "native":
                run = await run_native(case, workspace)
            else:
                run = await run_external(args.external_command, case, workspace)
            results.append({"id": case["id"], **run, "verification": verify(case, workspace)})
    report = {
        "runner": args.runner,
        "passed": sum(bool(item["verification"]["passed"]) for item in results),
        "total": len(results),
        "results": results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
