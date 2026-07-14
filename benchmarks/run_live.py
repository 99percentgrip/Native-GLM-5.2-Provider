#!/usr/bin/env python3
"""Run the native live benchmark and produce handoff-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class BenchmarkAlreadyRunning(RuntimeError):
    """Raised when another live benchmark owns the repository lock."""


def process_is_alive(pid: int) -> bool:
    if os.name == "nt":
        check = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            return True
        return any(
            len(row) > 1 and row[1] == str(pid) for row in csv.reader(check.stdout.splitlines())
        )
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LiveRunLock:
    def __init__(self, path: Path):
        self.path = path
        self.owner = self.path / "owner.json"
        self.acquired = False

    def __enter__(self) -> LiveRunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                try:
                    owner = json.loads(self.owner.read_text(encoding="utf-8"))
                    pid = int(owner.get("pid", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pid = 0
                if pid and process_is_alive(pid):
                    raise BenchmarkAlreadyRunning(
                        f"another live benchmark is already running (pid {pid})"
                    )
                shutil.rmtree(self.path, ignore_errors=True)
        else:
            raise BenchmarkAlreadyRunning("could not acquire the live benchmark lock")
        self.owner.write_text(
            json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()})
            + "\n",
            encoding="utf-8",
        )
        self.acquired = True
        return self

    def __exit__(self, *_: object) -> None:
        if not self.acquired:
            return
        try:
            owner = json.loads(self.owner.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            owner = {}
        if owner.get("pid") == os.getpid():
            shutil.rmtree(self.path, ignore_errors=True)
        self.acquired = False


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "quality" / f"live-{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the credential-safe native GLM quality/latency benchmark."
    )
    parser.add_argument("--repeat", type=int, default=3, help="attempts per case (1-20)")
    parser.add_argument("--case", action="append", dest="selected", help="run one case")
    parser.add_argument("--label", default="native-glm-acp")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate cases and credentials without making API requests",
    )
    args = parser.parse_args()
    if not 1 <= args.repeat <= 20:
        parser.error("--repeat must be between 1 and 20")

    from glm_acp.config import DEFAULT_MODEL, has_api_key

    validate = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "eval.py"), "--validate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if validate.returncode != 0:
        print("Benchmark catalog validation failed.", file=sys.stderr)
        return validate.returncode
    if not has_api_key():
        print(
            "No Z.ai credential is configured. Run `glm-acp --setup` or set "
            "ZAI_API_KEY, then retry. The key must not be placed on the command line.",
            file=sys.stderr,
        )
        return 2
    if args.check:
        print(f"Ready: benchmark catalog and credentials are valid; model={DEFAULT_MODEL}.")
        return 0

    output_dir = (args.output_dir or default_output_dir()).resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    json_path = output_dir / "native.json"
    markdown_path = output_dir / "report.md"
    command = [
        sys.executable,
        str(ROOT / "benchmarks" / "eval.py"),
        "--runner",
        "native",
        "--repeat",
        str(args.repeat),
        "--label",
        args.label,
        "--output",
        str(json_path),
        "--markdown-output",
        str(markdown_path),
    ]
    for case_id in args.selected or []:
        command.extend(("--case", case_id))
    environment = os.environ.copy()
    environment["GLM_ACP_SESSION_PERSISTENCE"] = "0"
    child: subprocess.Popen[bytes] | None = None

    def interrupt(_signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        with LiveRunLock(ROOT / "quality" / ".live-benchmark.lock"):
            output_dir.mkdir(parents=True)
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.DEVNULL,
            )
            return_code = child.wait()
    except BenchmarkAlreadyRunning as error:
        print(f"Cannot start benchmark: {error}.", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print(
            "Benchmark cancelled. Completed attempts remain in the output directory.",
            file=sys.stderr,
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if not json_path.is_file():
        print(
            f"Benchmark stopped before producing a report (exit code {return_code}).",
            file=sys.stderr,
        )
        return return_code or 1
    if not markdown_path.is_file():
        print("The JSON result exists, but the Markdown report is missing.", file=sys.stderr)
        return 1

    outcome = "all scored attempts passed" if return_code == 0 else "some attempts failed"
    print(f"Benchmark complete: {outcome}.")
    print(f"JSON result: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(
        "Send both files back for analysis; neither file contains your API key or reasoning trace."
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
