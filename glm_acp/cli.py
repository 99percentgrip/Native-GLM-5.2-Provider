"""Command-line entry point and terminal authentication setup."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__
from .agent import run
from .config import get_api_key, store_api_key
from .cron_cli import add_cron_parser, run_cron_command
from .plugin_cli import add_plugin_parser, run_plugin_command
from .uninstall import UninstallError, uninstall_release


def configure_credentials(
    prompt: Callable[[str], str] = getpass.getpass,
) -> int:
    """Interactively store a Z.ai API key without echoing it."""
    print("Native GLM ACP setup")
    print("Create or copy an API key from https://z.ai/")
    key = os.environ.get("ZAI_API_KEY") or os.environ.get("Z_AI_API_KEY")
    if not key:
        key = prompt("Z.ai API key: ")
    try:
        path = store_api_key(key)
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1
    print(f"Credentials saved to {path}")
    print("The key was not printed. Restart the ACP agent to use it.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glm-acp",
        description="Native ACP coding agent powered by Z.ai GLM models.",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--setup",
        action="store_true",
        help="store Z.ai API credentials for Registry and editor launches",
    )
    actions.add_argument(
        "--check-auth",
        action="store_true",
        help="check whether usable credentials are configured without printing them",
    )
    actions.add_argument(
        "--uninstall",
        action="store_true",
        help="remove a public frozen-binary installation and matching Zed custom entry",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="with --uninstall, also remove the stored Z.ai credential",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    add_cron_parser(subparsers)
    add_plugin_parser(subparsers)
    observe = subparsers.add_parser("observe", help="show secret-safe local reliability metrics")
    observe.add_argument("--json", action="store_true", dest="as_json")
    harden = subparsers.add_parser("harden", help="run offline fuzz and fault-injection checks")
    harden.add_argument("--iterations", type=int, default=250)
    harden.add_argument("--seed", type=int, default=5202)
    harden.add_argument("--json", action="store_true", dest="as_json")
    meta_eval = subparsers.add_parser(
        "meta-eval", help="gate a metacognitive candidate on fresh and transformed cases"
    )
    meta_eval.add_argument("baseline")
    meta_eval.add_argument("candidate")
    meta_cases = subparsers.add_parser(
        "meta-cases", help="print the built-in fresh and transformed evaluation case catalog"
    )
    meta_cases.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.purge and not args.uninstall:
        parser.error("--purge requires --uninstall")
    if args.command == "cron":
        return run_cron_command(args)
    if args.command == "plugin":
        return run_plugin_command(args)
    if args.command == "observe":
        from .observability import observability_snapshot, render_observability

        snapshot = observability_snapshot()
        print(
            json.dumps(snapshot, ensure_ascii=False, indent=2)
            if args.as_json
            else render_observability(snapshot)
        )
        return 0
    if args.command == "harden":
        from .resilience import run_hardening_checks

        result = run_hardening_checks(args.iterations, args.seed)
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            status = "passed" if result["passed"] else "failed"
            print(
                f"Offline hardening {status}: {result['iterations']} fuzz iterations; "
                f"promotion rollback fault={result['checks']['promotion_rollback_fault']}."
            )
        return 0 if result["passed"] else 1
    if args.command == "meta-cases":
        from .meta_learning import built_in_evaluation_cases

        cases = built_in_evaluation_cases()
        if args.as_json:
            print(json.dumps({"schema": 1, "cases": cases}, ensure_ascii=False, indent=2))
        else:
            for case in cases:
                suffix = f" (mutation of {case['mutation_of']})" if case.get("mutation_of") else ""
                print(f"{case['id']} [{case['split']}]{suffix}")
        return 0
    if args.command == "meta-eval":
        from .meta_learning import evaluate_metacognitive_reports

        reports = []
        try:
            for value in (args.baseline, args.candidate):
                path = Path(value)
                if path.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError(f"{path} exceeds 2 MiB")
                report = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(report, dict):
                    raise ValueError(f"{path} must contain a JSON object")
                reports.append(report)
            gate = evaluate_metacognitive_reports(reports[0], reports[1])
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Metacognitive evaluation failed closed: {error}")
            return 2
        print(json.dumps(gate.to_dict(), ensure_ascii=False, indent=2))
        return 0 if gate.passed else 1
    if args.setup:
        return configure_credentials()
    if args.check_auth:
        try:
            get_api_key()
        except RuntimeError:
            print("Z.ai credentials are not configured.")
            return 1
        print("Z.ai credentials are configured.")
        return 0
    if args.uninstall:
        try:
            result = uninstall_release(purge=args.purge)
        except UninstallError as error:
            print(f"Uninstall failed: {error}")
            return 1
        status = "scheduled for removal" if result.scheduled else "removed"
        print(f"Native GLM ACP commands {status}.")
        if result.zed_settings:
            print(f"Removed the matching Zed custom agent from {result.zed_settings}.")
            print(f"Zed settings backup: {result.zed_backup}")
        if args.purge:
            message = "removed" if result.credentials_removed else "were not present"
            print(f"Stored credentials {message}.")
        else:
            print("Stored credentials were preserved. Use --uninstall --purge to remove them.")
        print("Restart Zed to finish.")
        return 0
    asyncio.run(run())
    return 0
