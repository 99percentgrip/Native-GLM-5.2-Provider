"""Command-line entry point and terminal authentication setup."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
from collections.abc import Callable, Sequence

from . import __version__
from .agent import run
from .config import get_api_key, store_api_key
from .cron_cli import add_cron_parser, run_cron_command
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.purge and not args.uninstall:
        parser.error("--purge requires --uninstall")
    if args.command == "cron":
        return run_cron_command(args)
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
