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
    parser.add_argument(
        "--setup",
        action="store_true",
        help="store Z.ai API credentials for Registry and editor launches",
    )
    parser.add_argument(
        "--check-auth",
        action="store_true",
        help="check whether usable credentials are configured without printing them",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    asyncio.run(run())
    return 0
