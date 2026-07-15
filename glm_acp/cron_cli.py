"""CLI adapter for the native cron subsystem."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .cron import (
    CronError,
    create_job,
    get_job,
    list_jobs,
    pause_job,
    remove_job,
    resume_job,
    status,
    update_job,
)
from .cron_scheduler import daemon, tick


def add_cron_parser(subparsers: Any) -> None:
    cron = subparsers.add_parser("cron", help="manage persistent scheduled tasks")
    commands = cron.add_subparsers(dest="cron_command", required=True)
    listing = commands.add_parser("list", help="list jobs")
    listing.add_argument("--active", action="store_true", help="hide paused/completed jobs")
    listing.add_argument("--json", action="store_true")

    create = commands.add_parser("create", help="create a job")
    create.add_argument("--schedule", required=True)
    create.add_argument("--prompt", default="")
    create.add_argument("--name")
    create.add_argument("--workdir", default=".")
    create.add_argument("--timezone", default="UTC")
    create.add_argument("--repeat", type=int)
    create.add_argument("--skill", dest="skills", action="append", default=[])
    create.add_argument("--bundle", dest="bundles", action="append", default=[])
    create.add_argument("--script")
    create.add_argument("--no-agent", action="store_true")

    edit = commands.add_parser("edit", aliases=["update"], help="update a job")
    edit.add_argument("job_id")
    edit.add_argument("--schedule")
    edit.add_argument("--prompt")
    edit.add_argument("--name")
    edit.add_argument("--workdir")
    edit.add_argument("--timezone")
    edit.add_argument("--repeat", type=int)
    edit.add_argument("--script")
    edit.add_argument("--skill", dest="skills", action="append")
    edit.add_argument("--bundle", dest="bundles", action="append")
    edit.add_argument("--no-agent", action=argparse.BooleanOptionalAction, default=None)

    for name in ("pause", "resume", "run", "remove"):
        action = commands.add_parser(name, help=f"{name} a job")
        action.add_argument("job_id")
    commands.add_parser("status", help="show scheduler status")
    tick_parser = commands.add_parser("tick", help="claim and run due jobs once")
    tick_parser.add_argument("--concurrency", type=int, default=2)
    daemon_parser = commands.add_parser("daemon", help="run the foreground scheduler")
    daemon_parser.add_argument("--interval", type=float, default=30.0)
    daemon_parser.add_argument("--concurrency", type=int, default=2)


def _display(job: dict[str, Any]) -> str:
    state = job.get("state", "unknown")
    return (
        f"{job['id']} [{state}] {job.get('name', '')}\n"
        f"  schedule: {job.get('schedule_display')}\n"
        f"  next: {job.get('next_run_at') or '-'}; runs: {job.get('run_count', 0)}; "
        f"last: {job.get('last_status') or '-'}"
    )


def run_cron_command(args: argparse.Namespace) -> int:
    try:
        command = args.cron_command
        if command == "list":
            jobs = list_jobs(include_disabled=not args.active)
            if args.json:
                print(json.dumps(jobs, ensure_ascii=False, indent=2))
            else:
                print("\n".join(_display(job) for job in jobs) if jobs else "No scheduled jobs.")
            return 0
        if command == "create":
            workdir = __import__("pathlib").Path(args.workdir).resolve()
            job = create_job(
                schedule=args.schedule,
                prompt=args.prompt,
                name=args.name,
                workspace_root=str(workdir),
                workdir=str(workdir),
                timezone_name=args.timezone,
                repeat=args.repeat,
                skills=args.skills,
                bundles=args.bundles,
                script=args.script,
                no_agent=args.no_agent,
            )
            print(_display(job))
            return 0
        if command in {"edit", "update"}:
            updates = {
                key: value
                for key, value in {
                    "schedule": args.schedule,
                    "prompt": args.prompt,
                    "name": args.name,
                    "workdir": args.workdir,
                    "timezone": args.timezone,
                    "repeat": args.repeat,
                    "script": args.script,
                    "skills": args.skills,
                    "bundles": args.bundles,
                    "no_agent": args.no_agent,
                }.items()
                if value is not None
            }
            print(_display(update_job(args.job_id, updates)))
            return 0
        if command == "pause":
            print(_display(pause_job(args.job_id)))
            return 0
        if command == "resume":
            print(_display(resume_job(args.job_id)))
            return 0
        if command == "remove":
            if not remove_job(args.job_id):
                raise CronError(f"Cron job not found: {args.job_id}")
            print(f"Removed {args.job_id}")
            return 0
        if command == "run":
            if get_job(args.job_id) is None:
                raise CronError(f"Cron job not found: {args.job_id}")
            result = asyncio.run(tick(job_id=args.job_id, force=True))
            print(json.dumps(result))
            return 0 if result["succeeded"] else 1
        if command == "status":
            print(json.dumps(status(), indent=2))
            return 0
        if command == "tick":
            print(json.dumps(asyncio.run(tick(concurrency=args.concurrency))))
            return 0
        if command == "daemon":
            asyncio.run(daemon(interval=args.interval, concurrency=args.concurrency))
            return 0
        raise CronError(f"Unknown cron command: {command}")
    except (CronError, OSError) as error:
        print(f"Cron error: {error}")
        return 1
