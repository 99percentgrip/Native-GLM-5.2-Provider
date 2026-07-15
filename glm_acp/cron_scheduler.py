"""Bounded cron execution, delivery, ticking, and daemon lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from .cron import CLAIM_TTL_SECONDS, CronError, claim_due, cron_dir, finish_job, renew_claim
from .memory import read_learned_skill, read_skill_bundle
from .security import scan_promptware, wrap_untrusted_output

MAX_SCRIPT_SECONDS = 60
MAX_SCRIPT_OUTPUT = 64_000
DEFAULT_CONCURRENCY = 2
DEFAULT_AGENT_INACTIVITY_SECONDS = 600


class Delivery(Protocol):
    async def deliver(self, job: dict, content: str) -> None: ...


class LocalDelivery:
    async def deliver(self, job: dict, content: str) -> None:
        return None


class CallbackDelivery:
    """Narrow live-session adapter; future connectors can implement Delivery."""

    def __init__(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        self._callback = callback

    async def deliver(self, job: dict, content: str) -> None:
        session_id = job.get("origin_session_id")
        if session_id:
            await self._callback(str(session_id), content)


def is_silent(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lines = [line.strip().upper() for line in stripped.splitlines() if line.strip()]
    return stripped.upper().startswith("[SILENT]") or bool(
        lines and (lines[0] == "[SILENT]" or lines[-1] == "[SILENT]")
    )


def _safe_env() -> dict[str, str]:
    sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PRIVATE", "ACCESS")
    return {
        key: value
        for key, value in os.environ.items()
        if not any(part in key.upper() for part in sensitive) and key.upper() != "SSH_AUTH_SOCK"
    }


def _run_script_sync(job: dict) -> tuple[bool, str]:
    script = Path(job["script"]).resolve()
    root = Path(job["workspace_root"]).resolve()
    workdir = Path(job["workdir"]).resolve()
    try:
        script.relative_to(workdir)
        workdir.relative_to(root)
    except ValueError:
        return False, "Script or workdir escaped its recorded workspace"
    if not script.is_file() or script.is_symlink():
        return False, "Script is missing or is no longer a safe regular file"
    if script.suffix == ".py":
        command = [sys.executable, str(script)]
    elif script.suffix in {".sh", ".bash"}:
        command = ["bash", str(script)]
    else:
        command = [str(script)]
    kwargs: dict = {
        "cwd": workdir,
        "env": _safe_env(),
        "capture_output": True,
        "text": True,
        "errors": "replace",
        "timeout": MAX_SCRIPT_SECONDS,
        "check": False,
    }
    if os.name != "nt":
        kwargs["start_new_session"] = True
    try:
        result = subprocess.run(command, **kwargs)
    except subprocess.TimeoutExpired as error:
        text = ((error.stdout or "") + (error.stderr or ""))[:MAX_SCRIPT_OUTPUT]
        return False, f"Pre-run script timed out after {MAX_SCRIPT_SECONDS}s\n{text}"
    except OSError as error:
        return False, f"Could not execute script: {error}"
    text = (result.stdout + result.stderr)[:MAX_SCRIPT_OUTPUT]
    return result.returncode == 0, text


async def _run_agent(job: dict, prompt: str) -> str:
    """Run a fresh native agent session without persisting conversation history."""
    from .agent import GlmAcpAgent, Session

    class CaptureConnection:
        def __init__(self) -> None:
            self.output: list[str] = []
            self.last_activity = asyncio.get_running_loop().time()

        async def session_update(self, **kwargs):
            self.last_activity = asyncio.get_running_loop().time()
            update = kwargs.get("update")
            chunk = getattr(update, "content", None)
            if chunk is not None:
                text = getattr(chunk, "text", None)
                if text:
                    self.output.append(str(text))

        async def request_permission(self, **kwargs):
            raise RuntimeError("Scheduled runs cannot request interactive permission")

    connection = CaptureConnection()
    agent = GlmAcpAgent()
    agent.on_connect(connection)  # type: ignore[arg-type]
    session = Session(f"cron-{job['id']}-{job['claim']['token'][:8]}", job["workdir"])
    session.permission_mode = "bypass"
    session.scheduled_run = True
    session.task_context = "scheduled cron run"
    session.messages[0]["content"] += (
        "\n\nScheduled-run constraints:\n"
        "- This is a fresh isolated run with no interactive user present.\n"
        "- Never create, update, pause, resume, run, or remove cron jobs.\n"
        "- Begin or end with [SILENT] when there is nothing useful to report.\n"
    )
    session.messages.append({"role": "user", "content": prompt})
    try:
        raw_limit = os.environ.get("GLM_ACP_CRON_TIMEOUT", "").strip()
        try:
            inactivity_limit = float(raw_limit) if raw_limit else DEFAULT_AGENT_INACTIVITY_SECONDS
        except ValueError:
            inactivity_limit = DEFAULT_AGENT_INACTIVITY_SECONDS
        task = asyncio.create_task(agent._run_turn(session), name=f"cron-agent-{job['id']}")
        while not task.done():
            if inactivity_limit <= 0:
                await asyncio.shield(task)
                break
            idle = asyncio.get_running_loop().time() - connection.last_activity
            remaining = inactivity_limit - idle
            if remaining <= 0:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise CronError(
                    f"Scheduled agent produced no activity for {int(inactivity_limit)} seconds"
                )
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=min(5.0, remaining))
            except asyncio.TimeoutError:
                continue
        await task
        return "".join(connection.output).strip()
    finally:
        await agent._invalidate_session_client(session)
        await agent._mcp.aclose()


def _build_prompt(job: dict, script_output: str | None) -> str:
    sections: list[str] = []
    for name in job.get("skills") or []:
        sections.append(f"# Project skill: {name}\n{read_learned_skill(job['workdir'], name)}")
    for name in job.get("bundles") or []:
        sections.append(read_skill_bundle(job["workdir"], name))
    if script_output is not None:
        sections.append(
            "# Pre-run script output\n" + wrap_untrusted_output(script_output, "cron-script")
        )
    sections.append("# Scheduled task\n" + job.get("prompt", ""))
    prompt = "\n\n".join(sections)
    findings = scan_promptware(prompt)
    if findings:
        raise CronError(
            "Assembled cron prompt blocked: " + ", ".join(item.code for item in findings)
        )
    return prompt


async def run_claimed(job: dict, *, delivery: Delivery | None = None) -> bool:
    delivery = delivery or LocalDelivery()
    token = job["claim"]["token"]
    status = "error"
    output = ""
    error: str | None = None
    stop_heartbeat = asyncio.Event()

    async def heartbeat() -> None:
        interval = max(1.0, CLAIM_TTL_SECONDS / 3)
        while True:
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                if not await asyncio.to_thread(renew_claim, job["id"], token):
                    return

    heartbeat_task = asyncio.create_task(heartbeat(), name=f"cron-claim-{job['id']}")
    try:
        script_output = None
        if job.get("script"):
            ok, script_output = await asyncio.to_thread(_run_script_sync, job)
            if not ok:
                raise CronError(script_output)
        if job.get("no_agent"):
            output = script_output or ""
        else:
            output = await _run_agent(job, _build_prompt(job, script_output))
        status = "silent" if is_silent(output) else "ok"
        if status != "silent":
            await delivery.deliver(job, output)
        return True
    except asyncio.CancelledError:
        status = "cancelled"
        error = "Scheduled run cancelled during shutdown"
        raise
    except Exception as caught:
        error = str(caught)
        return False
    finally:
        stop_heartbeat.set()
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        await asyncio.shield(
            asyncio.to_thread(
                finish_job,
                job["id"],
                token,
                status=status,
                output=output,
                error=error,
            )
        )


@contextlib.contextmanager
def _tick_lock():
    from .cron import ensure_dirs

    ensure_dirs()
    handle = (cron_dir() / ".tick.lock").open("a+b")
    acquired = True
    try:
        if os.name != "nt":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                acquired = False
        else:  # pragma: no cover - Windows
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                acquired = False
        yield acquired
    finally:
        if acquired and os.name != "nt":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        elif acquired:  # pragma: no cover - Windows
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


async def tick(
    *,
    delivery: Delivery | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    job_id: str | None = None,
    force: bool = False,
) -> dict[str, int | bool]:
    if concurrency < 1 or concurrency > 16:
        raise CronError("Concurrency must be between 1 and 16")
    with _tick_lock() as acquired:
        if not acquired:
            return {"locked": True, "claimed": 0, "succeeded": 0}
        jobs = await asyncio.to_thread(claim_due, job_id=job_id, force=force)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(job: dict) -> bool:
        async with semaphore:
            return await run_claimed(job, delivery=delivery)

    results = await asyncio.gather(*(one(job) for job in jobs), return_exceptions=True)
    return {
        "locked": False,
        "claimed": len(jobs),
        "succeeded": sum(result is True for result in results),
    }


async def daemon(
    *,
    interval: float = 30.0,
    concurrency: int = DEFAULT_CONCURRENCY,
    stop_event: asyncio.Event | None = None,
    delivery: Delivery | None = None,
) -> None:
    if interval < 1:
        raise CronError("Daemon interval must be at least one second")
    stop = stop_event or asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signum, stop.set)
    try:
        while not stop.is_set():
            (cron_dir() / "daemon-heartbeat").write_text(
                str(datetime_now_timestamp()), encoding="ascii"
            )
            await tick(concurrency=concurrency, delivery=delivery)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signum)


def datetime_now_timestamp() -> float:
    import time

    return time.time()
