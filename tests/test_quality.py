"""Offline quality, recovery, benchmark, and process-level ACP checks."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp import PROTOCOL_VERSION, spawn_agent_process

from benchmarks.eval import (
    build_report,
    load_cases,
    persist_report,
    prepare,
    redact,
    run_external,
    verify,
)
from benchmarks.report import case_cell, load_report, row
from benchmarks.run_live import BenchmarkAlreadyRunning, LiveRunLock, process_is_alive
from glm_acp.agent import GlmAcpAgent, Session
from glm_acp.glm_client import StreamResult
from glm_acp.session_store import SessionStore
from glm_acp.tools import ToolResult
from glm_acp.tools import execute_tool as real_execute_tool


class ScriptedClient:
    preserve_thinking = False

    def __init__(self, results: list[StreamResult]) -> None:
        self.results = iter(results)
        self.calls = 0
        self.cancelled = False

    def begin_turn(self) -> None:
        self.cancelled = False

    async def stream_completion(self, **kwargs):
        self.calls += 1
        return next(self.results)


class LifecycleClient:
    async def request_permission(self, **kwargs):
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(self, **kwargs) -> None:
        return None

    async def ext_method(self, method, params):
        return {}

    async def ext_notification(self, method, params) -> None:
        return None


def configured_agent(client: ScriptedClient) -> GlmAcpAgent:
    agent = GlmAcpAgent()
    connection = MagicMock()
    connection.session_update = AsyncMock()
    connection.request_permission = AsyncMock()
    agent._conn = connection
    agent._client_for_session = lambda session: client
    return agent


def tool_result(call_id: str = "call-1") -> StreamResult:
    return StreamResult(
        tool_calls=[
            {
                "id": call_id,
                "function": {"name": "read_file", "arguments": {"path": "sample.txt"}},
            }
        ],
        finish_reason="tool_calls",
    )


@pytest.mark.asyncio
async def test_repeated_tool_batch_recovers_then_stops(tmp_path):
    (tmp_path / "sample.txt").write_text("stable", encoding="utf-8")
    client = ScriptedClient([tool_result(f"call-{index}") for index in range(1, 5)])
    agent = configured_agent(client)
    session = Session("loop", str(tmp_path))
    session.permission_mode = "bypass"

    assert await agent._run_turn(session) == "end_turn"
    assert client.calls == 4
    tool_messages = [item["content"] for item in session.messages if item["role"] == "tool"]
    assert sum("Tool-loop recovery" in item for item in tool_messages) == 2
    rendered_updates = str(agent._conn.session_update.call_args_list)
    assert "Stopped a repeated tool-call loop" in rendered_updates


@pytest.mark.asyncio
async def test_malformed_tool_arguments_receive_actionable_feedback(tmp_path):
    malformed = StreamResult(
        tool_calls=[
            {
                "id": "bad-call",
                "function": {"name": "read_file", "arguments": {"_raw": '{"path":'}},
            }
        ],
        finish_reason="tool_calls",
    )
    client = ScriptedClient([malformed, StreamResult(content="recovered", finish_reason="stop")])
    agent = configured_agent(client)
    session = Session("malformed", str(tmp_path))

    assert await agent._run_turn(session) == "end_turn"
    feedback = [item["content"] for item in session.messages if item["role"] == "tool"]
    assert len(feedback) == 1
    assert "malformed JSON" in feedback[0]


@pytest.mark.asyncio
async def test_failed_command_forces_one_verification_recovery_turn(tmp_path):
    failed_command = StreamResult(
        tool_calls=[
            {
                "id": "failed-check",
                "function": {
                    "name": "run_command",
                    "arguments": {"command": f'"{sys.executable}" -c "raise SystemExit(1)"'},
                },
            }
        ],
        finish_reason="tool_calls",
    )
    client = ScriptedClient(
        [
            failed_command,
            StreamResult(content="Looks complete.", finish_reason="stop"),
            StreamResult(content="Blocked; verification still fails.", finish_reason="stop"),
        ]
    )
    agent = configured_agent(client)
    session = Session("failed-verification", str(tmp_path))
    session.permission_mode = "bypass"

    assert await agent._run_turn(session) == "end_turn"
    assert client.calls == 3
    guard = [
        item["content"]
        for item in session.messages
        if item["role"] == "system" and "Automated verification guard" in item["content"]
    ]
    assert len(guard) == 1
    assert "Do not delete or weaken tests" in guard[0]


@pytest.mark.asyncio
async def test_unrelated_success_does_not_clear_failed_verification(tmp_path):
    def command_result(call_id: str, command: str) -> StreamResult:
        return StreamResult(
            tool_calls=[
                {
                    "id": call_id,
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": command},
                    },
                }
            ],
            finish_reason="tool_calls",
        )

    client = ScriptedClient(
        [
            command_result("failed-check", f'"{sys.executable}" -c "raise SystemExit(1)"'),
            command_result("unrelated-success", f'"{sys.executable}" -c "print(1)"'),
            StreamResult(content="Done.", finish_reason="stop"),
            StreamResult(content="Blocked after review.", finish_reason="stop"),
        ]
    )
    agent = configured_agent(client)
    session = Session("failed-verification-success", str(tmp_path))
    session.permission_mode = "bypass"

    assert await agent._run_turn(session) == "end_turn"
    assert client.calls == 4
    assert any(
        item["role"] == "system" and "Automated verification guard" in item["content"]
        for item in session.messages
    )


def test_verification_classifier_ignores_build_cache_paths():
    command = '"/home/user/.cache/uv/builds-v0/temp/bin/python" -c "print(1)"'
    assert GlmAcpAgent._is_verification_command(command) is False
    assert GlmAcpAgent._is_verification_command("python -m pytest -q") is True
    assert GlmAcpAgent._is_verification_command("npm run typecheck") is True


@pytest.mark.asyncio
async def test_file_change_forces_one_verification_turn(tmp_path):
    changed_file = StreamResult(
        tool_calls=[
            {
                "id": "write-change",
                "function": {
                    "name": "write_file",
                    "arguments": {"path": "client.py", "content": "value = 1\n"},
                },
            }
        ],
        finish_reason="tool_calls",
    )
    client = ScriptedClient(
        [
            changed_file,
            StreamResult(content="Done without testing.", finish_reason="stop"),
            StreamResult(content="Blocked; no test runner is available.", finish_reason="stop"),
        ]
    )
    agent = configured_agent(client)
    session = Session("unverified-change", str(tmp_path))
    session.permission_mode = "bypass"

    assert await agent._run_turn(session) == "end_turn"
    assert client.calls == 3
    guards = [
        item["content"]
        for item in session.messages
        if item["role"] == "system" and "files were changed" in item["content"]
    ]
    assert len(guards) == 1
    assert "successful verification command" in guards[0]


@pytest.mark.asyncio
async def test_successful_verification_clears_unverified_change(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    changed_file = StreamResult(
        tool_calls=[
            {
                "id": "write-change",
                "function": {
                    "name": "write_file",
                    "arguments": {"path": "client.py", "content": "value = 1\n"},
                },
            }
        ],
        finish_reason="tool_calls",
    )
    verification = StreamResult(
        tool_calls=[
            {
                "id": "successful-check",
                "function": {
                    "name": "run_command",
                    "arguments": {"command": f'"{sys.executable}" -m pytest -q test_smoke.py'},
                },
            }
        ],
        finish_reason="tool_calls",
    )
    client = ScriptedClient(
        [
            changed_file,
            verification,
            StreamResult(content="Verified.", finish_reason="stop"),
            StreamResult(content="No reusable lesson.", finish_reason="stop"),
        ]
    )
    agent = configured_agent(client)
    session = Session("verified-change", str(tmp_path))
    session.permission_mode = "bypass"

    async def execute_with_passing_verifier(name, arguments, sandbox, **kwargs):
        if name == "run_command":
            return ToolResult(output="Exit code: 0\n\nStdout:\n1 passed", exit_code=0)
        return await real_execute_tool(name, arguments, sandbox, **kwargs)

    monkeypatch.setattr("glm_acp.agent.execute_tool", execute_with_passing_verifier)

    assert await agent._run_turn(session) == "end_turn"
    assert client.calls == 4
    assert not any(
        item["role"] == "system" and "files were changed" in item["content"]
        for item in session.messages
    )
    assert any(
        item["role"] == "system" and "Learning review" in item["content"]
        for item in session.messages
    )


@pytest.mark.asyncio
async def test_verified_task_can_learn_project_skill(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    def tool_call(call_id: str, name: str, arguments: dict) -> StreamResult:
        return StreamResult(
            tool_calls=[{"id": call_id, "function": {"name": name, "arguments": arguments}}],
            finish_reason="tool_calls",
        )

    client = ScriptedClient(
        [
            tool_call(
                "write-change",
                "write_file",
                {"path": "client.py", "content": "value = 1\n"},
            ),
            tool_call(
                "successful-check",
                "run_command",
                    {"command": f'"{sys.executable}" -m pytest -q test_smoke.py'},
            ),
            StreamResult(content="Verified.", finish_reason="stop"),
            tool_call(
                "learn",
                "learn_skill",
                {
                    "name": "verify-client",
                    "description": "Verify client changes",
                    "instructions": "Run the focused client test after edits.",
                },
            ),
            StreamResult(content="Learned.", finish_reason="stop"),
        ]
    )
    agent = configured_agent(client)
    session = Session("learning", str(tmp_path))
    session.permission_mode = "bypass"

    async def execute_with_passing_verifier(name, arguments, sandbox, **kwargs):
        if name == "run_command":
            return ToolResult(output="Exit code: 0\n\nStdout:\n1 passed", exit_code=0)
        return await real_execute_tool(name, arguments, sandbox, **kwargs)

    monkeypatch.setattr("glm_acp.agent.execute_tool", execute_with_passing_verifier)

    assert await agent._run_turn(session) == "end_turn"
    assert (tmp_path / ".glm-acp/skills/verify-client/SKILL.md").is_file()


@pytest.mark.asyncio
async def test_skill_learning_is_rejected_before_verification(tmp_path):
    attempted_learning = StreamResult(
        tool_calls=[
            {
                "id": "premature-learning",
                "function": {
                    "name": "learn_skill",
                    "arguments": {
                        "name": "guess",
                        "description": "Unverified guess",
                        "instructions": "Assume this works.",
                    },
                },
            }
        ],
        finish_reason="tool_calls",
    )
    client = ScriptedClient(
        [attempted_learning, StreamResult(content="Not learned.", finish_reason="stop")]
    )
    agent = configured_agent(client)
    session = Session("unverified-learning", str(tmp_path))
    session.permission_mode = "bypass"

    assert await agent._run_turn(session) == "end_turn"
    assert not (tmp_path / ".glm-acp/skills/guess/SKILL.md").exists()
    assert any(
        item["role"] == "tool" and "only after a successful verification" in item["content"]
        for item in session.messages
    )


@pytest.mark.asyncio
async def test_agent_can_search_past_sessions_without_model_summarization(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.save(
        "prior-session",
        {
            "cwd": str(tmp_path),
            "title": "Prior cleanup",
            "messages": [{"role": "user", "content": "repair async cleanup regression"}],
        },
    )
    search_call = StreamResult(
        tool_calls=[
            {
                "id": "search-history",
                "function": {
                    "name": "session_search",
                    "arguments": {"query": "async cleanup"},
                },
            }
        ],
        finish_reason="tool_calls",
    )
    client = ScriptedClient(
        [search_call, StreamResult(content="I found the prior cleanup.", finish_reason="stop")]
    )
    agent = configured_agent(client)
    agent._store = store
    session = Session("current-session", str(tmp_path))

    assert await agent._run_turn(session) == "end_turn"
    results = [item["content"] for item in session.messages if item["role"] == "tool"]
    assert any("prior-session" in result and "async cleanup" in result for result in results)


def test_benchmark_catalog_is_broad_and_valid():
    cases = load_cases()
    assert len(cases) >= 10
    assert {"typescript-cache", "go-table-fix", "rust-state-machine"} <= {
        case["id"] for case in cases
    }


def test_multi_file_case_observes_discount_order(tmp_path):
    case = next(item for item in load_cases() if item["id"] == "multi-file-regression")
    prepare(case, tmp_path)
    assert verify(case, tmp_path)["passed"] is False
    (tmp_path / "shop/order.py").write_text(
        "from shop.pricing import discounted\n\n"
        "def total(subtotal: float, discount: float, tax_rate: float) -> float:\n"
        "    return discounted(subtotal, discount) * (1 + tax_rate)\n",
        encoding="utf-8",
    )
    shutil.rmtree(tmp_path / "shop/__pycache__", ignore_errors=True)
    assert verify(case, tmp_path)["passed"] is True


def test_benchmark_fixture_cannot_escape_workspace(tmp_path):
    case = {"files": {"../escape.py": "bad"}}
    with pytest.raises(ValueError, match="escapes workspace"):
        prepare(case, tmp_path / "workspace")


@pytest.mark.asyncio
async def test_external_benchmark_handles_fast_exit(tmp_path):
    result = await run_external(
        [sys.executable, "-c", "raise SystemExit(0)"],
        {"prompt": "ignored", "timeout": 2},
        tmp_path,
    )
    assert result["stop_reason"] == "completed"
    assert result["runner_exit_code"] == 0


def test_benchmark_redaction(monkeypatch):
    monkeypatch.setenv("PRIVATE_API_KEY", "super-secret-value")
    assert redact("failure: super-secret-value") == "failure: [REDACTED]"


def test_native_benchmark_missing_credentials_is_safe(tmp_path):
    env = os.environ.copy()
    env.pop("ZAI_API_KEY", None)
    env.pop("Z_AI_API_KEY", None)
    env["GLM_ACP_CONFIG_DIR"] = str(tmp_path / "empty-config")
    result = subprocess.run(
        [sys.executable, "benchmarks/eval.py", "--case", "python-bugfix"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "configured Z.ai credentials" in result.stderr
    assert "Traceback" not in result.stderr


def test_live_benchmark_check_is_secret_safe(tmp_path):
    env = os.environ.copy()
    env["ZAI_API_KEY"] = "operator-secret-value"
    env["GLM_ACP_CONFIG_DIR"] = str(tmp_path / "config")
    result = subprocess.run(
        [sys.executable, "benchmarks/run_live.py", "--check"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "model=glm-5.2" in output
    assert "operator-secret-value" not in output


def test_live_benchmark_lock_rejects_concurrent_owner(tmp_path):
    lock_path = tmp_path / ".live-benchmark.lock"
    with LiveRunLock(lock_path):
        with pytest.raises(BenchmarkAlreadyRunning, match="already running"):
            with LiveRunLock(lock_path):
                pass
    assert not lock_path.exists()


def test_live_benchmark_lock_recovers_stale_owner(tmp_path):
    lock_path = tmp_path / ".live-benchmark.lock"
    lock_path.mkdir()
    (lock_path / "owner.json").write_text('{"pid": 999999999}\n', encoding="utf-8")
    with LiveRunLock(lock_path):
        assert lock_path.is_dir()
    assert not lock_path.exists()


def test_windows_process_probe_uses_read_only_tasklist(monkeypatch):
    tasklist = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout='"python.exe","123","Console","1","10 K"\n'
        )
    )
    monkeypatch.setattr("benchmarks.run_live.os.name", "nt")
    monkeypatch.setattr("benchmarks.run_live.subprocess.run", tasklist)

    assert process_is_alive(123) is True
    assert process_is_alive(456) is False
    assert tasklist.call_args_list[0].args[0] == [
        "tasklist",
        "/FI",
        "PID eq 123",
        "/FO",
        "CSV",
        "/NH",
    ]


def test_incremental_benchmark_artifacts_are_atomic(tmp_path):
    result = {
        "id": "case",
        "attempt": 1,
        "elapsed_seconds": 1.0,
        "first_delta_seconds": 0.2,
        "input_tokens": 10,
        "output_tokens": 2,
        "verification": {"passed": True, "exit_code": 0, "summary": "ok"},
    }
    report = build_report(
        label="candidate",
        runner="native",
        repeat=3,
        planned_total=3,
        status="running",
        candidate={"model": "glm-5.2"},
        results=[result],
    )
    json_path = tmp_path / "native.json"
    markdown_path = tmp_path / "report.md"
    persist_report(report, json_path, markdown_path)

    loaded = load_report(json_path)
    assert loaded["completed"] == 1
    assert loaded["planned_total"] == 3
    assert "Partial report: 1/3 attempts completed" in markdown_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_million_token_estimate_boundary():
    content = "x" * 3_500_000
    estimate = GlmAcpAgent._estimate_tokens([{"role": "user", "content": content}])
    assert 1_000_000 <= estimate <= 1_000_010


def test_quality_report_row(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "label": "candidate",
                "pass_rate": 1.0,
                "passed": 2,
                "total": 2,
                "skipped": 0,
                "results": [
                    {"elapsed_seconds": 1.0, "input_tokens": 10, "output_tokens": 5},
                    {
                        "elapsed_seconds": 3.0,
                        "first_delta_seconds": 0.5,
                        "input_tokens": 20,
                        "output_tokens": 7,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report = load_report(path)
    assert row(report) == [
        "candidate",
        "100.0%",
        "2/2",
        "0",
        "2.00",
        "0.50",
        "30",
        "12",
    ]
    assert case_cell(report, "missing") == "skipped"


@pytest.mark.asyncio
async def test_stdio_initialize_is_clean_jsonrpc(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "ZAI_API_KEY": "test-key",
            "GLM_ACP_SESSION_PERSISTENCE": "0",
            "HOME": str(tmp_path),
        }
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "glm_acp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert process.stdin is not None and process.stdout is not None
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": 1,
            "clientInfo": {"name": "quality-test", "version": "1"},
            "clientCapabilities": {"terminal": True},
        },
    }
    process.stdin.write((json.dumps(request) + "\n").encode())
    await process.stdin.drain()
    response = json.loads(await asyncio.wait_for(process.stdout.readline(), timeout=5))
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == 1
    assert response["result"]["authMethods"][0]["id"] == "zai-api-key-setup"
    process.terminate()
    await asyncio.wait_for(process.wait(), timeout=5)


@pytest.mark.asyncio
async def test_sdk_process_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setenv("GLM_ACP_SESSION_PERSISTENCE", "0")
    monkeypatch.setenv("HOME", str(tmp_path))

    async with spawn_agent_process(LifecycleClient(), sys.executable, "-m", "glm_acp") as (
        connection,
        _process,
    ):
        initialized = await connection.initialize(protocol_version=PROTOCOL_VERSION)
        assert initialized.protocol_version == PROTOCOL_VERSION
        created = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        assert created.session_id
        assert created.config_options
        await connection.cancel(session_id=created.session_id)
        closed = await connection.close_session(session_id=created.session_id)
        assert closed is not None
