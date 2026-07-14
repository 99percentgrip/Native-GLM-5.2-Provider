"""Offline quality, recovery, benchmark, and process-level ACP checks."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp import PROTOCOL_VERSION, spawn_agent_process

from benchmarks.eval import load_cases, prepare, redact
from benchmarks.report import case_cell, load_report, row
from glm_acp.agent import GlmAcpAgent, Session
from glm_acp.glm_client import StreamResult


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


def test_benchmark_catalog_is_broad_and_valid():
    cases = load_cases()
    assert len(cases) >= 10
    assert {"typescript-cache", "go-table-fix", "rust-state-machine"} <= {
        case["id"] for case in cases
    }


def test_benchmark_fixture_cannot_escape_workspace(tmp_path):
    case = {"files": {"../escape.py": "bad"}}
    with pytest.raises(ValueError, match="escapes workspace"):
        prepare(case, tmp_path / "workspace")


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
                    {"elapsed_seconds": 3.0, "input_tokens": 20, "output_tokens": 7},
                ],
            }
        ),
        encoding="utf-8",
    )
    report = load_report(path)
    assert row(report) == ["candidate", "100.0%", "2/2", "0", "2.00", "30", "12"]
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
