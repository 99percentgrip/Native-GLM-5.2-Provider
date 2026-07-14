"""Minimal MCP bridge for Z.ai and user-configured Streamable HTTP servers."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from . import __version__
from .config import config_dir, get_api_key

MCP_CONFIG_ENV = "GLM_ACP_MCP_CONFIG"
MCP_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_SERVERS: dict[str, dict[str, Any]] = {
    "zai_search": {
        "url": "https://api.z.ai/api/mcp/web_search_prime/mcp",
        "auth": "zai",
    },
    "zai_reader": {
        "url": "https://api.z.ai/api/mcp/web_reader/mcp",
        "auth": "zai",
    },
    "zai_vision": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@z_ai/mcp-server@latest"],
        "auth": "zai_vision",
        "builtin_tools": [
            "ui_to_artifact",
            "extract_text_from_screenshot",
            "diagnose_error_screenshot",
            "understand_technical_diagram",
            "analyze_data_visualization",
            "ui_diff_check",
            "image_analysis",
            "video_analysis",
        ],
    },
}

MCP_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the current web using Z.ai Coding Plan Web Search MCP.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_reader",
            "description": "Read and extract a public web page using Z.ai Web Reader MCP.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vision_analyze",
            "description": (
                "Analyze a local image with the official Z.ai Vision MCP. "
                "Requires Node.js 22+ and npx; use a workspace file path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["path", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_list_tools",
            "description": "List tools exposed by a configured MCP server.",
            "parameters": {
                "type": "object",
                "properties": {"server": {"type": "string"}},
                "required": ["server"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_call",
            "description": (
                "Call a configured MCP tool. Built-in servers are zai_search and "
                "zai_reader; custom Streamable HTTP servers can be configured locally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["server", "tool", "arguments"],
            },
        },
    },
]


class McpError(RuntimeError):
    pass


def mcp_config_path() -> Path:
    override = os.environ.get(MCP_CONFIG_ENV)
    return Path(override).expanduser() if override else config_dir() / "mcp.json"


def load_mcp_servers() -> dict[str, dict[str, Any]]:
    """Load custom MCP servers without allowing them to replace Z.ai presets."""
    servers = {name: dict(value) for name, value in DEFAULT_SERVERS.items()}
    try:
        payload = json.loads(mcp_config_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return servers
    configured = payload.get("servers", {}) if isinstance(payload, dict) else {}
    if isinstance(configured, dict):
        for name, value in configured.items():
            if (
                name not in servers
                and isinstance(value, dict)
                and (value.get("url") or value.get("command"))
            ):
                servers[name] = dict(value)
    return servers


@dataclass
class McpResponse:
    result: Any
    session_id: str | None = None


class McpManager:
    """Stateless-by-default MCP JSON-RPC client with session header support."""

    def __init__(self, servers: dict[str, dict[str, Any]] | None = None) -> None:
        self.servers = servers or load_mcp_servers()
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._sessions: dict[str, str] = {}
        self._initialized: set[str] = set()
        self._request_id = 0
        self._discovered: dict[str, list[dict[str, Any]]] = {}
        self._stdio: dict[str, asyncio.subprocess.Process] = {}
        self._stdio_locks: dict[str, asyncio.Lock] = {}
        self._init_locks: dict[str, asyncio.Lock] = {}

    def _reset_protocol_state(self, server: str) -> None:
        self._sessions.pop(server, None)
        self._initialized.discard(server)
        self._discovered.pop(server, None)

    def _server(self, name: str) -> dict[str, Any]:
        server = self.servers.get(name)
        if not server:
            raise McpError(f"Unknown MCP server: {name}")
        return server

    def _client(self, name: str) -> httpx.AsyncClient:
        client = self._clients.get(name)
        if client is not None:
            return client
        server = self._server(name)
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if server.get("auth") == "zai":
            headers["Authorization"] = f"Bearer {get_api_key()}"
        configured_headers = server.get("headers", {})
        if isinstance(configured_headers, dict):
            for key, value in configured_headers.items():
                if isinstance(key, str) and isinstance(value, str):
                    headers[key] = os.path.expandvars(value)
        client = httpx.AsyncClient(base_url=str(server["url"]), headers=headers, timeout=60)
        self._clients[name] = client
        return client

    async def _stdio_process(self, name: str) -> asyncio.subprocess.Process:
        process = self._stdio.get(name)
        if process is not None and process.returncode is None:
            return process
        if process is not None:
            self._stdio.pop(name, None)
            self._reset_protocol_state(name)
        server = self._server(name)
        command = str(server.get("command", ""))
        executable = shutil.which(command)
        if not executable:
            raise McpError(
                f"MCP server {name} requires '{command}' on PATH; "
                "Z.ai Vision MCP requires Node.js 22+ and npx"
            )
        env = os.environ.copy()
        if server.get("auth") == "zai_vision":
            env["Z_AI_API_KEY"] = get_api_key()
            env["Z_AI_MODE"] = "ZAI"
        configured_env = server.get("env", {})
        if isinstance(configured_env, dict):
            for key, value in configured_env.items():
                if isinstance(key, str) and isinstance(value, str):
                    env[key] = os.path.expandvars(value)
        process = await asyncio.create_subprocess_exec(
            executable,
            *(str(arg) for arg in server.get("args", [])),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        self._stdio[name] = process
        self._stdio_locks.setdefault(name, asyncio.Lock())
        return process

    async def _stdio_rpc(self, server: str, method: str, params: dict[str, Any]) -> Any:
        process = await self._stdio_process(server)
        self._request_id += 1
        request_id = self._request_id
        payload = (
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        lock = self._stdio_locks[server]
        async with lock:
            if process.stdin is None or process.stdout is None:
                raise McpError(f"MCP server {server} has no stdio transport")
            process.stdin.write(payload)
            await process.stdin.drain()
            while True:
                try:
                    raw = await asyncio.wait_for(process.stdout.readline(), timeout=60)
                except asyncio.TimeoutError as exc:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
                    self._stdio.pop(server, None)
                    self._reset_protocol_state(server)
                    raise McpError(f"MCP server {server} timed out") from exc
                if not raw:
                    self._stdio.pop(server, None)
                    self._reset_protocol_state(server)
                    raise McpError(f"MCP server {server} closed its connection")
                try:
                    response = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if response.get("id") != request_id:
                    continue
                if response.get("error"):
                    error = response["error"]
                    raise McpError(str(error.get("message", error)))
                return response.get("result")

    @staticmethod
    def _decode_response(response: httpx.Response) -> Any:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            events = []
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    if raw:
                        events.append(json.loads(raw))
            if not events:
                raise McpError("MCP server returned an empty event stream")
            return events[-1]
        try:
            return response.json()
        except ValueError as exc:
            raise McpError("MCP server returned invalid JSON") from exc

    async def _rpc(self, server: str, method: str, params: dict[str, Any]) -> Any:
        if self._server(server).get("type") == "stdio":
            return await self._stdio_rpc(server, method, params)
        self._request_id += 1
        headers = {
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": method,
        }
        if method == "tools/call" and params.get("name"):
            headers["Mcp-Name"] = str(params["name"])
        if server in self._sessions:
            headers["Mcp-Session-Id"] = self._sessions[server]
        response = await self._client(server).post(
            "",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            },
        )
        if response.status_code >= 400:
            raise McpError(f"MCP HTTP error {response.status_code}: {response.text[:300]}")
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._sessions[server] = session_id
        payload = self._decode_response(response)
        if isinstance(payload, dict) and payload.get("error"):
            error = payload["error"]
            raise McpError(str(error.get("message", error)))
        return payload.get("result") if isinstance(payload, dict) else payload

    async def _notify(self, server: str, method: str) -> None:
        if self._server(server).get("type") == "stdio":
            process = await self._stdio_process(server)
            if process.stdin is None:
                raise McpError(f"MCP server {server} has no stdin")
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}).encode() + b"\n")
            await process.stdin.drain()
            return
        headers = {
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": method,
        }
        if server in self._sessions:
            headers["Mcp-Session-Id"] = self._sessions[server]
        response = await self._client(server).post(
            "",
            headers=headers,
            json={"jsonrpc": "2.0", "method": method},
        )
        if response.status_code >= 400:
            raise McpError(f"MCP notification failed with HTTP {response.status_code}")

    async def _ensure_initialized(self, server: str) -> None:
        if server in self._initialized:
            return
        lock = self._init_locks.setdefault(server, asyncio.Lock())
        async with lock:
            if server in self._initialized:
                return
            await self._rpc(
                server,
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "native-glm-acp", "version": __version__},
                },
            )
            await self._notify(server, "notifications/initialized")
            self._initialized.add(server)

    @staticmethod
    def _session_expired(error: McpError) -> bool:
        message = str(error)
        return "MCP HTTP error 404" in message or "MCP HTTP error 410" in message

    async def _rpc_with_reconnect(self, server: str, method: str, params: dict[str, Any]) -> Any:
        try:
            return await self._rpc(server, method, params)
        except McpError as error:
            if self._server(server).get("type") == "stdio" or not self._session_expired(error):
                raise
            self._reset_protocol_state(server)
            await self._ensure_initialized(server)
            return await self._rpc(server, method, params)

    async def list_tools(self, server: str) -> list[dict[str, Any]]:
        await self._ensure_initialized(server)
        result = await self._rpc_with_reconnect(server, "tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        normalized = tools if isinstance(tools, list) else []
        self._discovered[server] = normalized
        return normalized

    async def call(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        await self._ensure_initialized(server)
        discovered = self._discovered.get(server) or await self.list_tools(server)
        available = [str(item.get("name", "")) for item in discovered]
        resolved = tool
        if tool not in available and available:
            keywords = [
                key
                for key in ("search", "reader", "image", "vision", "analysis")
                if key in tool.lower()
            ]
            resolved = next(
                (
                    candidate
                    for key in keywords
                    for candidate in available
                    if key in candidate.lower()
                ),
                tool,
            )
        schema = next((item for item in discovered if item.get("name") == resolved), {})
        properties = (schema.get("inputSchema") or {}).get("properties", {})
        remapped = dict(arguments)
        if "query" in remapped and "query" not in properties and "search_query" in properties:
            remapped["search_query"] = remapped.pop("query")
        return await self._rpc_with_reconnect(
            server, "tools/call", {"name": resolved, "arguments": remapped}
        )

    async def invoke_preset(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a first-party Z.ai MCP capability with a stable ACP schema."""
        if name == "web_search":
            return await self.call(
                "zai_search", "webSearchPrime", {"search_query": arguments["query"]}
            )
        if name == "web_reader":
            return await self.call("zai_reader", "webReader", {"url": arguments["url"]})
        if name == "vision_analyze":
            return await self.call(
                "zai_vision",
                "image_analysis",
                {"image_path": arguments["path"], "prompt": arguments["prompt"]},
            )
        raise McpError(f"Unknown MCP preset: {name}")

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
        for process in self._stdio.values():
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        self._stdio.clear()
        self._sessions.clear()
        self._initialized.clear()
        self._discovered.clear()
        self._stdio_locks.clear()
        self._init_locks.clear()
