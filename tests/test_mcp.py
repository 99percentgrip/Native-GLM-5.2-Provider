import json

import httpx
import pytest

from glm_acp.mcp import (
    McpError,
    McpManager,
    load_mcp_servers,
    remove_mcp_server,
    save_mcp_server,
)


def test_builtin_servers_include_all_zai_capabilities():
    servers = load_mcp_servers()
    assert {"zai_search", "zai_reader", "zai_vision"} <= set(servers)


@pytest.mark.asyncio
async def test_http_mcp_initializes_discovers_remaps_and_calls(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        requests.append((body, request.headers))
        method = body.get("method")
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {}}
        elif method == "tools/list":
            # The actual Z.ai tool name is snake_case ``web_search_prime``
            # (not camelCase ``webSearchPrime`` — Z.ai responds with
            # ``Tool not found: webSearchPrime`` for the wrong name).
            # ``webReader`` (camelCase) is correct on the reader side; the
            # naming asymmetry is real and confirmed against the live API.
            result = {
                "tools": [
                    {
                        "name": "web_search_prime",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"search_query": {"type": "string"}},
                        },
                    }
                ]
            }
        else:
            assert body["params"]["arguments"] == {"search_query": "latest GLM docs"}
            result = {"content": [{"type": "text", "text": "result"}]}
        return httpx.Response(
            200,
            headers={"MCP-Session-Id": "safe-test-session"},
            json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
        )

    manager = McpManager({"test": {"url": "https://mcp.invalid/example", "auth": "zai"}})
    manager._clients["test"] = httpx.AsyncClient(
        base_url="https://mcp.invalid/example", transport=httpx.MockTransport(handler)
    )
    result = await manager.call("test", "web_search", {"query": "latest GLM docs"})
    assert result["content"][0]["text"] == "result"
    assert [body.get("method") for body, _ in requests] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    # glm_acp must call Z.ai's actual snake_case tool name, not camelCase.
    assert requests[-1][1]["Mcp-Name"] == "web_search_prime"
    assert requests[-1][0]["params"]["name"] == "web_search_prime"
    await manager.aclose()


@pytest.mark.asyncio
async def test_concurrent_discovery_initializes_once(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    methods = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        method = body.get("method")
        methods.append(method)
        if method == "notifications/initialized":
            return httpx.Response(202)
        result = (
            {"protocolVersion": "2025-06-18", "capabilities": {}}
            if method == "initialize"
            else {"tools": []}
        )
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": result})

    manager = McpManager({"test": {"url": "https://mcp.invalid/example"}})
    manager._clients["test"] = httpx.AsyncClient(
        base_url="https://mcp.invalid/example", transport=httpx.MockTransport(handler)
    )
    await __import__("asyncio").gather(manager.list_tools("test"), manager.list_tools("test"))
    assert methods.count("initialize") == 1
    assert methods.count("notifications/initialized") == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_expired_http_session_reinitializes_once(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    methods = []
    expired = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal expired
        body = json.loads(request.content or b"{}")
        method = body.get("method")
        methods.append(method)
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
        elif method == "tools/list":
            result = {"tools": [{"name": "lookup", "inputSchema": {"type": "object"}}]}
        elif expired:
            expired = False
            return httpx.Response(410, text="expired session")
        else:
            result = {"content": [{"type": "text", "text": "recovered"}]}
        return httpx.Response(
            200,
            headers={"MCP-Session-Id": f"session-{methods.count('initialize')}"},
            json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
        )

    manager = McpManager({"test": {"url": "https://mcp.invalid/example"}})
    manager._clients["test"] = httpx.AsyncClient(
        base_url="https://mcp.invalid/example", transport=httpx.MockTransport(handler)
    )
    result = await manager.call("test", "lookup", {})
    assert result["content"][0]["text"] == "recovered"
    assert methods.count("initialize") == 2
    assert methods.count("tools/call") == 2
    await manager.aclose()


def test_save_mcp_server_creates_config(monkeypatch, tmp_path):
    """save_mcp_server writes a new server to the config file."""
    config_file = tmp_path / "mcp.json"
    monkeypatch.setenv("GLM_ACP_MCP_CONFIG", str(config_file))

    save_mcp_server("my-server", {"url": "https://example.com/mcp"})

    assert config_file.exists()
    data = json.loads(config_file.read_text())
    assert data["servers"]["my-server"]["url"] == "https://example.com/mcp"


def test_save_mcp_server_protects_builtins(monkeypatch, tmp_path):
    """save_mcp_server rejects overriding built-in Z.ai presets."""
    monkeypatch.setenv("GLM_ACP_MCP_CONFIG", str(tmp_path / "mcp.json"))

    with pytest.raises(McpError, match="Cannot override"):
        save_mcp_server("zai_search", {"url": "https://evil.com"})


def test_remove_mcp_server(monkeypatch, tmp_path):
    """remove_mcp_server deletes a custom server from the config."""
    config_file = tmp_path / "mcp.json"
    monkeypatch.setenv("GLM_ACP_MCP_CONFIG", str(config_file))

    save_mcp_server("temp-server", {"url": "https://example.com"})
    assert remove_mcp_server("temp-server") is True

    data = json.loads(config_file.read_text())
    assert "temp-server" not in data.get("servers", {})


def test_remove_mcp_server_not_found(monkeypatch, tmp_path):
    """remove_mcp_server returns False when the server doesn't exist."""
    monkeypatch.setenv("GLM_ACP_MCP_CONFIG", str(tmp_path / "mcp.json"))
    assert remove_mcp_server("nonexistent") is False


def test_remove_mcp_server_protects_builtins(monkeypatch, tmp_path):
    """remove_mcp_server rejects removing built-in Z.ai presets."""
    monkeypatch.setenv("GLM_ACP_MCP_CONFIG", str(tmp_path / "mcp.json"))

    with pytest.raises(McpError, match="Cannot remove"):
        remove_mcp_server("zai_search")


def test_save_then_load_round_trip(monkeypatch, tmp_path):
    """A saved custom server appears in load_mcp_servers alongside built-ins."""
    monkeypatch.setenv("GLM_ACP_MCP_CONFIG", str(tmp_path / "mcp.json"))

    save_mcp_server("custom", {"url": "https://my.server/mcp"})
    servers = load_mcp_servers()

    assert "custom" in servers
    assert servers["custom"]["url"] == "https://my.server/mcp"
    assert "zai_search" in servers
    assert "zai_reader" in servers
