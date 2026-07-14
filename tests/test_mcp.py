import json

import httpx
import pytest

from glm_acp.mcp import McpManager, load_mcp_servers


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
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "webSearchPrime",
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
    assert requests[-1][1]["Mcp-Name"] == "webSearchPrime"
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
