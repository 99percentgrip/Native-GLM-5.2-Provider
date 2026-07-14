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
