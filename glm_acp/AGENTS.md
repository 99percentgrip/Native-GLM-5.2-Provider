# glm_acp

ACP agent server for Z.ai GLM models.

## Purpose

Implements the Agent Client Protocol (ACP) server that Zed launches as a
subprocess. Wraps the Z.ai BigModel API directly to provide native reasoning
streaming, 1M context, and auto-continuation for long generations.

## Ownership

- **Entry point**: `__main__.py` → `agent.py:run()`
- **ACP protocol**: `agent.py` — implements `acp.Agent` (initialize, new_session, load_session, resume_session, close_session, list_sessions, prompt, set_config_option, set_session_mode)
- **GLM API client**: `glm_client.py` — SSE streaming, reasoning/content separation, tool_call assembly, auto-continuation
- **Tools**: `tools.py` — file/shell operations sandboxed to workspace roots
- **Config**: `config.py` — model registry, API key, constants
- **Persistence**: `session_store.py` — JSON file store for conversation state in `~/.glm-acp/sessions/`

## Local Contracts

### Token stream routing

GLM SSE deltas are split at parse time and never mixed:
- `delta.reasoning_content` → `on_reasoning()` callback → `agent_thought_chunk` session update (thinking view)
- `delta.content` → `on_content()` callback → `agent_message_chunk` session update (code/response)
- `delta.tool_calls` → accumulated, then reported via `tool_call` / `tool_call_update`

### Auto-continuation

When `finish_reason == "length"` and no tool calls are pending, the client
auto-sends a bare "continue" message (up to `MAX_AUTO_CONTINUATIONS` = 20
times) so long multi-file refactors don't stall mid-generation.

### Context compaction (Claude Code parity)

When estimated token usage exceeds `COMPACTION_THRESHOLD` (85%) of the model's
context window, `_maybe_compact()` in `agent.py` fires:

1. The system prompt is preserved verbatim.
2. The most recent `COMPACTION_KEEP_RECENT` (4) messages are kept verbatim.
3. Everything else is sent to `GlmClient.summarize_messages()` which makes a
   dedicated non-streaming API call with a structured summarization prompt
   (disabled thinking, `COMPACTION_SUMMARY_MAX_TOKENS` ceiling).
4. The summary is wrapped in `<conversation_summary>` tags and inserted as a
   user message between the system prompt and the preserved recent messages.

This mirrors Claude Code's compaction: summarize the past, keep the present.

### Usage reporting

After each API call, `agent.py` sends a `UsageUpdate` session notification to
the client (Zed) with `size` (context window) and `used` (estimated tokens).
If the API returns `usage.prompt_tokens`, that exact value is used; otherwise
a heuristic estimate (chars ÷ 4) is applied. This drives Zed's context bar.

**Important:** The `UsageUpdate` must include `session_update="usage_update"`
(the ACP discriminant field). Omitting it causes a Pydantic validation error
and crashes the turn.

### Config options

Session config options advertised to the client:
- `model` (category: `model`) — GLM model selector
- `thought_level` (category: `thought_level`) — reasoning depth: Off / Standard (all models); Deep · High / Deep · Max (GLM-5.2 only, maps to `reasoning_effort: high|max`)
- `permission_mode` (category: `permissions`) — tool execution permission: Ask / Read Only / Bypass

### Deep Thinking (GLM-5.2)

GLM-5.2 supports `reasoning_effort` as a top-level API parameter (values:
`"high"`, `"max"`). The `thought_level` config option maps the UI selection to
both `thinking.type` and `reasoning_effort` in the API request body. When the
model is switched away from GLM-5.2, deep levels are hidden and the thought
level falls back to Standard automatically.

### Permission system

Destructive tools (`write_file`, `edit_file`, `run_command`) are gated by the
session's `permission_mode`:
- **Ask** — read-only tools run freely; destructive tools trigger a
  `session/request_permission` round-trip so Zed can show an approval dialog
- **Read Only** — destructive tools are blocked entirely; the model is told
  why and can adapt
- **Bypass** — all tools auto-approved, no prompts

Permission state is stored per-session and persisted to disk.

### Sandbox

All file tool operations validate paths against the session `cwd` and
`additional_directories`. Paths outside workspace roots raise `ToolError`.

### Session persistence & history replay

Conversation state (messages, model, mode, title) is persisted to disk
(`~/.glm-acp/sessions/<id>.json`) after every prompt turn and config change.
On `session/load` and `session/resume`, the agent rebuilds the `Session` from
disk via `Session.from_dict`.

**Critical:** The ACP `LoadSessionResponse` and `ResumeSessionResponse` only
carry `modes`, `config_options`, and `models` — they do **not** include message
history. To make the restored conversation visible in the editor UI, the agent
must replay it back via `session_update` notifications. `_replay_history()`
walks the persisted messages and sends each user turn as a
`user_message_chunk` and each assistant turn as an `agent_message_chunk`.
System messages and tool-result entries are skipped (internal bookkeeping).
The server runs with `use_unstable_protocol=True` to expose
`session/list`, `session/resume`, and `session/close`.

## Work Guidance

- Match existing code style: `from __future__ import annotations`, dataclasses for state, type hints throughout
- Keep `glm_client.py` free of ACP-specific imports — it's a pure API wrapper
- Keep `agent.py` free of HTTP/SSE logic — it's a pure ACP layer
- Never write reasoning text to files; it flows only through `agent_thought_chunk`

## Verification

```bash
# Import check
. .venv/bin/activate
python3 -c "from glm_acp.agent import GlmAcpAgent; print('OK')"

# ACP handshake test (no real API key needed)
ZAI_API_KEY=test python3 -c "
from glm_acp.agent import GlmAcpAgent
import asyncio
agent = GlmAcpAgent()
r = asyncio.run(agent.initialize(protocol_version=1))
assert r.protocol_version == 1
s = asyncio.run(agent.new_session(cwd='/tmp'))
assert s.config_options[0].category == 'model'
assert s.config_options[1].category == 'thought_level'
print('Handshake OK')
"
```

No test framework is set up yet. Add pytest when tests grow beyond smoke checks.

## Child DOX Index

No children. All modules are flat in this directory.
