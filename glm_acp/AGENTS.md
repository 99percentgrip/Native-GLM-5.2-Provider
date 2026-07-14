# glm_acp

ACP agent server for Z.ai GLM models.

## Purpose

Implements the Agent Client Protocol (ACP) server that Zed launches as a
subprocess. Wraps the Z.ai BigModel API directly to provide native reasoning
streaming, 1M context, and auto-continuation for long generations.

## Ownership

- **Entry point**: `__main__.py` → `cli.py:main()` → `agent.py:run()`
- **CLI, terminal auth, and uninstall routing**: `cli.py` → `main()` / `configure_credentials()`
- **Public-install removal**: `uninstall.py` — frozen-copy validation, command/PATH cleanup, credential purge, and guarded Zed JSONC editing
- **Frozen executable entry**: `launcher.py` → absolute import of `cli.main()`
- **ACP protocol**: `agent.py` — implements `acp.Agent` (initialize, new_session, load_session, resume_session, close_session, list_sessions, prompt, set_config_option, set_session_mode)
- **GLM API client**: `glm_client.py` — SSE/tool streaming, preserved thinking, cancellation, retry, cache usage, auto-continuation
- **MCP**: `mcp.py` — official Z.ai remote/local services and configured HTTP/stdio servers
- **Project knowledge**: `memory.py` — root instructions and opt-in project-local memory
- **Tools**: `tools.py` — file/shell operations sandboxed to workspace roots
- **Config**: `config.py` — model registry, API key, constants
- **Persistence**: `session_store.py` — JSON file store for conversation state in `~/.glm-acp/sessions/`

### Entry point resolution

The agent is launched by Zed as `python3 -m glm_acp`. For `-m` to resolve
the module from any cwd (not just this repo's directory), the package MUST
be installed into the venv: `uv pip install -e .` from the repo root. See
the root `AGENTS.md` "Install and distribution (binding)" section.

Public ACP Registry installs launch the frozen `native-glm-acp` executable
instead. The `glm-acp` console script, `python -m glm_acp`, and frozen binary
must all route through `cli.main()`.

## Local Contracts

### Registry authentication

- `initialize()` always advertises `zai-api-key-setup` as Terminal Auth with `args=["--setup"]`.
- `--setup` prompts with hidden input and atomically stores credentials in a user-only file.
- `ZAI_API_KEY` and `Z_AI_API_KEY` override stored credentials.
- API keys must never appear in stdout, stderr, logs, test output, or ACP messages.
- `authenticate()` succeeds only for the advertised method and configured credentials.

### Public-install removal

- `glm-acp --uninstall` operates only from the user-local frozen-binary install directory; source and Registry-managed executables fail safely with guidance.
- Removal deletes both public command aliases and only the exact PATH marker created by the installer, including a profile selected through `GLM_ACP_SHELL_PROFILE`.
- A matching custom `agent_servers.glm-acp` entry is removed from Zed JSONC only after a same-directory command match, with a sibling backup created first; Registry and unrelated entries remain untouched.
- Stored credentials survive normal uninstall. `--uninstall --purge` removes only the credential file and does not delete sessions or other configuration.

### Token stream routing

GLM SSE deltas are split at parse time and never mixed:
- `delta.reasoning_content` → `on_reasoning()` callback → `agent_thought_chunk` session update (thinking view)
- `delta.content` → `on_content()` callback → `agent_message_chunk` session update (code/response)
- `delta.tool_calls` → accumulated, then reported via `tool_call` / `tool_call_update`

Small text and reasoning deltas are coalesced before ACP updates. An HTTP 200
stream that ends without `[DONE]` or a finish reason is incomplete: retry it
only if no user-visible delta was emitted; otherwise preserve the partial
response and report `network_error` without replaying duplicate text.

### Auto-continuation

When `finish_reason == "length"` and no tool calls are pending, the client
auto-sends a bare "continue" message (up to `MAX_AUTO_CONTINUATIONS` = 20
times) so long multi-file refactors don't stall mid-generation. Exhausting the
cap reports `continuation_limit`; it never silently presents a capped response
as complete.

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
Compaction is transactional: invalid, missing, or empty summaries leave the
original history unchanged. Tool call IDs and names remain explicit in the
summary transcript so results cannot be detached from their calls.

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
- `generation_profile` (category: `other`) — Balanced provider defaults, Precise temperature 0.7, or Exploratory top-p 0.98; non-default profiles adjust only one sampling control

### Deep Thinking (GLM-5.2)

GLM-5.2 supports `reasoning_effort` as a top-level API parameter (values:
`"high"`, `"max"`). The `thought_level` config option maps the UI selection to
both `thinking.type` and `reasoning_effort` in the API request body. When the
model is switched away from GLM-5.2, deep levels are hidden and the thought
level falls back to Standard automatically.

Coding Plan Standard and Deep High/Max requests set
`thinking.clear_thinking=false`, and exact returned `reasoning_content` is
retained in assistant history as required for subsequent requests. Disabled
thinking and Standard API standard reasoning clear prior traces.

### Permission system

Destructive tools (`write_file`, `edit_file`, `apply_patch`, `run_command`,
`store_memory`, generic/local MCP execution) are gated by the
session's `permission_mode`:
- **Ask** — read-only tools run freely; destructive tools trigger a
  `session/request_permission` round-trip so Zed can show an approval dialog
- **Read Only** — destructive tools are blocked entirely; the model is told
  why and can adapt
- **Bypass** — all tools auto-approved, no prompts

Permission state is stored per-session and persisted to disk.

### Image / screenshot handling

GLM-5.2, GLM-5-Turbo, and GLM-4.7 are **text-only models** — the Z.ai
Coding Plan API endpoint rejects image content with error 1210/1213.
Direct vision model access (GLM-5V-Turbo) is not included in the Coding
Plan API.

When a user pastes a screenshot from the clipboard:
1. `_extract_prompt_parts()` separates image data from text in the ACP
   prompt blocks
2. `_save_images()` writes each image to `.glm-acp-images/` inside the
   session's workspace root
3. The user message sent to the model includes the saved file paths and a
   note that GLM-5.2 cannot view images directly
4. The user is notified in the panel where the screenshot was saved

This prevents the "prompt parameter was not received normally" (1213) crash
and preserves the screenshot for the built-in permission-gated Z.ai Vision MCP
tool or manual inspection. GLM-5V-Turbo is available for direct multimodal use
on Standard API and BigModel with a 200K context window.

### Sandbox

All file tool operations validate paths against the session `cwd` and
`additional_directories`. Paths outside workspace roots raise `ToolError`.
Text-file tools decode strict UTF-8 with universal-newline normalization and
consistently treat invalid UTF-8 or NUL-containing data as binary on every
supported platform.

Filesystem tools run off the ACP event-loop thread. Search uses `rg` when
available with a portable fallback. Read/search calls in the
same model batch may execute concurrently, while edits and commands remain
ordered. Tool and embedded-resource output is bounded; truncated file reads
include a `start_line` continuation hint. Command output streams live, final
results include the exit code and bounded stdout/stderr, and timeouts terminate
the complete process group. Child commands receive normal runtime variables but
not common inherited API keys, tokens, passwords, secrets, private/access keys,
credentials, or SSH agent access.

Within one model turn, three consecutive identical tool-call batches trigger a
synthetic recovery result. If the model repeats the batch again, the turn stops
instead of consuming the full iteration budget. Malformed JSON tool arguments
are rejected with schema-oriented corrective feedback before permission or
execution.

Command results carry structured exit codes. After a failed or timed-out
command, the model receives one automatic verification-recovery turn before it
may finish, directing it to inspect the failure, preserve tests, correct the
root cause, and rerun the narrowest relevant check or report a genuine blocker.
After a successful file edit, the model likewise receives one recovery turn if
it tries to finish before a successful verification command has been observed.

### MCP and durable memory

Z.ai Search and Reader use authenticated Streamable HTTP MCP. Vision uses the
official optional `@z_ai/mcp-server@latest` stdio server and therefore requires
Node.js 22+; starting it is permission-gated. Custom MCP servers come from the
user-only `mcp.json` and may reference environment variables for headers.
Concurrent MCP discovery initializes each server once. HTTP 404/410 session
expiry performs one clean reinitialize-and-retry, while dead or timed-out stdio
processes discard stale protocol state before restart.

Root `AGENTS.md`, `CLAUDE.md`, and `GLM.md` are loaded into the managed system
prompt. Reusable knowledge is opt-in and stored only after permission in the
workspace's `.glm-acp/memory.md`; secrets and transient reasoning must not be stored.

### Session persistence & history replay

Conversation state (messages, model, mode, title) is persisted to disk
(`~/.glm-acp/sessions/<id>.json`) after every prompt turn and config change.
On `session/load` and `session/resume`, the agent rebuilds the `Session` from
disk via `Session.from_dict`.

Each session also has a small `.meta` sidecar for listing without parsing full
conversation histories. Disk persistence is dispatched off the event loop,
turns are serialized per session, and one pooled GLM HTTP client is reused until
the session's model, endpoint, or thinking configuration changes. Agent
shutdown and session close must close pooled clients.

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
# Full test suite
.venv/bin/python3 -m pytest tests/ -q

# Import check
.venv/bin/python3 -c "from glm_acp.agent import GlmAcpAgent; print('OK')"

# ACP handshake test (no real API key needed)
ZAI_API_KEY=test GLM_ACP_SESSION_PERSISTENCE=0 .venv/bin/python3 -c "
from glm_acp.agent import GlmAcpAgent
import asyncio
class ClientStub:
    async def session_update(self, **kwargs):
        pass
agent = GlmAcpAgent()
agent.on_connect(ClientStub())
r = asyncio.run(agent.initialize(protocol_version=1))
assert r.protocol_version == 1
s = asyncio.run(agent.new_session(cwd='/tmp'))
assert s.config_options[0].category == 'model'
assert s.config_options[1].category == 'thought_level'
asyncio.run(agent.aclose())
print('Handshake OK')
"
```

Tests live in `tests/` (pytest + pytest-asyncio). Run before merging any
change to `glm_client.py`, `agent.py`, `tools.py`, or `session_store.py`.

## Child DOX Index

No children. All modules are flat in this directory.
