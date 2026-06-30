# GLM-ACP

A native Agent Client Protocol (ACP) server for Z.ai GLM models. Runs as a
subprocess inside Zed's Agent Panel — no Zed recompilation required.

## What this solves

The generic `openai_compatible` provider in Zed doesn't expose GLM's native
features. This ACP agent does:

- **1M context window** — uses the native Z.ai API directly, no context cap
- **Live reasoning traces** — `reasoning_content` streams into Zed's thinking view via `agent_thought_chunk`
- **No mid-generation stalls** — auto-continues on `finish_reason=length` so long refactors don't stop halfway
- **Context compaction** — auto-summarizes older messages when approaching the context window limit (Claude Code–style compaction), so long conversations don't hit context errors
- **Usage reporting** — sends live token usage to Zed's context bar via `UsageUpdate` notifications
- **Model selector** — switch between GLM models in the panel (config option, `model` category)
- **Deep Thinking** — GLM-5.2 supports Deep · High and Deep · Max reasoning levels via `reasoning_effort` (config option, `thought_level` category)
- **Permission modes** — Ask (approve edits/commands before they run), Read Only (block all writes), or Bypass (auto-approve everything) (config option, `permissions` category)
- **API plan switcher** — switch between Coding Plan, Standard API, and BigModel endpoints from the chat dropdown (config option, `api_endpoint` category)
- **Task plans** — the model creates a live todo list for multi-step tasks, visible as a checklist in the panel with pending/in-progress/completed states

## Install

```bash
cd /path/to/glm-acp
uv pip install -e .
```

Get your API key at https://z.ai/

## Configure Zed

Open Zed → Settings → Agent Settings → Add Agent → Add Custom Agent, then
add to `settings.json`:

```json
{
  "agent_servers": {
    "glm-acp": {
      "type": "custom",
      "command": "/path/to/glm-acp/.venv/bin/python3",
      "args": ["-m", "glm_acp"],
      "cwd": "/path/to/glm-acp",
      "env": {
        "ZAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Restart Zed, open the Agent Panel, and select **Z.ai GLM** from the agent
dropdown.

## Architecture

```
glm_acp/
├── __main__.py    # Entry point — launches the ACP agent over stdio
├── agent.py       # ACP agent: initialize, session, prompt turn loop
├── config.py      # Model registry, API key, constants
├── glm_client.py  # Streaming Z.ai API client (SSE, reasoning, tool calls)
└── tools.py       # File/shell tools sandboxed to workspace roots
```

### Token flow

```
Z.ai API ──SSE──> glm_client.py ──callbacks──> agent.py ──session/update──> Zed
  │                    │                                       │
  │ reasoning_content  │ on_reasoning()                        │ agent_thought_chunk
  │ content            │ on_content()                          │ agent_message_chunk
  │ tool_calls         │ on_tool_call_started()                │ tool_call / tool_call_update
  │ usage              │ StreamResult.usage                    │ usage_update (context bar)
  │                    │                                       │
  │                    │ ◄── summarize_messages() ─────────────┘ (compaction)
```

Reasoning never touches file output — it flows exclusively into
`agent_thought_chunk` updates, which Zed renders in the collapsible thinking
view. Code content goes to `agent_message_chunk`. Token usage from the API
is reported back to Zed via `usage_update`.

### Context compaction

When estimated token usage exceeds **85%** of the model's context window:

1. The **system prompt** is preserved verbatim.
2. The **4 most recent messages** are kept verbatim.
3. **All older messages** are sent to a dedicated summarization API call
   (disabled thinking, structured summary prompt).
4. The summary is wrapped in `<conversation_summary>` tags and inserted as
   a user message between the system prompt and the preserved recent messages.

This mirrors Claude Code's compaction strategy: summarize the past, keep
the present.

## Troubleshooting

### Agent crashes with `UsageUpdate` validation error

If you see something like:

```
Error: 1 validation error for UsageUpdate
sessionUpdate
  Field required [type=missing, input_value={'size': 1000000, 'used': 888}, input_type=dict]
```

This means the `UsageUpdate` model is being constructed without the required
`session_update` discriminant field. The ACP protocol requires every session
update to carry a `session_update` field identifying the update type.

**Fix:** Ensure `_report_usage()` in `agent.py` passes
`session_update="usage_update"` when constructing the `UsageUpdate`:

```python
update = UsageUpdate(
    session_update="usage_update",
    size=session.context_size,
    used=used,
)
```

### Agent crashes on startup (API key)

If the agent fails immediately, make sure `ZAI_API_KEY` is set in the agent
server's `env` block in Zed's `settings.json`. Get a key at https://z.ai/.

## Models

| Model | Context | Use case |
|---|---|---|
| GLM-5.2 (Flagship) | 1M | Maximum reasoning, coding, and long-horizon agentic tasks (default) |
| GLM-5-Turbo | 1M | Flagship model optimized for speed — complex tasks with lower latency |
| GLM-4.7 | 1M | Balanced model for daily development and routine tasks |

## API Plans

Switch from the chat dropdown (**API Plan** selector):

| Plan | Endpoint | Notes |
|---|---|---|
| Coding Plan (default) | `api.z.ai/api/coding/paas/v4` | Included with GLM Coding Plan subscription — GLM-5.2, 5-Turbo, 4.7 |
| Standard API | `api.z.ai/api/paas/v4` | Pay-as-you-go — broader model access including vision models |
| BigModel (CN) | `open.bigmodel.cn/api/paas/v4` | Chinese mainland endpoint |

The plan determines which models and features are available. Vision model
support (glm-4.5v, etc.) requires the Standard API or BigModel plan with
sufficient balance.

## Tools

The agent exposes these tools to the model:

- `read_file`, `write_file`, `edit_file` — file operations (sandboxed)
- `list_directory`, `search_files`, `grep` — code exploration
- `run_command` — shell execution for builds, tests, git
- `update_plan` — create/update a task plan (todo checklist) shown in the panel

All file paths are validated against the session's working directory and
additional workspace roots.

## License

Apache-2.0
