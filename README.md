# Native GLM ACP

[![CI](https://github.com/99percentgrip/Native-GLM-5.2-Provider/actions/workflows/ci.yml/badge.svg)](https://github.com/99percentgrip/Native-GLM-5.2-Provider/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/99percentgrip/Native-GLM-5.2-Provider)](https://github.com/99percentgrip/Native-GLM-5.2-Provider/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A native Agent Client Protocol (ACP) server for Z.ai GLM models. Runs as a
subprocess inside Zed's Agent Panel — no Zed recompilation required.

## Features

### Core

- **1M context window** — native Z.ai API, no context cap
- **Live reasoning traces** — `reasoning_content` streams into Zed's thinking view
- **No mid-generation stalls** — auto-continues on `finish_reason=length`
- **Usage reporting** — live token usage in Zed's context bar
- **Context compaction** — auto-summarizes older messages at 85% capacity (Claude Code–style)
- **Persistent sessions** — conversations survive Zed restarts, replayed on load
- **Multi-root workspaces** — full support for additional workspace directories

### API Resilience

- **Automatic retry** — exponential backoff (1s → 2s → 4s) on 429/500/502/503/504 and network errors
- **Cost tracking** — cumulative input/output tokens per session, shown in `/status`
- **Real cancellation** — the Cancel button actually aborts in-flight API requests
- **Token estimation** — calibrated 3.5 chars/token heuristic, handles vision content blocks

### Chat Dropdown Config Options

All configurable from the Zed agent panel — no restart needed:

| Option | Values | Description |
|---|---|---|
| **Model** | GLM-5.2, GLM-5-Turbo, GLM-4.7 (+ vision on Standard/BigModel) | Model list syncs to the selected API plan |
| **Reasoning** | Off, Standard, Deep · High, Deep · Max | Deep levels are GLM-5.2 exclusive |
| **API Plan** | Coding Plan, Standard API, BigModel (CN) | Switch endpoints; vision models appear on Standard/BigModel |
| **Permissions** | Ask, Read Only, Bypass | Gate destructive tools (write/edit/run) |

### Slash Commands

Type these in the chat input:

| Command | Description |
|---|---|
| `/compact` | Manually trigger context compaction |
| `/clear-plan` | Clear the current task plan / todo list |
| `/clear-history` | Wipe conversation history (keeps settings) |
| `/diff` | Show git diff of all uncommitted changes |
| `/export` | Export the conversation as a Markdown file |
| `/status` | Show model, plan, context usage, cost, message count |

### Task Plans

For any task with 3+ steps, the model automatically creates a live todo list
visible as a checklist in the panel. Each task shows pending / in-progress /
completed status with priority indicators.

### Project Context

The system prompt auto-detects your project on session creation:

- **Languages:** Python, JavaScript/TypeScript, Rust, Go, Ruby, Java
- **Frameworks:** Next.js, React, Vue
- **Package managers:** uv, Poetry, npm, Yarn, pnpm
- **VCS:** git detection

### Search Quality

`search_files` and `grep` respect `.gitignore` patterns and always skip
`.git`, `node_modules`, `__pycache__`, `.venv`, `dist`, `build`.

## Install

### Release binary

Download the archive for your platform from
[GitHub Releases](https://github.com/99percentgrip/Native-GLM-5.2-Provider/releases),
extract it, then run the one-time terminal setup:

```bash
./native-glm-acp --setup
```

The setup prompts without echoing the API key and stores it in a user-only
configuration file. You can also keep using `ZAI_API_KEY` or `Z_AI_API_KEY`;
environment variables take precedence over stored credentials.

Default credential locations:

- Linux: `~/.config/glm-acp/credentials.json`
- macOS: `~/Library/Application Support/glm-acp/credentials.json`
- Windows: `%APPDATA%\glm-acp\credentials.json`

Set `GLM_ACP_CONFIG_DIR` to override the configuration directory. The key is
never printed or written to logs.

Configure the extracted executable as a custom Zed agent, or install it from
the ACP Registry after the public submission is accepted.

### Development installation

The agent must be **installed into its virtualenv** so the `glm_acp`
Python module resolves regardless of which directory Zed launches the
subprocess from. A bare `git clone` is not enough — Zed sets the
subprocess `cwd` to whatever project you have open, and without an
install `python -m glm_acp` will fail with `ModuleNotFoundError` (exit
code 1) in any repo other than this one.

```bash
cd /path/to/glm-acp
uv pip install -e .
```

> ⚠️ If the agent crashes on startup in other repos with no visible
> error, re-run the install command above — the package is missing from
> the venv's `site-packages`.

Get your API key at https://z.ai/

## Configure Zed

Open Zed → Settings → Agent Settings → Add Agent → Add Custom Agent, then
add to `settings.json`:

```json
{
  "agent_servers": {
    "glm-acp": {
      "type": "custom",
      "command": "/path/to/native-glm-acp",
      "args": []
    }
  }
}
```

Restart Zed, open the Agent Panel, and select **Z.ai GLM** from the agent
dropdown.

If you use the development installation instead, keep the Python command,
`-m glm_acp` argument, working directory, and optional `ZAI_API_KEY` environment
entry from the earlier setup style.

## Models

| Model | Context | Plans | Use case |
|---|---|---|---|
| GLM-5.2 (Flagship) | 1M | All | Maximum reasoning, coding, agentic tasks (default) |
| GLM-5-Turbo | 1M | All | Flagship optimized for speed |
| GLM-4.7 | 1M | All | Balanced daily development |
| GLM-4.5V (Vision) | 128K | Standard, BigModel | Screenshots, diagrams, charts |
| GLM-4.6V (Vision) | 128K | Standard, BigModel | Newer vision model with improved OCR |

## API Plans

| Plan | Endpoint | Notes |
|---|---|---|
| Coding Plan (default) | `api.z.ai/api/coding/paas/v4` | Subscription — text models only |
| Standard API | `api.z.ai/api/paas/v4` | Pay-as-you-go — text + vision models |
| BigModel (CN) | `open.bigmodel.cn/api/paas/v4` | Chinese mainland endpoint |

Vision model support requires Standard API or BigModel with sufficient balance.

## Architecture

```
glm_acp/
├── __main__.py      # Module entry point — routes through cli.main()
├── cli.py           # Console entry point and terminal credential setup
├── launcher.py      # Frozen-executable entry point
├── agent.py         # ACP agent: session lifecycle, prompt loop, slash commands
├── config.py        # Model registry, API endpoints, constants
├── glm_client.py    # Streaming Z.ai API client (SSE, retry, reasoning, tools)
├── session_store.py # Persistent JSON session storage (~/.glm-acp/sessions/)
└── tools.py         # File/shell/search tools sandboxed to workspace roots
```

### Token flow

```
Z.ai API ──SSE──> glm_client.py ──callbacks──> agent.py ──session_update──> Zed
  │                    │                                       │
  │ reasoning_content  │ on_reasoning()                        │ agent_thought_chunk
  │ content            │ on_content()                          │ agent_message_chunk
  │ tool_calls         │ on_tool_call_started()                │ tool_call / tool_call_update
  │ usage              │ StreamResult.usage                    │ usage_update (context bar)
  │                    │                                       │
  │                    │ ◄── summarize_messages() ─────────────┘ (compaction)
```

### Context compaction

When token usage exceeds **85%** of the context window:

1. System prompt preserved verbatim
2. 4 most recent messages kept (boundary adjusted to never split tool-call pairs)
3. Older messages summarized via a dedicated API call
4. Summary wrapped in `<conversation_summary>` tags

### Session persistence

- State saved to `~/.glm-acp/sessions/<session-id>.json` after every turn
- On restart, `load_session` / `resume_session` replays history + plan + config
- Sessions listed in Zed's history sidebar via `session/list`
- Fork support: duplicate a session to experiment with different approaches

## Tools

| Tool | Description |
|---|---|
| `read_file` | Read file contents (with optional line range) |
| `write_file` | Create or overwrite a file |
| `edit_file` | Find-and-replace a unique text block |
| `list_directory` | List directory entries |
| `search_files` | Glob pattern search (`.gitignore`-aware) |
| `grep` | Regex content search (`.gitignore`-aware) |
| `run_command` | Shell execution for builds, tests, git |
| `update_plan` | Create/update the task plan checklist |

All file paths are validated against workspace roots. `update_plan` is
available in both Ask and Code modes.

## Testing

```bash
cd /path/to/glm-acp
.venv/bin/python3 -m pytest tests/ -v
```

Release verification also builds the wheel, source distribution, and frozen
PyInstaller executable. GitHub Actions runs the test suite and frozen-binary
smoke test on Linux, macOS x86-64, and Windows x86-64.

## Troubleshooting

### Agent crashes on startup in other repos (exit 1, no error)

The most common cause: **the `glm_acp` package is not installed in the
virtualenv.** Python only finds it when run from this repo's directory
(the cwd is on `sys.path`), so opening any other project in Zed makes
`python -m glm_acp` exit with `ModuleNotFoundError`. Fix:

```bash
cd /path/to/glm-acp
uv pip install -e .
```

You can confirm it's installed by checking for the editable finder:

```bash
ls .venv/lib/*/site-packages/ | grep glm_acp
# expect: glm_acp-0.2.0.dist-info  (and editable-install metadata)
```

### Agent reports missing API credentials

Run `native-glm-acp --setup` (or `glm-acp --setup` for a Python install), or
set `ZAI_API_KEY` in the agent server's `env` block. Get a key at https://z.ai/.

### Vision models return content filter errors

The Coding Plan endpoint applies a strict content filter on image inputs.
Switch to **Standard API** in the API Plan dropdown and ensure you have
balance on your Z.ai account.

### Rate limit errors (429)

The agent automatically retries with exponential backoff (3 attempts). If
errors persist, the model will receive the error and can inform the user.

## License

Apache-2.0. Copyright 2025 Aleksejs Kozlitins.

## Author

Created and maintained by **Aleksejs Kozlitins**.
