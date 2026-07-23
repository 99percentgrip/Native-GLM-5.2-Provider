# glm_acp

Open-source ACP-native coding agent runtime for Z.ai GLM models.

## Purpose

Implements the Agent Client Protocol (ACP) server that Zed launches as a
subprocess. Wraps the Z.ai BigModel API directly to provide native reasoning
streaming, 1M context, and auto-continuation for long generations.

## Ownership

- **Entry point**: `__main__.py` → `cli.py:main()` → `agent.py:run()`
- **CLI, terminal auth, and uninstall routing**: `cli.py` → `main()` / `configure_credentials()`
- **Standalone terminal frontends**: `terminal_cli.py` owns line/JSON routing and `tui.py` owns the cross-platform Textual conversation/reasoning/tool/plan/status panels, approval/settings modals, and key bindings; both must use the same `GlmAcpAgent` session/update and permission interfaces as ACP editors without forking the harness loop
- **Public-install removal**: `uninstall.py` — frozen-copy validation, command/PATH cleanup, credential purge, and guarded Zed JSONC editing
- **Frozen executable entry**: `launcher.py` → absolute import of `cli.main()`
- **ACP protocol**: `agent.py` — implements `acp.Agent` (initialize, new_session, load_session, resume_session, close_session, list_sessions, prompt, set_config_option, set_session_mode)
- **GLM API client**: `glm_client.py` — SSE/tool streaming, preserved thinking, cancellation, retry, cache usage, auto-continuation
- **MCP**: `mcp.py` — official Z.ai remote/local services and configured HTTP/stdio servers
- **Project knowledge and learning**: `memory.py` — scoped memory, relevant skills/bundles, telemetry, curation, and evaluated candidates
- **Project discovery**: `project_context.py` — repository roots, progressive instruction files, manifests, package managers, git state, and canonical verification commands
- **Verification evidence**: `verification.py` — persistent edit generations and bounded canonical command outcomes
- **Awareness and completion evidence**: `awareness.py` — bounded typed epistemic records, harness-issued evidence references, scope-aware invalidation, and completion certificates
- **Metacognitive control**: `metacognition.py` — typed uncertainty classification, deterministic adaptive execution modes, and telemetry-derived aggregate capability profiles
- **Grounded deliberation**: `deliberation.py` — evidence-only criticism, falsifiable hypothesis testing, and deterministic value-of-information action ranking
- **Repository intelligence**: `repository_intelligence.py` — bounded lazy dependency/test/ownership world slices, pre-edit impact prediction, observed comparison, and risk-triggered pre-mortems
- **Safe metacognitive learning**: `meta_learning.py` — typed causal attribution, inert strategy drafts, and explicit fresh/time-split/mutated evaluation-gated promotion
- **Post-write diagnostics**: `diagnostics.py` — deterministic syntax checks and lazy optional LSP clients
- **Lifecycle extensions**: `hooks.py` — user-owned, hash-pinned, workspace-scoped lifecycle commands
- **Trajectory evidence**: `telemetry.py` and `observability.py` — bounded metadata-only events plus local aggregate quality, latency, cache, tool, and safety reporting
- **Failure-driven evaluation**: `failure_corpus.py` — metadata-only drafts and permission-gated runnable project regression cases
- **Workspace checkpoints**: `checkpoints.py` — default-off opt-in auto-checkpointing, shared compressed Git-compatible blob objects, automatic bounded retention/GC, large/secret exclusion, verified legacy migration, exact agent hashes, and conflict-aware rollback
- **Explicit references**: `references.py` — bounded workspace-contained references with language/task/change-aware ranking
- **Declarative controls**: `policy.py` and `workflows.py` — ordered allow/ask/deny rules and static dependency graphs
- **OS command isolation**: `os_sandbox.py` — Linux Bubblewrap, capability-detected macOS Seatbelt, Windows Job Object containment, and required-mode fail closure
- **Isolated extension state**: `profiles.py` and `plugins.py` — validated profile paths plus hash-pinned, Ed25519-verifiable data-only packages
- **Implementation workers**: `worktrees.py` — detached creation, digest inspection, conflict-aware transactional promotion/rollback, and reviewed cleanup
- **Offline hardening**: `resilience.py` — deterministic parser fuzzing, malformed telemetry, and real promotion rollback fault injection
- **Loop guardrails**: `guardrails.py` — repeated failure and unchanged read-only result detection
- **Untrusted-context defense**: `security.py` — promptware findings, stored-context blocking, and tool/MCP/recall delimiters
- **Tools**: `tools.py` — file/shell operations sandboxed to workspace roots
- **Config**: `config.py` — model registry, API key, constants
- **Persistence and recall**: `session_store.py` — JSON conversation state plus a local redacted SQLite FTS5 search index
- **Scheduled automation**: `cron.py`, `cron_scheduler.py`, and `cron_cli.py` — persistent jobs, isolated execution, claims, artifacts, and CLI lifecycle

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
3. Decisions, fixes, unresolved work, plan state, edited paths, command outcomes,
   and session lineage are extracted deterministically before older messages are
   discarded. Verified decisions/fixes become inspectable, permission-required
   memory proposals rather than automatic writes.
4. Everything else is sent to `GlmClient.summarize_messages()` which makes a
   dedicated non-streaming API call with a structured summarization prompt
   (disabled thinking, `COMPACTION_SUMMARY_MAX_TOKENS` ceiling). The optional
   auxiliary model performs this call when its context can hold the source;
   otherwise compaction falls back to the main model. `/compact <focus>` adds
   bounded guidance.
5. The summary is wrapped in `<conversation_summary>` tags and inserted as a
   user message between the system prompt and the preserved recent messages.

This mirrors Claude Code's compaction: summarize the past, keep the present.
Compaction is transactional: invalid, missing, or empty summaries leave the
original history unchanged. Tool call IDs and names remain explicit in the
summary transcript so results cannot be detached from their calls.
Context-pressure messages fire once at 60%, 75%, and 85% until compaction
reduces the tier. Completion reports the exact retained categories and a
persisted deterministic summary-quality score; a 15-point decline from the
previous compaction produces a warning.

### Promptware defense

- Root instructions, project/user memory, and learned skills are scanned before
  prompt injection; suspicious stored sections are blocked.
- File, tool, MCP, embedded-resource, delegated-worker, session-recall output,
  and context passed into a delegate are wrapped in `untrusted_context`
  delimiters before the receiving model sees them.
- Findings are defense in depth and never replace sandboxing, secret removal,
  destructive-tool permissions, or operator review.
- Learning rejects credential-shaped and prompt-injection content before writing.

### Epistemic awareness and completion

- `EpistemicLedger` persists at most 100 typed records and 200 metadata-only evidence events; it never stores tool bodies, prompts, reasoning, commands, credentials, or raw external content.
- Model records may cite only harness-issued evidence IDs and valid current goal/criterion IDs. Secret-shaped or promptware-shaped summaries are rejected.
- Reads, searches, edits, diagnostics, verification, and the current user request create concise evidence events. Relevant later edits make scope-overlapping support stale; user-request evidence is edit-independent.
- `/awareness` renders active observations, assumptions, hypotheses, contradictions, unknowns, capability limits, stale state, fresh evidence IDs, completion coverage, and the next evidence need.
- Persistent goals reach the bounded auxiliary judge only when every goal/criterion has a fresh evidence-backed observation, no contradiction remains active, and changed files have fresh verification.
- Ordinary turns emit informational completion-certificate metrics without adding a new hard completion loop.

### Metacognitive controller

- The controller separately classifies ambiguity, knowledge gaps, diagnostic uncertainty, capability limits, verification gaps, and permission uncertainty from task-family heuristics, the epistemic ledger, permission mode, and verification freshness.
- It selects one bounded posture: `direct` for simple low-risk work, `grounded` for minimum-evidence work, `deliberate` for competing diagnoses, or `high-assurance` for operations, high risk, and unverified edits.
- Mode guidance is managed prompt metadata only. It cannot alter provider reasoning settings, expand permissions, bypass policy/sandbox/plugin trust, invoke delegates, or write trusted rules.
- Metadata-only `capability_outcome` events aggregate success, failure, token cost, latency, and targeted/full verification by a fixed task family and coarse ecosystem/VCS/session-mode/OS label. They store no task text, paths, prompts, commands, outputs, or identities and disappear when telemetry is disabled.
- Three or more weak matching outcomes may raise the selected posture by one level. Empirical history never lowers the deterministic baseline, keeping small direct tasks from accumulating reflection overhead.
- `/metacognition` exposes current classes, risk, mode, and matching aggregate profile; `/status` and `/observability` expose bounded mode/outcome summaries.

### Grounded deliberation

- Deliberate/high-assurance diagnosis uses exactly two or three distinct hypothesis records. Each contains a concise statement, observable prediction, observable falsifier, and `untested`, `supported`, `refuted`, or `inconclusive` status; tested states require fresh non-user harness evidence IDs.
- Hypotheses come from one thinking-disabled auxiliary call over the objective and fresh evidence only, with a bounded deterministic fallback. Objective changes clear them; stale evidence resets affected tests.
- The independent critic runs at most twice per turn and receives no conversation history, assistant answer, or `reasoning_content`. Its packet is limited to the objective/criteria, bounded credential-redacted Git diff, fresh harness evidence, hypothesis results, and completion certificate.
- Structural verification guards run before the critic. An auxiliary approval must cite fresh evidence; relevant edits invalidate the verdict. Critic output is advisory loop guidance and never authorizes tools, workers, policy changes, or completion by itself.
- Value-of-information ranking scores only available/virtual actions by expected information gain, reliability, and cost. Direct tasks receive no ranking; Read Only never recommends command execution; permissions and policy still evaluate the selected action normally.
- `/deliberation` exposes hypothesis/test state, ranked actions, and the latest critic verdict. Telemetry records only counts, enums, tool names, scores, and match outcomes—never objectives, hypotheses, diffs, evidence bodies, paths, or reasoning.

### Repository intelligence

- Non-trivial tasks build a lazy bounded slice of at most 96 path nodes and 192 relationships from current targets/changes, manifests, progressive instructions, nearby tests, statically resolved imports, semantic-tool paths, CODEOWNERS, and project-matched failure classes.
- Repository intelligence never snapshots the checkout or persists source bodies. Repeated prompt refreshes reuse the unchanged slice; direct small tasks receive no world-model or pre-mortem context.
- Impact predictions are updated while evidence gathering remains in the pre-edit generation, then freeze across the first mutation. Observed paths and canonical checks are compared afterward, and unexpected impact remains inspectable through `/repository`.
- Counterfactual pre-mortems are deterministic, limited to five observable failure/detection pairs, and appear only for risk score 6 or higher. Predictions and pre-mortems are advisory and never establish coverage, expand permissions, or replace fresh verification.

### Safe metacognitive learning

- Causal attribution stores only fixed cause/intervention enums, tool class, harness evidence IDs, correction state, and edit generation. It stores no task text, output body, command, path, identity, credential, or private reasoning.
- Corrected failures may draft only allowlisted strategies for clarification, browsing, LSP navigation, hypothesis branching, verification, or safe stopping. Drafts are inert and cannot alter execution.
- Promotion requires two distinct supporting causal attributions, an explicit command, and a compatible passing baseline/candidate gate. The gate requires gains overall and on both fresh time-split and transformed cases, zero per-case/safety regression, improved false-completion and unsupported-claim counts when present, and no material calibration, clarification, evidence-freshness, contradiction, repetition, latency, token, or small-task-overthinking regression.
- `glm-acp meta-cases`, `glm-acp meta-eval`, `/meta-learning`, and `/repository` are inspectable surfaces. Promoted strategies remain advisory and cannot change trusted policy, permissions, sandboxing, plugin trust, or delegation authority.

### Auxiliary routing and delegation

- `auxiliary_model=main` uses the primary model; another non-vision GLM model on
  the active API plan handles titles, compression, recall ranking, advisory skill
  evaluation, and delegated analysis, with local/deterministic fallback where
  possible.
- `delegate_task` is permission-gated and limited to three workers per parent
  turn. All workers share 24 tool calls, 120K input tokens, and 16K output tokens.
  Each worker has six read/search iterations and 180 seconds; it cannot edit,
  execute, call MCP, access credentials, or delegate recursively, so depth is one.
- Auxiliary usage contributes to parent totals. Compression clients are pooled;
  delegated clients are transient and all clients close with the session.

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
- `auxiliary_model` (category: `other`) — main model or a non-vision GLM model on the active plan for titles, compaction, recall ranking, skill evaluation, and bounded delegation
- `mixture_mode` (category: `other`) — off by default; Reference review runs up to two independent non-vision GLM advisers once per user turn and leaves the primary model as aggregator/actor

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

### Scheduled automation

- `cronjob` is one stable permission-gated tool for create/list/update/pause/resume/run/remove; scheduled sessions cannot call it recursively.
- Jobs persist in the user configuration directory with user-only permissions and accept relative one-shots, intervals, strict five-field cron, or aware ISO timestamps.
- Every dispatch is atomically claimed across processes. Healthy long runs renew their claim, missed recurring slots collapse to one future slot, and stale claims remain recoverable.
- Runs use fresh non-persisted sessions, project context, optional learned skills/bundles, a 600-second inactivity watchdog by default, and bounded redacted result artifacts. `[SILENT]` suppresses live delivery but not history.
- Script prechecks are workspace-contained, non-symlink files with scrubbed environments, bounded output, and a 60-second timeout. Script-only jobs make no model request.
- The ACP process starts a background ticker and `glm-acp cron daemon` provides a dedicated foreground scheduler. Multiple tickers are safe; running jobs may be paused for future runs but cannot be updated or removed until completion.

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

Result-aware loop detection also tracks identical-argument failures, same-tool
failure streaks with changed arguments, and unchanged read/search output hashes.
It emits corrective warnings before its bounded stop thresholds. Repeated
unchanged read/search results are replaced in model context by a compact digest.

Project facts detect repository manifests, package managers, git state, and
canonical checks. Command results carry structured exit codes into a persisted
verification ledger. Only structurally matched checks count; output claims,
non-executing flags, and shell constructs that can mask verifier status do not.
Every successful edit advances the edit generation, invalidating older passes.
After a failed or timed-out
command, the model receives one automatic verification-recovery turn before it
may finish, directing it to inspect the failure, preserve tests, correct the
root cause, and rerun the narrowest relevant check or report a genuine blocker.
After a successful file edit, the model likewise receives one recovery turn if
it tries to finish before a successful verification command has been observed.

Each successful write is read back before the model continues. Python, JSON,
and TOML receive deterministic syntax validation. If already installed,
`pyright-langserver`, `typescript-language-server`, `gopls`, or `rust-analyzer`
also receives the versioned document over LSP stdio and returns semantic
diagnostics. Missing, timed-out, or failed servers fall back safely and are never
installed by the agent.

Installed language servers also back the read-only `semantic_code` tool for
document/workspace symbols, definitions, references, hover, implementations,
rename preparation, and call hierarchy. Tool positions are 1-based and converted
to LSP 0-based positions; missing or failed servers return bounded status.

`apply_patch_set` validates up to 20 existing UTF-8 files, exact pre-edit SHA-256
hashes, every unified-diff hunk, and deterministic Python/JSON/TOML syntax before
writing. It commits all candidates or restores previously written files after a
commit failure. Every changed file advances verification evidence and receives
post-write diagnostics. `batch_read` runs up to 20 read/list/search operations
concurrently with per-result and aggregate output bounds.

The managed system message keeps volatile project, knowledge, and task context
after a byte-stable prefix marker. Status exposes the stable-prefix hash and
provider cache-hit ratio; cache layout changes must preserve scoped instructions
and compaction/session compatibility.

Trajectory telemetry stores no prompts, content, tool arguments, commands,
outputs, raw paths, reasoning, credentials, or raw session IDs and is disabled
by `GLM_ACP_TELEMETRY=0`. `/observability` and `glm-acp observe` read at most
50,000 events/20 MiB, ignore corrupt records, and report aggregates only.
Lifecycle hooks require an exact executable SHA-256,
optional exact workspace scope, argv execution without a shell,
credential-scrubbed environments, a 10-second maximum timeout, fail-open
isolation, and a three-nudge pre-verification cap.

`browser_ui` is a stable permission-gated adapter over isolated headless
Playwright MCP. It allowlists navigation, accessibility snapshots, console/network
evidence, screenshots, ordinary interaction, waits, and close; arbitrary browser
JavaScript evaluation is absent and the MCP child receives no inherited credentials.

Auto-checkpoint is **OFF by default**. The agent only snapshots the workspace
before a mutating user turn when the operator opts in via `/checkpoint auto on`
or sets `GLM_ACP_AUTO_CHECKPOINT=1`; this prevents large workspaces (for
example ones containing big `*.sqlite`, `node_modules/`, or `.git/` trees) from
filling the disk with checkpoint data on every edit. The toggle persists in
`config_dir()/checkpoint-auto.json` (schema 1, profile-scoped); `auto` shows
the current state and source, `auto on/off` writes the value atomically, and
`auto reset` clears the override back to the default OFF. Manual `/checkpoint`
remains available regardless of the toggle. When enabled, snapshots default to
20,000 files/250 MiB, exclude common credential, SSH, private-key, and `.env`
paths, exclude individual files larger than 25 MiB by default, and record current
hashes after each successful mutation. File bodies are compressed loose Git blob
objects in a private shadow store; identical content deduplicates across projects
without reading or changing workspace Git metadata. Small schema-2 manifests
reference objects and preserve executable modes.
`/checkpoint limits <files> <MiB>` atomically persists profile-isolated limits;
`limits` displays their source and `limits reset` restores defaults.
`GLM_ACP_CHECKPOINT_MAX_FILES` and `GLM_ACP_CHECKPOINT_MAX_MIB` take
precedence. Values remain hard-bounded to 1,000,000 files/10,240 MiB and invalid
configuration fails closed. `/rollback` restores only recorded paths whose current
hash still equals the exact agent-produced hash; any later conflict aborts the
entire rollback before writes.

Checkpoint creation automatically retains ten manifests per project for 30 days,
enforces a 1,024 MiB global object ceiling, and garbage-collects unreferenced
objects. `/checkpoint storage` exposes and configures those bounds plus maximum
file size; `/checkpoint prune` applies them immediately. Schema-1 full-copy
snapshots remain readable. `/checkpoint migrate-legacy` verifies every old file
hash, writes and rereads the schema-2 manifest, and only then removes the verified
legacy directory. Clear operations require an explicit `confirm` argument.

Explicit `@file:`, `@folder:`, `@symbol:`, and `@diff` prompt references are
limited to 12 references and 48,000 aggregate characters. Paths remain inside
workspace roots, common secret files are rejected/omitted, folder expansion is
file-bounded, and all expanded content is untrusted model data. Folder/symbol
budgets rank language-specific definitions and references, task terms, tests,
manifests, and current Git changes before deterministic path tie-breaking.

`.glm-acp/policy.json` version 1 supplies at most 100 ordered allow/ask/deny
rules over tool globs, path globs, and bounded safe command regular expressions.
Invalid policy fails closed, policy `allow` never bypasses the session permission
mode, Read Only cannot be overridden, and every nested workflow step is evaluated.
`run_workflow` accepts a static acyclic graph of at
most 12 existing allowlisted tools, executes in dependency order, and stops on
the first command failure; it cannot generate steps or run orchestration code.

Main-session OS command isolation is opt-in through `GLM_ACP_OS_SANDBOX`.
Bubblewrap mounts Linux runtime paths read-only, declared workspace roots
writable, a private temporary home, and optionally an unshared network. macOS
uses `sandbox-exec` only when present with a deny-default Seatbelt profile and
labels the deprecated backend explicitly. Windows `auto` mode uses a Job Object
to contain/terminate the child tree; because that is not filesystem or network
isolation, required mode rejects it. Worktree command tools always require real
filesystem/network isolation and fail closed otherwise.

Worker inspection hashes the complete bounded binary patch and reports affected
paths. Promotion requires that exact digest plus a fresh verifier run inside the
network-disabled worker sandbox. `git apply --check` rejects primary conflicts
before writes; ordinary apply never uses partial-reject mode, post-apply faults
reverse the whole patch, and the worker remains preserved until an explicit
digest-pinned discard.

`GLM_ACP_PROFILE` accepts a validated 1-32 character identifier. `default`
preserves legacy paths; named profiles isolate credentials, sessions, telemetry,
hooks, MCP configuration, cron state, plugins, checkpoints, worktrees, and
private user memory. Plugin packages require schema 1, explicit permission
scopes, exact SHA-256 for every file plus an installed manifest pin, a 2 MiB/32-file
bound, and data-only file types; executable package content is rejected and prompt
fragments are scanned. Optional Ed25519 signatures cover canonical manifests,
must match an explicitly trusted publisher key, and may be made mandatory with
`GLM_ACP_REQUIRE_SIGNED_PLUGINS=1`. Private-key creation/signing is CLI-only.

Failed tool traces create deduplicated private drafts containing only hashed
project identity, tool name, normalized failure class, and file extensions.
They never enter model context automatically. The permission-gated
`failure_corpus` tool requires a reviewed prompt, bounded non-secret fixtures,
argv verifier, and timeout before writing a runnable project-local case.
`glm-acp harden` is credential-free and bounded to 10-5,000 deterministic fuzz
iterations plus a real Git post-apply rollback fault.

Persistent `/goal` state and `/subgoal` acceptance criteria survive session
reloads and forks. A strict-JSON auxiliary judge evaluates the response, changed
paths, and fresh verification evidence after each candidate completion. It may
continue the same tool loop, pause on completion or a genuine blocker, and
automatically pauses after 20 judged turns; explicit resume resets that budget.

### MCP and durable memory

Z.ai Search and Reader use authenticated Streamable HTTP MCP. Vision uses the
official optional `@z_ai/mcp-server@latest` stdio server and therefore requires
Node.js 22+; starting it is permission-gated. Custom MCP servers come from the
user-only `mcp.json` and may reference environment variables for headers.
Concurrent MCP discovery initializes each server once. HTTP 404/410 session
expiry performs one clean reinitialize-and-retry, while dead or timed-out stdio
processes discard stale protocol state before restart.

`.hermes.md`, `HERMES.md`, `AGENTS.md`, `CLAUDE.md`, `GLM.md`, `.cursorrules`,
and `.cursor/rules/*.mdc` files are discovered progressively from the
project root toward accessed paths. When a direct mutation first reveals a
closer scoped instruction file, that call is deferred without changing the file
so the acting model can retry with the updated managed prompt. Reusable facts are opt-in and stored only after permission in the
workspace's `.glm-acp/memory.md`; secrets and transient reasoning must not be stored.

After a task passes a recognized verification command, the turn receives
one bounded learning review. Non-obvious reusable procedures may be written only
after permission to `.glm-acp/skills/<name>/SKILL.md`. Skill metadata is loaded
into the managed prompt, full instructions are read on demand, credential-like
content is rejected, and deletion is limited to this agent-owned directory.

Explicit user facts and preferences may be stored after permission in private
cross-project `user.md`; project facts remain in `.glm-acp/memory.md`. Both are
bounded, secret-filtered, inspectable, and support exact forgetting.

Learned-skill telemetry tracks views, uses, and revisions. Deterministic curation
marks 30-day idle skills stale and reversibly archives 90-day idle skills only
after permission. Pinned skills are protected by code and curation never deletes.
Content hashes route manually changed skills to review, and deterministic
description overlap evidence may suggest consolidation but never auto-merges.

Optional `platforms`, `environments`, `requires_tools`, and semantic `tasks`
frontmatter gates keep irrelevant learned skills out of the managed prompt.
Task-gated metadata is re-evaluated against every current user request.
Project-local bundles group up to 12 existing relevant skills without copying
their bodies and load them progressively through `read_skill_bundle`; stored
bundle instructions are re-scanned when read.

Skill evolution is candidate-based. Failed benchmark traces can create a bounded
non-promotable `.draft.json` safeguard proposal. Compatible completed held-out
baseline/candidate reports must cover identical attempts, improve pass rate,
avoid every per-case regression, avoid any median-latency regression, and avoid
any input-plus-output token-cost regression. Passing only stages
`.candidates/<name>.json`; promotion still requires permission, verifies the
staged content hash, and keeps the active skill unchanged until that action.

`session_store.py` indexes user, assistant, and tool content locally with FTS5
for discovery and contextual scrolling. It excludes system prompts and reasoning
fields, redacts credential-shaped content, backfills legacy JSON sessions, and
removes index rows when a session is deleted.

### Session persistence & history replay

Conversation state (messages, model, mode, title) is persisted to disk
(`~/.glm-acp/sessions/<id>.json` for the default profile, or its named profile
subdirectory) after every prompt turn and config change.
On `session/load` and `session/resume`, the agent rebuilds the `Session` from
disk via `Session.from_dict`.

Each session also has a small `.meta` sidecar for listing without parsing full
conversation histories. Disk persistence is dispatched off the event loop,
turns are serialized per session, and one pooled GLM HTTP client is reused until
the session's model, endpoint, or thinking configuration changes. Agent
shutdown and session close must close pooled clients.
Session close releases runtime resources but preserves persisted/searchable
history; deletion is a separate storage operation and must never be inferred
from close.
Forks persist `parent_session_id` plus `branch_root_id`; `/lineage` exposes direct
children and identifies the parent session as the rollback path.
Instruction targets, verification ledger, epistemic ledger, metacognitive assessment,
grounded-deliberation conclusions, persistent goal/subgoals, judge budget, and Mixture-of-Agents selection are serialized with the session; read
fingerprints, active checkpoint state, and reference-response caches remain runtime-only.

**Critical:** The ACP `LoadSessionResponse` and `ResumeSessionResponse` only
carry `modes`, `config_options`, and `models` — they do **not** include message
history. To make the restored conversation visible in the editor UI, the agent
must replay it back via `session_update` notifications. `_replay_history()`
walks the persisted messages and sends each user turn as a
`user_message_chunk` and each assistant turn as an `agent_message_chunk`.
System messages and tool-result entries are skipped (internal bookkeeping).
The server runs with `use_unstable_protocol=True` to expose
`session/list`, `session/resume`, and `session/close`.

### Standalone terminal frontend

`glm-acp chat` and `native-glm-acp chat` construct `GlmAcpAgent`, attach a
terminal `Client`, and call the same initialize/new-or-resume/config/prompt methods
as an ACP editor. The frontend may render or serialize updates but must not copy
the tool loop, system prompt, session model, permission rules, or slash-command
handling. One-shot Ask mode fails closed because no operator is available;
Read Only and Bypass remain explicit choices. Bare command invocation must remain
the stdio ACP server for Zed and Registry compatibility.

Interactive TTY input/output selects the full-screen Textual interface; `--plain`
retains the line REPL, while `--prompt`, `--stdin`, and `--json` remain
non-full-screen automation surfaces. The TUI must expose separate conversation,
reasoning, tool, plan, usage, and session state, awaited fail-closed approval
modals with bounded credential-redacted arguments, cancellation, and live settings
that call `set_config_option`/`set_session_mode`. F1 must submit the shared
`/help` command directly, F2 must report the reasoning-panel state, and
presentation controls must also remain reachable through `/reasoning-panel`,
`/settings`, and `/clear-view`. Typing `/` must show a keyboard- and
pointer-selectable menu sourced from the agent's live `available_commands_update`
plus presentation/configuration commands. `/plan`, `/thinking`, and `/model`
must appear first and expose the exact API plans, model-compatible provider
thinking levels, and plan-compatible models; `/api-plan`, `/endpoint`, and
`/reasoning` remain aliases. `/permission`, `/mode`, `/generation`, `/auxiliary`,
and `/mixture` must call the shared session configuration methods. F3 settings
must initially and reactively filter models by plan and thinking levels by
model. The empty-state layout keeps reasoning collapsed and uses compact status,
activity, and plan surfaces. TUI state is presentation-only and must never
become an alternate source of session truth or stored reasoning.

The single-line composer must normalize terminal-routed, explicit Ctrl-V, or
explicit Ctrl-Shift-V multiline paste into one retained prompt rather than inheriting Textual
`Input`'s first-line-only behavior or its app-internal-only clipboard; leading
blank lines cannot erase the paste. OS clipboard fallbacks use `pbpaste`,
PowerShell `Get-Clipboard`, `wl-paste`, `xclip`, or `xsel` only when available,
only on an explicit paste shortcut, without a shell or credential environment, with a
one-second timeout and one-million-character bound. It remains in normal
vertical flow so its full three-row border ends at or before the Footer instead
of being clipped by overlapping dock layout.

The composer-adjacent activity line derives only from existing TUI lifecycle
and ACP session updates. It animates startup, thinking, reasoning, tool work,
and cancellation with a single paused-when-idle timer; approval, completion,
failure, cancellation, and ready states are static, and terminal labels remain
single-line and bounded. `GLM_ACP_TUI_ANIMATION=0` (also `false`, `no`, or `off`)
must disable frame motion while preserving state text. This surface must not
change shared-agent behavior or appear in plain and JSON automation modes.

The visible Footer quit binding is Ctrl-X because Ctrl-Q may be consumed by
POSIX XON/XOFF flow control; F10 and `/exit` are equivalent, and Ctrl-Q remains
hidden compatibility only. Footer entries must stay pointer-actionable.
Shutdown cancels only the TUI's initialization, prompt, provider-usage, and
settings workers through Textual's lifecycle, then bounds shared-agent cleanup
to three seconds so a stalled subprocess cannot freeze terminal exit. TUI
shutdown state must not reuse Textual `MessagePump` private names. Provider
usage uses a daemon-backed synchronous request path in the TUI so stalled DNS
cannot keep the UI event loop or process alive; ACP `/usage` retains the async
client path.

`GlmClient.query_plan_usage()` may issue one GET to the official Z.ai or
BigModel HTTPS monitor origin derived from the configured endpoint. It sends the
credential only to the fixed allowlist, normalizes at most 16 quota windows and
32 MCP details per window, and returns no response bodies on errors.
`GlmAcpAgent.query_provider_usage()` and `/usage` expose the same authoritative
snapshot to ACP clients. The TUI performs one non-blocking startup refresh,
shows 5-hour/weekly/MCP percentages in the sidebar, and refreshes details only
on explicit `/usage` or plan changes; it must not estimate account quota from
session token usage.

## Work Guidance

- Match existing code style: `from __future__ import annotations`, dataclasses for state, type hints throughout
- Keep `glm_client.py` free of ACP-specific imports — it's a pure API wrapper
- Keep `agent.py` free of HTTP/SSE logic — it's a pure ACP layer
- Never write reasoning text to files; it flows only through `agent_thought_chunk`
- Do not weaken the verification prerequisite, permission gate, secret filter, size bounds, or project-local ownership boundary for learned skills.
- Do not auto-promote evolved skills, let delegates mutate state, or treat promptware heuristics as a security boundary.
- Keep session recall local and bounded; never index system prompts, exact reasoning traces, or credential-like values.

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
async def verify():
    agent = GlmAcpAgent()
    agent.on_connect(ClientStub())
    r = await agent.initialize(protocol_version=1)
    assert r.protocol_version == 1
    s = await agent.new_session(cwd='/tmp')
    assert s.config_options[0].category == 'model'
    assert s.config_options[1].category == 'thought_level'
    await agent.aclose()
asyncio.run(verify())
print('Handshake OK')
"
```

Tests live in `tests/` (pytest + pytest-asyncio). Run before merging any
change to `glm_client.py`, `agent.py`, `tools.py`, or `session_store.py`.

## Child DOX Index

No children. All modules are flat in this directory.
