# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

- Public releases and ACP Registry metadata identify Aleksejs Kozlitins as author and use Apache-2.0.
- Registry installation uses version-pinned frozen binaries for Linux x86-64/ARM64, macOS Intel/Apple Silicon, and Windows x86-64.
- Public GitHub installation provides checksum-verifying, user-local installers that expose both `native-glm-acp` and `glm-acp` without requiring Python, Node.js, or administrator privileges.
- Public frozen installs provide `glm-acp --uninstall`; credentials are preserved unless the user explicitly adds `--purge`, and source or Registry-managed copies must not self-delete.
- Standalone terminal interaction is discoverable and editor-parity oriented: typing `/` opens the live agent command catalog with `/plan`, `/thinking`, and `/model` first; `/plan` names Coding Plan, Standard API, and BigModel (CN) directly; `/thinking` controls provider reasoning while F2 and `/reasoning-panel` control only its view; all model, plan, thinking, permission, mode, generation, auxiliary-model, and Mixture-of-Agents changes use the same session APIs as ACP editors.
- Standalone terminal composition accepts terminal-routed bracketed paste plus explicit Ctrl-V and Ctrl-Shift-V OS clipboard shortcuts without dropping a leading-newline prompt, presents multiline content safely in the single-line composer, and keeps the complete composer border above the footer. Optional platform clipboard readers run only on explicit paste, without a shell or credential environment, under a one-second timeout and one-million-character result bound.
- Standalone TUI activity is visible but presentation-only: a low-overhead status line animates startup, thinking, reasoning, tool work, and cancellation; shows static approval, completion, failure, and ready states; bounds streamed labels; and honors `GLM_ACP_TUI_ANIMATION=0` without changing ACP, plain, or JSON behavior.
- Terminal quit is delivery-safe and bounded: the visible/clickable footer uses Ctrl-X, F10 and `/exit` are equivalents, Ctrl-Q is hidden because POSIX flow control may swallow it, TUI lifecycle state never shadows Textual internals, provider telemetry DNS/HTTP cannot hold the UI event loop open, and shutdown waits at most three seconds for shared-resource cleanup.
- Provider limits remain authoritative and credential-safe: `/usage` and the TUI sidebar use Z.ai's official monitor endpoint for 5-hour, weekly, and MCP Coding Plan quota windows, make no local quota estimates, and never send credentials to a custom API host.
- Terminal authentication must never echo or log `ZAI_API_KEY`; environment credentials take precedence over the user-only stored credential file.
- Push-to-talk voice uses local Whisper exclusively: audio is captured via `arecord`/`afrecord`, transcribed on-device with `faster-whisper` (base model, 74 MB), never sent to any API, and the transcribed text is appended to the composer for review before sending; `GLM_ACP_WHISPER_MODEL` selects the model size.
- Notification sounds are opt-in and bounded: `GLM_ACP_SOUND=1` enables a terminal bell on turn completion or failure with a 5-second cooldown, suppressed during voice recording to prevent feedback loops; the default is off.
- Desktop notifications are smart and rate-limited: `GLM_ACP_NOTIFY=0` disables them; otherwise they fire only for turns exceeding 10 seconds, at most once per 30 seconds, via `notify-send` (Linux), `osascript` (macOS), or PowerShell (Windows).
- Agent learning is inspectable, permission-gated, secret-safe, and reversible: facts/skills stay project-local, while explicitly approved user preferences use private cross-project storage.
- Advanced learning remains evidence-gated: failed traces may produce drafts, but candidates require higher held-out pass rate with no per-case, median-latency, or token-cost regression and explicit promotion; delegation is read-only, depth-one, shared-budgeted, and permission-gated.
- Coding reliability prioritizes progressively scoped repository instructions, edit-fresh verification evidence, persistent judged goals and acceptance criteria, post-write syntax/semantic diagnostics, unchanged-read deduplication, opt-in reference-model aggregation, and result-aware loop stops.
- Awareness remains bounded and inspectable: typed epistemic records cite only harness-issued metadata evidence; relevant edits invalidate support; active contradictions and unsupported criteria block persistent-goal completion before the auxiliary judge; no chain-of-thought is stored.
- Metacognitive control remains deterministic and advisory: uncertainty classes and execution modes derive from inspectable runtime facts; aggregate capability profiles are metadata-only and profile-isolated; empirical history may raise assurance but never expand permissions, authorize workers, change trusted policy, or store reasoning.
- Grounded deliberation remains evidence-only and bounded: an isolated thinking-disabled critic receives only objectives, a credential-redacted diff, fresh harness evidence, hypothesis outcomes, and completion metadata; ambiguous diagnosis uses two or three falsifiable alternatives with fresh evidence-backed tests; value-of-information ranking is advisory and cannot bypass permissions or policy.
- Repository intelligence remains lazy and metadata-bounded: it never snapshots source bodies, direct small tasks incur no world-model overhead, pre-edit impact predictions freeze at mutation, high-risk pre-mortems are inspectable, and observed impact never substitutes for fresh verification.
- Safe metacognitive learning remains causal, inert by default, and evaluation-gated: fixed strategy drafts require two evidence-backed attributions, explicit promotion, gains on fresh time-split and transformed cases, and no quality, safety, calibration, latency, token, or small-task-overthinking regression; promotion never changes trusted authority.
- Advanced execution remains declarative and fail-closed: checkpoints are conflict-aware, secret-safe, and profile-configurable within hard bounds; context references are bounded and language-ranked; policy rules inspect nested workflow steps; worker promotion is verification- and digest-gated with transactional rollback; profiles isolate user state; plugin packages are permission-scoped, data-only, hash-pinned, and optionally require trusted Ed25519 publishers.
- Quality evidence remains private and reproducible: failure drafts contain metadata only until explicitly promoted into outcome-based cases; local observability never stores bodies or raw identities; fuzzing and fault injection run offline and deterministically.

## Project Purpose

Native GLM ACP is an open-source ACP-native coding agent runtime for Z.ai GLM models. It is a standalone Python package that Zed (or any ACP-compatible editor) launches as a subprocess over stdio. It wraps the Z.ai GLM Coding Plan API directly — not the generic openai_compatible wrapper — to unlock GLM's 1M context window, live reasoning traces, and long-running generation without stalls.

- Language: Python 3.10+
- Transport: ACP over stdio (JSON-RPC 2.0)
- APIs: Z.ai Coding Plan, Standard API, and BigModel (CN)
- Models: GLM-5.2, GLM-5-Turbo, GLM-4.7, GLM-5V-Turbo, GLM-4.5V, and GLM-4.6V according to the selected API plan
- Entry points: `glm-acp`, `python3 -m glm_acp`, and the frozen `native-glm-acp` executable; append `chat` for the standalone terminal frontend while bare invocation remains ACP stdio

## Current Project Status

- Package and ACP implementation version is `1.9.4` from `glm_acp.__version__`.
- GitHub release `v1.9.4` publishes the five supported frozen binaries, checksums, provenance attestations, Python distributions, Registry metadata, the icon, checksum-verifying Unix and Windows installers, and safe one-command uninstall support.
- ACP Registry publication is tracked in `agentclientprotocol/registry#439` and remains pending until Registry maintainers merge it.
- Source installs, the `glm-acp` console script, module execution, and frozen binaries share `cli.main()`.
- `glm-acp chat` and `native-glm-acp chat` open a cross-platform full-screen Textual interface over the full existing `GlmAcpAgent` runtime without an editor; a live `/` completion menu consumes the same available-command updates as Zed, puts API-plan/thinking/model controls first, and exposes every session setting through the shared APIs. Reasoning starts collapsed; F2 and `/reasoning-panel` toggle only its view, while `/thinking` changes the actual provider level. **F4 cycles a four-view working-tree panel** on the left side (session changes, git status, diff, file browser) sharing one screen location. **F5 toggles push-to-talk** — records via `arecord`, transcribes with local `faster-whisper` (base model, offline, free), and appends to the composer; faster-whisper is bundled in the frozen binary (156 MB on Linux). The composer stays **always enabled during active turns** — Enter **queues prompts** that auto-drain FIFO when each turn completes, with a visible queue-status line showing count and preview. **Notification sounds** (opt-in via `GLM_ACP_SOUND=1`, terminal bell, 5s cooldown) and **smart desktop notifications** (only for turns >10 s, rate-limited to 1/30 s, `notify-send`/`osascript`/PowerShell) fire on turn complete/fail. The session sidebar shows a compact **awareness indicator** (execution mode, evidence count, risk score, active contradictions). Agent output is rendered as **structured Markdown** (headers, bullets, code blocks) with streaming-safe debounce. The composer retains terminal-routed or Ctrl-V OS clipboard content as a usable single-line prompt and stays fully above the footer. A bounded, low-overhead activity line animates startup, thinking, reasoning, tool work, and cancellation, then reports approval/completion/failure/ready states without becoming runtime truth; `GLM_ACP_TUI_ANIMATION=0` disables motion. Compact conversation/activity/plan/status panels, credential-redacted modal approvals, F1 help, F3 settings, `/settings`, and `/clear-view` remain presentation-only while persistence, tools, MCP/browser integration, workers, learning, awareness, repository intelligence, checkpoints, and verification stay shared with ACP clients. `--plain`, `--prompt`, `--stdin`, and `--json` preserve line and automation surfaces.
- `/usage` works in ACP editors and terminal frontends; the TUI performs one non-blocking startup refresh and shows live provider-reported 5-hour, weekly, and monthly MCP quota percentages in the session sidebar, while an explicit `/usage` refresh displays available used/limit/remaining/reset details.
- Public frozen binaries support one-command removal of installer-owned commands, PATH markers, and matching custom Zed configuration with an automatic settings backup.
- ACP initialization advertises Registry-compatible `zai-api-key-setup` Terminal Auth.
- Terminal setup stores credentials atomically without echoing or logging the key; environment credentials take precedence.
- GitHub Actions tests Python 3.10–3.13 and packages Linux x86-64/ARM64, macOS Intel/Apple Silicon, and Windows x86-64 binaries.
- Official Z.ai Web Search, Web Reader, and optional local Vision MCP capabilities are exposed alongside configurable MCP servers.
- Root-to-target `.hermes.md`/Hermes, AGENTS, Claude, GLM, and Cursor instructions plus permission-gated `.glm-acp/memory.md` knowledge are progressively loaded into model context; direct writes defer when they first reveal closer rules.
- Successfully verified tasks receive one bounded learning review; approved reusable procedures are progressively loaded, usage-tracked, refinable, pinnable, reversibly archivable, and forgettable.
- Private user-profile memory and redacted FTS5 session recall provide cross-project and cross-session learning without indexing system prompts or reasoning traces.
- Promptware scanning blocks suspicious stored context and delimits tool, MCP, embedded-resource, and recalled output as untrusted data.
- Structured compaction preserves decisions, fixes, unresolved work, plan/edit/verification evidence, and memory proposals; it accepts an optional focus, scores summary quality over time, reports retained categories and pressure at 60%/75%/85%, and may use a configurable auxiliary GLM model.
- The auxiliary GLM path covers titles, compression, recall ranking, skill evaluation, and bounded workers. Workers provide permission-gated read-only investigation/review under shared token/tool budgets, strict iteration/time limits, and no recursive delegation.
- ACP forks persist parent/root lineage, while relevant skill metadata, bundles, and benchmark-gated candidate promotion extend learning without automatic replacement.
- Project facts and canonical checks are auto-detected; edit-fresh verification evidence persists, and post-write Python/JSON/TOML syntax plus optional Python/TypeScript/Go/Rust LSP diagnostics feed the acting model.
- Persistent goals and subgoal acceptance criteria use a bounded auxiliary completion judge. Opt-in Mixture-of-Agents runs cached parallel reference reviews while the primary GLM remains the aggregator and sole actor.
- A typed epistemic ledger tracks observations, assumptions, hypotheses, contradictions, unknowns, and capability limits with provenance and scope-aware freshness. `/awareness` shows the state and completion certificate; metadata-only observability reports evidence coverage and prevented unsupported completions.
- A bounded metacognitive controller separates ambiguity, knowledge, diagnostic, capability, verification, and permission uncertainty; selects direct, grounded, deliberate, or high-assurance posture; and uses redacted outcome aggregates by task family and coarse environment to escalate weak historical cases without overthinking trivial work.
- Deliberate diagnosis generates two or three falsifiable hypotheses and tracks tests against fresh evidence IDs; a separately prompted auxiliary critic reviews only goals, bounded redacted diffs, fresh evidence, and completion metadata, while deterministic value-of-information ranking prioritizes the cheapest reliable allowed evidence action.
- Lazy repository intelligence combines bounded LSP/tool paths, imports, tests, manifests, instructions, CODEOWNERS, current changes, and project-matched failure classes; it predicts files/checks/packaging/platforms before edits, compares observed impact afterward, and adds deterministic pre-mortems only for high-risk work.
- Safe metacognitive learning attributes corrected failures to fixed cause/intervention classes and drafts allowlisted strategies without activating them; `meta-cases` and `meta-eval` enforce overall, fresh, transformed, per-case, safety, calibration, evidence, latency, token, and restraint gates before explicit promotion.
- Repeated identical tool batches, repeated failures, and unchanged read-only results are interrupted before the 50-iteration ceiling; unchanged reads are deduplicated, malformed JSON arguments receive corrective feedback, and shell tools do not inherit common credential environment variables.
- Installed language servers provide read-only semantic navigation, transactional hash-pinned multi-file patches commit all-or-nothing, and bounded batch reads reduce tool round trips without arbitrary code execution.
- Stable managed-prompt prefixes expose cache-hit ratios; metadata-only redacted trajectories and hash-pinned lifecycle hooks add evidence and policy without storing prompts, outputs, commands, reasoning, credentials, or raw session IDs.
- Permission-gated isolated Playwright MCP supplies accessibility, console, network, screenshot, and interaction evidence without arbitrary browser JavaScript evaluation or inherited credentials.
- Bounded secret-safe checkpoints precede workspace mutations only when auto-checkpoint is explicitly enabled (default **off** via `/checkpoint auto on` or `GLM_ACP_AUTO_CHECKPOINT=1`); compressed Git-compatible content-addressed objects deduplicate file bodies across projects, per-project history/age and a global ceiling prune automatically, large files are excluded, verified legacy copies can be migrated, and exact post-agent hashes make `/rollback` stop on later conflicts instead of overwriting them.
- Explicit `@file:`, `@folder:`, `@symbol:`, and `@diff` references stay workspace-contained, bounded, secret-aware, and delimited as untrusted context.
- Ordered repository policy rules, static dependency workflows, optional Bubblewrap isolation, detached worktree implementation workers, named user profiles, and permission-scoped hash-pinned data-only plugin packages provide safe extensibility without arbitrary orchestration code or automatic merges.
- Cross-platform containment capability-detects Linux Bubblewrap and macOS Seatbelt, adds Windows process-tree Job Objects without treating them as filesystem isolation, and keeps required mode fail-closed.
- Detached workers support exact-digest inspection, required isolated verification, conflict-aware transactional promotion, rollback-on-fault, and reviewed discard while preserving the worker after promotion.
- Explicit folder/symbol references spend their fixed budget on language-aware definitions, references, task terms, tests, manifests, and current changes.
- Metadata-only failure drafts can be permission-gated into runnable project-local regression cases; a local observability dashboard and deterministic offline hardening command expose reliability evidence without prompts, outputs, commands, paths, reasoning, credentials, or raw session IDs.
- Data-only plugin packages support explicit Ed25519 publisher trust, CLI-only private-key signing, signature enforcement policy, and exact manifest verification in addition to content hashes.
- Expired MCP HTTP sessions and restarted stdio servers reinitialize automatically with per-server initialization locking.
- The opt-in quality harness provides 11 outcome-based Python, TypeScript, Go, and Rust cases plus a credential-safe one-command runner with single-run locking, visible progress, and incremental JSON/Markdown handoff reports; live runs remain outside ordinary CI.
- Persistent scheduled automation supports relative one-shots, intervals, timezone-aware five-field cron, and aware ISO timestamps; permission-gated management, fresh non-persisted runs, skills/bundles, script prechecks, script-only mode, `[SILENT]`, renewable cross-process claims, and bounded redacted artifacts are available through ACP and `glm-acp cron`.

## Install and distribution (binding)

Source checkouts MUST install the package into the venv that Zed launches:

```bash
cd /path/to/glm-acp
uv pip install -e .
```

Without this, `python3 -m glm_acp` only resolves when run from this repo's
directory (Python puts the cwd on `sys.path`). Zed sets the subprocess cwd
to whatever project is open, so an uninstalled package crashes with
`ModuleNotFoundError` (exit 1) in any other repository. A bare `git clone`
is not enough for source-based launches. Public Registry installs use the
frozen `native-glm-acp` executable and do not require Python or a
repository-specific virtualenv.

Verify the install:

```bash
ls .venv/lib/*/site-packages/ | grep glm_acp
# expect: editable glm_acp metadata and glm_acp-1.9.4.dist-info
```

## Verification

```bash
uv sync --frozen --extra dev
uv run --frozen pytest tests/ -q
uv run --frozen pip-audit
uv build
uv run --frozen pyinstaller --noconfirm --clean --onefile --name native-glm-acp --collect-all acp glm_acp/launcher.py
dist/native-glm-acp --version
```

Before Registry submission, also run the official Registry schema builder and
authentication verifier against the published version-pinned archives.

## Child DOX Index

| Path | Purpose | Ownership |
|------|---------|-----------|
| `glm_acp/` | Python ACP agent, GLM client, tools, configuration, and CLI | Python implementation |
| `tests/` | Offline behavioral, security, packaging, and Registry verification | Python implementation |
| `registry/` | ACP Registry manifest template and icon | Release engineering |
| `.github/` | Cross-platform CI and release automation | Release engineering |
| `pyproject.toml` | Package metadata, dependencies, entry point, and build configuration | Python implementation |
| `uv.lock` | Reproducible dependency resolution | Python implementation |
| `README.md` | Installation, operation, security, and release guide | Project maintainers |
| `benchmarks/` | Opt-in native/external coding-agent quality evaluation | Quality engineering |
| `scripts/` | Runtime-free public installers for published frozen binaries | Release engineering |
