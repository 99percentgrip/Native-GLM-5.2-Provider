# Tests

## Purpose

Own offline verification for ACP behavior, tools, persistence, packaging, authentication, and Registry assets.

## Ownership

- `test_agent.py` covers ACP lifecycle, capabilities, authentication, plans, and prompts.
- `test_cli.py` covers terminal setup and secret-safe status output.
- `test_config.py` covers models, endpoints, credential precedence, and secure persistence.
- `test_registry_package.py` covers public Registry identity and release URL invariants.
- `test_installers.py` covers the user-local command aliases, checksum enforcement, platform mapping, release publication contract, and post-install TUI discovery guidance.
- `test_uninstall.py` covers frozen-copy guards, default and installer-selected profile cleanup, command/PATH removal, credential preservation and purge, and surgical Zed JSONC cleanup with backup.
- Remaining modules cover tools, sessions, streaming, compaction, and GLM HTTP behavior.
- `test_mcp.py` and `test_memory.py` cover remote MCP, scoped memory, verified skills, telemetry, pinning, curation, and reversible archival.
- `test_security.py` covers promptware detection, stored-context blocking, and untrusted-output delimiters.
- `test_session_store.py` covers JSON persistence plus redacted FTS5 discovery, scrolling, legacy backfill, and deletion.
- `test_quality.py` covers tool-loop, failed-verification and unverified-edit recovery, benchmark locking/incremental reports, million-token estimation, and real stdio/SDK ACP process lifecycles.
- `test_cron.py` covers schedule forms, secure persistence, cross-process claims, mutation safety, script-only runs, CLI/tool management, and daemon lifecycle.
- `test_reliability.py` covers progressive repository rules, project facts, fresh verification evidence, syntax/LSP fallback, unchanged-read deduplication, result-aware loop guards, persistent goals/subgoals, and Mixture-of-Agents aggregation.
- `test_extensions.py` covers LSP semantic requests, stable cache prefixes, redacted trajectories, hash-pinned hooks, and the allowlisted Playwright adapter.
- `test_safety_roadmap.py` covers checkpoints/rollback conflicts, bounded references, policy closure, OS sandbox selection, declarative workflows, isolated profiles, worktree lifecycle, and hash-pinned plugins.
- `test_hardening_roadmap.py` covers macOS/Windows sandbox capability truth, digest-pinned worker promotion and rollback, language-aware references, failure-case promotion, observability, Ed25519 plugin trust, and offline fuzz/fault injection.
- `test_awareness.py` covers typed epistemic state, provenance validation, secret/promptware rejection, scope-aware staleness, contradiction resolution, completion certificates, `/awareness`, judge gating, and awareness observability.
- `test_metacognition.py` covers uncertainty separation, risk/mode selection, small-task restraint, empirical escalation, telemetry opt-out, profile corruption/promptware rejection, persistence, commands, and observability.
- `test_deliberation.py` covers evidence-only critic isolation, bounded falsifiable hypotheses, fresh test evidence, stale invalidation, value-of-information ranking, diff redaction, persistence, commands, and metadata-only observability.
- `test_repository_intelligence.py` covers lazy bounded world slices, imports/tests/instructions/ownership/failure metadata, prediction comparison, high-risk pre-mortems, small-task restraint, causal strategy drafts, explicit promotion, fresh/mutated evaluation gates, CLI surfaces, persistence, and metadata-only observability.
- `test_terminal_cli.py` covers standalone parser parity, streamed update rendering, reasoning visibility control, fail-closed non-interactive permissions, credential-redacted permission context, and routing every session option through the shared agent methods.
- `test_tui.py` covers full-screen panel mounting, retained leading-newline multiline paste, composer/Footer non-overlap, shared-runtime prompts, live slash-catalog filtering, keyboard completion/navigation, primary `/plan` and `/thinking` parity across all API plans and model-compatible levels, plan-filtered models, settings filtering, provider-quota startup/manual refresh and sidebar rendering, the animated lifecycle/tool status line and its motion-disable setting, pointer-actionable footer controls, terminal-safe Ctrl-X quit metadata, non-shadowing Textual lifecycle state, bounded stalled cleanup, inline configuration through shared session APIs, every documented function-key action, local presentation controls, thinking visibility, credential-redacted modal approvals, and automatic interactive TUI routing.

## Local Contracts

- Tests must not require a real Z.ai API key or make live model requests.
- Secret tests must assert that credential values never appear in output.
- Registry tests must fail when source, manifest, and archive versions diverge.
- Platform-specific tests must skip when their operating-system semantics are unavailable.
- Streaming tests must cover incomplete HTTP 200 responses, retry boundaries,
  delta coalescing, continuation caps, and exact Deep High/Max request fields.
- Provider-usage tests must prove official-host-only credential routing, exact
  5-hour/weekly/MCP normalization, bounded malformed-response handling, and no
  live account calls.
- Tool tests must cover output bounds and make command exit status observable.
- Persistence tests must verify metadata-sidecar listing and deletion behavior.
- Learning tests must cover verification gating, progressive skill discovery, secret rejection, workspace containment, and deletion limited to agent-owned skills.
- Learning lifecycle tests must cover profile privacy, exact forgetting, usage/revision telemetry, pinned protection, stale/archive thresholds, and restoration.
- Advanced learning tests must cover semantic task and unavailable-tool relevance gates, trace-derived non-promotable drafts, bundle tamper scanning, matching per-case attempt counts, held-out pass improvement, zero latency/token-cost regression, candidate integrity, explicit promotion, and promptware rejection.
- Auxiliary-routing tests must cover titles, compression fallback, recall ranking, and advisory skill evaluation with usage accounting.
- Delegation tests must prove read-only tool exposure, shared worker/tool/token bounds, strict depth, usage accounting, and no recursive or mutating worker surface.
- Compaction tests must cover focus guidance, decisions/fixes/unresolved evidence, permission-required memory proposals, retained categories, quality-decline detection, auxiliary context fallback, pressure tiers, and transactional preservation.
- Fork/session tests must persist parent/root lineage and keep rollback paths visible without deleting either branch.
- Curator tests must cover manual content drift and evidence-only overlap detection without automatic merging.
- Session-search tests must prove system/reasoning/credential exclusion and legacy-session coverage.
- Process-level ACP tests must use test credentials, isolated HOME state, and the official SDK lifecycle helpers where available.
- Cron tests must isolate the configuration directory, avoid live model calls, and prove claim ownership, secret scrubbing, workspace containment, and clean daemon shutdown.
- Reliability tests must prove direct writes defer for newly discovered scoped instructions, shell/output spoofing cannot create verification evidence, later edits invalidate passes, optional diagnostics fail safely, goal state round-trips, and MoA references are reused once per user turn.
- Extension tests must prove LSP position conversion, patch-set atomicity and syntax rejection, bounded batch reduction, telemetry exclusion, hook hash drift disablement, and Playwright allowlisting.
- Safety-roadmap tests must prove auto-checkpoint is off by default and only toggled on via the slash command or env var, sensitive content is never copied into checkpoints, content-addressed bodies deduplicate across projects, retention prunes, large files are excluded, verified legacy snapshots migrate before removal, configurable limits persist within hard bounds and honor environment precedence, rollback stops on later hashes, references cannot escape or expand common secrets, invalid policy fails closed, workflows are acyclic/bounded, required OS isolation selects a real backend or rejects, dirty worktrees are preserved, profiles cannot traverse, and plugin tampering/executable content is rejected.
- Hardening-roadmap tests must prove platform backends do not overclaim isolation, reviewed worker hashes and verification gate atomic promotion, primary conflicts/faults preserve prior content, language ranking prioritizes definitions, failure drafts exclude bodies/paths/secrets, observability ignores corruption, signatures require exact trusted keys, and fuzzing is deterministic/offline.
- Awareness tests must prove unsupported evidence/criteria fail closed, relevant edits stale support without invalidating user evidence, every criterion and fresh post-edit verification gates persistent completion, active contradictions block completion, external bodies are excluded, and incomplete certificates bypass the auxiliary judge.
- Metacognition tests must prove every uncertainty category remains distinct, unverified edits and release operations select high assurance, trivial tasks remain direct, weak profiles only escalate after enough outcomes, corrupt/malicious metadata is ignored, telemetry opt-out disables profiles, and no task text or path enters outcome events.
- Deliberation tests must prove direct tasks add no overhead, diagnosis uses two or three distinct predictions/falsifiers, test results cite fresh non-user evidence, stale support resets tests and critic approval, approval cites known evidence, critic packets exclude primary reasoning, diff secrets are redacted, Read Only excludes command recommendations, and telemetry stores metadata only.
- Repository-intelligence tests must prove source bodies never enter persisted state, traversal and context stay bounded, direct trivial tasks add no model overhead, predictions freeze before mutation and compare with observed files/checks, ownership and failure history remain metadata-only, and pre-mortems are risk-triggered.
- Safe metacognitive-learning tests must prove attribution is typed and evidence-bounded, drafts cannot self-promote, promotion requires two supports plus explicit fresh/transformed improvement, safety and cost regressions fail closed, all eleven adversarial cases have mutations, and observability excludes task/tool bodies.

## Work Guidance

- Use temporary directories and monkeypatched environment variables for credentials and state.
- Keep network behavior deterministic with existing HTTP mocks.

## Verification

- Run `.venv/bin/python3 -m pytest tests/ -q`.

## Child DOX Index

No children.
