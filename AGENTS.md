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
- Terminal authentication must never echo or log `ZAI_API_KEY`; environment credentials take precedence over the user-only stored credential file.

## Project Purpose

This project implements a native ACP (Agent Client Protocol) server for Z.ai GLM models. It is a standalone Python package that Zed (or any ACP-compatible editor) launches as a subprocess over stdio. It wraps the Z.ai GLM Coding Plan API directly — not the generic openai_compatible wrapper — to unlock GLM's 1M context window, live reasoning traces, and long-running generation without stalls.

- Language: Python 3.10+
- Transport: ACP over stdio (JSON-RPC 2.0)
- APIs: Z.ai Coding Plan, Standard API, and BigModel (CN)
- Models: GLM-5.2, GLM-5-Turbo, GLM-4.7, GLM-5V-Turbo, GLM-4.5V, and GLM-4.6V according to the selected API plan
- Entry points: `glm-acp`, `python3 -m glm_acp`, and the frozen `native-glm-acp` executable

## Current Project Status

- Package and ACP implementation version is `0.3.0` from `glm_acp.__version__`.
- Source installs, the `glm-acp` console script, module execution, and frozen binaries share `cli.main()`.
- ACP initialization advertises Registry-compatible `zai-api-key-setup` Terminal Auth.
- Terminal setup stores credentials atomically without echoing or logging the key; environment credentials take precedence.
- GitHub Actions tests Python 3.10–3.13 and packages Linux x86-64/ARM64, macOS Intel/Apple Silicon, and Windows x86-64 binaries.
- Official Z.ai Web Search, Web Reader, and optional local Vision MCP capabilities are exposed alongside configurable MCP servers.
- Root project instructions and permission-gated `.glm-acp/memory.md` knowledge are loaded into model context.
- Tagged releases publish checksums, provenance attestations, Python distributions, Registry metadata, and the icon.

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
# expect: editable glm_acp metadata and glm_acp-0.3.0.dist-info
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
