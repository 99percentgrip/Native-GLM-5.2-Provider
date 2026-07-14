# Tests

## Purpose

Own offline verification for ACP behavior, tools, persistence, packaging, authentication, and Registry assets.

## Ownership

- `test_agent.py` covers ACP lifecycle, capabilities, authentication, plans, and prompts.
- `test_cli.py` covers terminal setup and secret-safe status output.
- `test_config.py` covers models, endpoints, credential precedence, and secure persistence.
- `test_registry_package.py` covers public Registry identity and release URL invariants.
- Remaining modules cover tools, sessions, streaming, compaction, and GLM HTTP behavior.

## Local Contracts

- Tests must not require a real Z.ai API key or make live model requests.
- Secret tests must assert that credential values never appear in output.
- Registry tests must fail when source, manifest, and archive versions diverge.
- Platform-specific tests must skip when their operating-system semantics are unavailable.

## Work Guidance

- Use temporary directories and monkeypatched environment variables for credentials and state.
- Keep network behavior deterministic with existing HTTP mocks.

## Verification

- Run `.venv/bin/python3 -m pytest tests/ -q`.

## Child DOX Index

No children.
