# Benchmarks

## Purpose

Own opt-in, reproducible coding-agent quality evaluation without placing live API usage in CI.

## Ownership

- `eval.py` runs native or external agents against isolated task fixtures and emits JSON results.
- `cases.json` defines prompts, fixture files, verification commands, and time limits.

## Local Contracts

- Live native runs require the user's existing Z.ai credential configuration.
- Benchmark output must never contain API keys, authentication paths, reasoning traces, or session IDs.
- Cases must verify observable repository outcomes rather than judge prose subjectively.
- External runners receive the task prompt over stdin and the isolated workspace as their cwd.

## Work Guidance

- Keep default cases small enough for routine release comparisons.
- Add cases for confirmed quality regressions and important tool workflows.

## Verification

- Run `.venv/bin/python3 benchmarks/eval.py --list`.

## Child DOX Index

No children.
