# Registry assets

## Purpose

Own the ACP Registry manifest template and the monochrome catalog icon.

## Ownership

- `agent.json` defines the public `native-glm-acp` identity and immutable release URLs.
- `icon.svg` is the 16×16 monochrome Registry icon.

## Local Contracts

- Manifest version must equal `glm_acp.__version__` and the release tag.
- Binary archives must be version-pinned GitHub release URLs for Linux x86-64/ARM64, macOS Intel/Apple Silicon, and Windows x86-64.
- The release workflow generates its published `agent.json` from the same identity and URL contract.

## Work Guidance

- Keep the icon small, monochrome, and legible at 16×16.
- Update every platform URL together when the version changes.

## Verification

- Run `.venv/bin/python3 -m pytest tests/test_registry_package.py -q`.
- Run the ACP Registry `build_registry.py --dry-run` and auth verifier before submission.

## Child DOX Index

No children.
