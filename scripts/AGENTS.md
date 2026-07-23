# Installation scripts

## Purpose

Own public, runtime-free installation entry points for GitHub release binaries.

## Ownership

- `install.sh` installs verified Linux and macOS archives for the detected architecture.
- `install.ps1` installs the verified Windows x86-64 archive and maintains the user PATH.

## Local Contracts

- Installers download only versioned or `latest` assets from the public GitHub release.
- Every frozen archive must pass its published SHA-256 check before extraction or installation.
- Installation is user-local by default and must not require administrator privileges.
- Both `native-glm-acp` and the user-facing `glm-acp` command must be installed.
- Successful installation points users to credential setup, the full-screen
  `glm-acp chat` frontend, its live `/` command menu, F1 help, and Ctrl-X exit.
- Any PATH modification must retain the exact `# Native GLM ACP` marker consumed by `glm-acp --uninstall`.
- Unsupported operating systems and architectures fail with an actionable message.

## Work Guidance

- Keep dependencies limited to standard operating-system tools.
- Never accept or inspect Z.ai credentials during package installation.

## Verification

- Run `bash -n scripts/install.sh`.
- Run `.venv/bin/python3 -m pytest tests/test_installers.py -q`.

## Child DOX Index

No children.
