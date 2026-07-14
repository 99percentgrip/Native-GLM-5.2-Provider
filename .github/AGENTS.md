# GitHub automation

## Purpose

Own continuous verification and reproducible cross-platform release publication.

## Ownership

- `workflows/ci.yml` validates formatting, linting, tests, packages, and frozen binaries.
- CI audits the locked Python dependency graph before package publication.
- `workflows/release.yml` builds, checksums, attests, and publishes tagged release artifacts.
- `workflows/quality.yml` manually runs opt-in live outcome benchmarks and publishes a secret-safe job summary/artifact.

## Local Contracts

- A `vX.Y.Z` tag must exactly match `glm_acp.__version__`.
- The declared Python 3.10+ compatibility range must be tested through Python 3.13.
- Release archives must match the filenames in `registry/agent.json`.
- Linux x86-64/ARM64, macOS Intel/Apple Silicon, and Windows x86-64 artifacts must each run `--version` before publication.
- Each frozen executable must remain below the 30 MiB release-size ceiling.
- Published archives receive SHA-256 files and GitHub build-provenance attestations.
- Build Python distributions before downloading frozen-binary artifacts so temporary
  release files cannot enter the source distribution.
- Ordinary push/PR CI validates the benchmark catalog but never spends live model tokens.
- Live quality runs require the repository `ZAI_API_KEY` secret and explicit workflow dispatch.

## Work Guidance

- Keep CI and release installs locked with `uv.lock`.
- Pin security-sensitive GitHub-maintained actions to immutable commits when practical.

## Verification

- Pull requests and pushes run `.github/workflows/ci.yml`.
- Version tags run `.github/workflows/release.yml`.
- Manual quality runs use `.github/workflows/quality.yml`.

## Child DOX Index

No children.
