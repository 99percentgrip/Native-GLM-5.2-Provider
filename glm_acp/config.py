"""GLM model registry and configuration constants."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "glm-5.2"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_TOKENS = 128_000
MAX_AUTO_CONTINUATIONS = 20

# Retry configuration for transient API errors (429, 500, 502, 503, 504)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds, exponential: 1s, 2s, 4s
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Per-model max_tokens limits.  Models not listed here fall back to
# DEFAULT_MAX_TOKENS.
MAX_TOKENS_BY_MODEL: dict[str, int] = {}

# Vision models (models with a "v" suffix). These support image inputs
# but do not support thinking / reasoning_effort.
VISION_MODELS = frozenset({"glm-4.5v", "glm-4.6v"})

# --- API endpoints (plans) ---
# The user can switch between these from the chat dropdown so they're not
# locked into a single plan.  Each maps to a different Z.ai base URL.
API_ENDPOINTS: dict[str, dict[str, str]] = {
    "coding": {
        "name": "Coding Plan",
        "description": "Z.ai Coding Plan — GLM-5.2, GLM-5-Turbo, GLM-4.7 (default)",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
    },
    "standard": {
        "name": "Standard API",
        "description": "Z.ai standard API — pay-as-you-go, broader model access incl. vision",
        "base_url": "https://api.z.ai/api/paas/v4",
    },
    "bigmodel": {
        "name": "BigModel (CN)",
        "description": "BigModel open platform (China) — Chinese mainland endpoint",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
}
DEFAULT_API_ENDPOINT = "coding"

# --- Token estimation (heuristic) ---
# _estimate_tokens uses 3.5 chars/token (code is denser than natural
# language which averages ~4 chars/token). The ratio is applied locally
# in GlmAcpAgent._estimate_tokens rather than referenced from here.

# --- Context compaction (Claude Code parity) ---
# Trigger compaction when estimated context usage exceeds this fraction of the
# model's context window.
COMPACTION_THRESHOLD = 0.85
# Number of most-recent messages to preserve verbatim after compaction.
COMPACTION_KEEP_RECENT = 4
# Max tokens for the summarization call itself.
COMPACTION_SUMMARY_MAX_TOKENS = 16_384

# Context window sizes in tokens, keyed by model id.
CONTEXT_WINDOW_TOKENS: dict[str, int] = {
    "glm-5.2": 1_000_000,
    "glm-5-turbo": 1_000_000,
    "glm-4.7": 1_000_000,
    "glm-4.5v": 131_072,
    "glm-4.6v": 131_072,
}

COMPACTION_SYSTEM_PROMPT = """\
You are a conversation summarizer for an AI coding assistant. Your job is to \
create a concise but information-dense summary of the conversation so far, \
so that the assistant can continue working with full context after the older \
messages are compacted.

Your summary MUST preserve:
1. The user's original goal and any refined requirements
2. Key decisions made and their rationale
3. Files that were read, created, or modified — include paths and a brief \
description of their current state / important contents
4. Any errors encountered and how they were resolved (or remain unresolved)
5. Pending tasks or next steps that were planned
6. Any important code snippets, function signatures, or configuration values \
that are still relevant
7. Tool results that contain critical information (e.g. test output, command \
results, search results)

Write the summary as a clear, structured document. Be specific — include \
actual file paths, function names, error messages, and values. Do not \
hallucinate information that was not in the conversation. If something is \
uncertain, note the uncertainty.

Format:
## Goal
...
## Work Done
...
## Key Files
...
## Decisions
...
## Pending / Next Steps
...
## Important Context
...
"""

COMPACTION_USER_PREFIX = (
    "Here is the conversation to summarize. Produce a comprehensive summary "
    "following the structure in your instructions:\n\n---\n\n"
)

MODELS: dict[str, dict[str, Any]] = {
    "glm-5.2": {
        "name": "GLM-5.2 (Flagship)",
        "description": "Latest flagship — maximum reasoning, coding, and long-horizon agentic tasks",
        "context_window": "1M",
        "plans": ["coding", "standard", "bigmodel"],
    },
    "glm-5-turbo": {
        "name": "GLM-5-Turbo",
        "description": "Flagship model optimized for speed — complex tasks with lower latency",
        "context_window": "1M",
        "plans": ["coding", "standard", "bigmodel"],
    },
    "glm-4.7": {
        "name": "GLM-4.7",
        "description": "Balanced model for daily development and routine tasks",
        "context_window": "1M",
        "plans": ["coding", "standard", "bigmodel"],
    },
    "glm-4.5v": {
        "name": "GLM-4.5V (Vision)",
        "description": "Vision-capable — analyze screenshots, diagrams, charts",
        "context_window": "128K",
        "plans": ["standard", "bigmodel"],
    },
    "glm-4.6v": {
        "name": "GLM-4.6V (Vision)",
        "description": "Vision model — newer vision model with improved OCR and image understanding",
        "context_window": "128K",
        "plans": ["standard", "bigmodel"],
    },
}

THOUGHT_LEVELS: dict[str, dict[str, Any]] = {
    "disabled": {
        "name": "Off",
        "description": "No reasoning — fast responses for simple tasks",
        "thinking_type": "disabled",
        "reasoning_effort": None,
        "models": None,  # None = all models
    },
    "enabled": {
        "name": "Standard",
        "description": "Full reasoning traces streamed live",
        "thinking_type": "enabled",
        "reasoning_effort": None,
        "models": None,
    },
    "high": {
        "name": "Deep · High",
        "description": "Deeper multi-step reasoning for complex tasks (GLM-5.2 only)",
        "thinking_type": "enabled",
        "reasoning_effort": "high",
        "models": ["glm-5.2"],
    },
    "max": {
        "name": "Deep · Max",
        "description": "Maximum reasoning depth — deepest analysis (GLM-5.2 only)",
        "thinking_type": "enabled",
        "reasoning_effort": "max",
        "models": ["glm-5.2"],
    },
}


def thought_levels_for_model(model: str) -> dict[str, dict[str, Any]]:
    """Return the subset of thought levels available for the given model.

    Vision models don't support thinking — only 'disabled' is available.
    Deep reasoning levels are restricted to models that list them.
    """
    if model in VISION_MODELS:
        return {"disabled": THOUGHT_LEVELS["disabled"]}
    return {
        k: v
        for k, v in THOUGHT_LEVELS.items()
        if v["models"] is None or model in v["models"]
    }


def models_for_plan(plan: str) -> dict[str, dict[str, Any]]:
    """Return the subset of models available on the given API plan."""
    return {
        model_id: info
        for model_id, info in MODELS.items()
        if plan in info.get("plans", [])
    }


# Tools that modify the filesystem or execute commands — these require
# user permission when the session is in "ask" mode and are blocked in
# "read" mode.
DESTRUCTIVE_TOOLS = frozenset({"write_file", "edit_file", "run_command"})

AUTH_METHOD_ID = "zai-api-key-setup"
CONFIG_DIR_ENV = "GLM_ACP_CONFIG_DIR"
CREDENTIALS_FILENAME = "credentials.json"


def config_dir() -> Path:
    """Return the per-user configuration directory without creating it."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "glm-acp"
    if sys_platform() == "darwin":
        return Path.home() / "Library" / "Application Support" / "glm-acp"
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    return (
        Path(xdg_config_home) / "glm-acp"
        if xdg_config_home
        else Path.home() / ".config" / "glm-acp"
    )


def sys_platform() -> str:
    """Small indirection that keeps platform selection easy to test."""
    import sys

    return sys.platform


def credentials_path() -> Path:
    return config_dir() / CREDENTIALS_FILENAME


def load_stored_api_key() -> str | None:
    """Load the locally stored API key, returning None for invalid state."""
    try:
        payload = json.loads(credentials_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    key = payload.get("zai_api_key") if isinstance(payload, dict) else None
    return key.strip() if isinstance(key, str) and key.strip() else None


def store_api_key(key: str) -> Path:
    """Atomically store an API key in a user-only configuration file."""
    normalized = key.strip()
    if not normalized:
        raise ValueError("API key cannot be empty")

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:
        pass

    target = credentials_path()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        dir=directory,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"zai_api_key": normalized}, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return target


def has_api_key() -> bool:
    return bool(
        os.environ.get("ZAI_API_KEY")
        or os.environ.get("Z_AI_API_KEY")
        or load_stored_api_key()
    )


def get_api_key() -> str:
    key = (
        os.environ.get("ZAI_API_KEY")
        or os.environ.get("Z_AI_API_KEY")
        or load_stored_api_key()
    )
    if not key:
        raise RuntimeError(
            "Z.ai API credentials are required. Run `glm-acp --setup` or set "
            "ZAI_API_KEY. Get your key at https://z.ai/"
        )
    return key
