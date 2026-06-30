"""GLM model registry and configuration constants."""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.2"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_TOKENS = 128_000
MAX_AUTO_CONTINUATIONS = 20

# --- Token estimation (heuristic) ---
CHARS_PER_TOKEN = 4  # ~4 chars per token for mixed English/code content

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

MODELS: dict[str, dict[str, str]] = {
    "glm-5.2": {
        "name": "GLM-5.2 (Flagship)",
        "description": "Latest flagship — maximum reasoning, coding, and long-horizon agentic tasks",
        "context_window": "1M",
    },
    "glm-5-turbo": {
        "name": "GLM-5-Turbo",
        "description": "Flagship model optimized for speed — complex tasks with lower latency",
        "context_window": "1M",
    },
    "glm-4.7": {
        "name": "GLM-4.7",
        "description": "Balanced model for daily development and routine tasks",
        "context_window": "1M",
    },
}

THOUGHT_LEVELS: dict[str, dict[str, str]] = {
    "disabled": {
        "name": "Off",
        "description": "No reasoning — fast responses for simple tasks",
    },
    "enabled": {
        "name": "On",
        "description": "Full reasoning traces streamed live",
    },
}


def get_api_key() -> str:
    key = os.environ.get("ZAI_API_KEY") or os.environ.get("Z_AI_API_KEY")
    if not key:
        raise RuntimeError(
            "ZAI_API_KEY environment variable is required. "
            "Get your key at https://z.ai/"
        )
    return key
