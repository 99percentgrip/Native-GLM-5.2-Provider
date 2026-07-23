"""Tests for glm_acp.config — model registry, plans, thought levels."""

import os

import pytest

from glm_acp.config import (
    API_ENDPOINTS,
    CONFIG_DIR_ENV,
    CONTEXT_WINDOW_TOKENS,
    DEFAULT_API_ENDPOINT,
    DEFAULT_MODEL,
    DESTRUCTIVE_TOOLS,
    GENERATION_PROFILES,
    MAX_RETRIES,
    MODELS,
    RETRYABLE_STATUS_CODES,
    VISION_MODELS,
    credentials_path,
    get_api_key,
    has_api_key,
    load_stored_api_key,
    models_for_plan,
    store_api_key,
    thought_levels_for_model,
)


class TestModelRegistry:
    def test_all_models_have_context_window(self):
        for model_id in MODELS:
            assert model_id in CONTEXT_WINDOW_TOKENS, (
                f"{model_id} missing from CONTEXT_WINDOW_TOKENS"
            )

    def test_all_models_have_plans(self):
        for model_id, info in MODELS.items():
            assert "plans" in info, f"{model_id} missing 'plans' key"
            assert len(info["plans"]) > 0, f"{model_id} has no plans"

    def test_vision_models_flagged(self):
        assert VISION_MODELS == frozenset({"glm-5v-turbo", "glm-4.5v", "glm-4.6v"})

    def test_default_model_exists(self):
        assert DEFAULT_MODEL in MODELS

    def test_context_window_sizes(self):
        assert CONTEXT_WINDOW_TOKENS["glm-5.2"] == 1_000_000
        assert CONTEXT_WINDOW_TOKENS["glm-5-turbo"] == 200_000
        assert CONTEXT_WINDOW_TOKENS["glm-4.7"] == 200_000
        assert CONTEXT_WINDOW_TOKENS["glm-5v-turbo"] == 200_000
        assert CONTEXT_WINDOW_TOKENS["glm-4.5v"] == 65_536


class TestPlanModelSync:
    def test_coding_plan_excludes_vision(self):
        models = models_for_plan("coding")
        assert len(models) == 3
        assert "glm-4.5v" not in models
        assert "glm-4.6v" not in models

    def test_standard_plan_includes_vision(self):
        models = models_for_plan("standard")
        assert len(models) == 6
        assert "glm-5v-turbo" in models
        assert "glm-4.5v" in models
        assert "glm-4.6v" in models

    def test_bigmodel_plan_includes_vision(self):
        models = models_for_plan("bigmodel")
        assert len(models) == 6

    def test_default_plan_is_coding(self):
        assert DEFAULT_API_ENDPOINT == "coding"


class TestThoughtLevels:
    def test_glm52_has_all_levels(self):
        levels = thought_levels_for_model("glm-5.2")
        assert len(levels) == 4
        assert set(levels.keys()) == {"disabled", "enabled", "high", "max"}

    def test_glm47_excludes_deep(self):
        levels = thought_levels_for_model("glm-4.7")
        assert len(levels) == 2
        assert set(levels.keys()) == {"disabled", "enabled"}

    def test_vision_model_supports_standard_thinking(self):
        levels = thought_levels_for_model("glm-4.5v")
        assert set(levels) == {"disabled", "enabled"}

    def test_vision_model_supports_standard_thinking_46v(self):
        levels = thought_levels_for_model("glm-4.6v")
        assert set(levels) == {"disabled", "enabled"}


class TestConstants:
    def test_destructive_tools(self):
        assert "write_file" in DESTRUCTIVE_TOOLS
        assert "edit_file" in DESTRUCTIVE_TOOLS
        assert "run_command" in DESTRUCTIVE_TOOLS
        assert "apply_patch" in DESTRUCTIVE_TOOLS
        assert "apply_patch_set" in DESTRUCTIVE_TOOLS
        assert "learn_skill" in DESTRUCTIVE_TOOLS
        assert "forget_skill" in DESTRUCTIVE_TOOLS
        assert "store_user_profile" in DESTRUCTIVE_TOOLS
        assert "forget_memory" in DESTRUCTIVE_TOOLS
        assert "manage_skill" in DESTRUCTIVE_TOOLS
        assert "curate_skills" in DESTRUCTIVE_TOOLS
        assert "manage_skill_bundle" in DESTRUCTIVE_TOOLS
        assert "evolve_skill" in DESTRUCTIVE_TOOLS
        assert "delegate_task" in DESTRUCTIVE_TOOLS
        assert "vision_analyze" in DESTRUCTIVE_TOOLS
        assert "browser_ui" in DESTRUCTIVE_TOOLS
        assert "read_file" not in DESTRUCTIVE_TOOLS

    def test_retry_config(self):
        assert MAX_RETRIES == 3
        assert 429 in RETRYABLE_STATUS_CODES
        assert 500 in RETRYABLE_STATUS_CODES
        assert 400 not in RETRYABLE_STATUS_CODES

    def test_api_endpoints_have_urls(self):
        for endpoint_id, info in API_ENDPOINTS.items():
            assert "base_url" in info
            assert info["base_url"].startswith("https://")

    def test_generation_profiles_adjust_one_sampling_control(self):
        assert GENERATION_PROFILES["balanced"]["temperature"] is None
        assert GENERATION_PROFILES["balanced"]["top_p"] is None
        for profile in ("precise", "exploratory"):
            info = GENERATION_PROFILES[profile]
            assert sum(info[key] is not None for key in ("temperature", "top_p")) == 1


class TestApiKey:
    def test_missing_key_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("Z_AI_API_KEY", raising=False)
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
        with pytest.raises(RuntimeError, match="ZAI_API_KEY"):
            get_api_key()

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "test-key-123")
        assert get_api_key() == "test-key-123"

    def test_key_from_alt_env(self, monkeypatch):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.setenv("Z_AI_API_KEY", "alt-key-456")
        assert get_api_key() == "alt-key-456"

    def test_stored_key_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("Z_AI_API_KEY", raising=False)
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))

        path = store_api_key("  stored-secret  ")

        assert path == credentials_path()
        assert load_stored_api_key() == "stored-secret"
        assert get_api_key() == "stored-secret"
        assert has_api_key() is True
        if os.name != "nt":
            assert path.stat().st_mode & 0o077 == 0

    def test_environment_key_precedes_stored_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
        store_api_key("stored-secret")
        monkeypatch.setenv("ZAI_API_KEY", "environment-secret")
        assert get_api_key() == "environment-secret"

    def test_invalid_stored_state_is_ignored(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("Z_AI_API_KEY", raising=False)
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
        credentials_path().write_text("not-json", encoding="utf-8")
        assert load_stored_api_key() is None
        assert has_api_key() is False

    def test_empty_key_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
        with pytest.raises(ValueError, match="cannot be empty"):
            store_api_key("   ")
