import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from glm_acp.memory import (
    append_memory,
    append_user_profile,
    curate_learned_skills,
    forget_learned_skill,
    forget_memory,
    forget_user_profile,
    list_learned_skills,
    manage_learned_skill,
    project_knowledge,
    read_learned_skill,
    read_memory,
    read_user_profile,
    skill_curator_status,
    write_learned_skill,
)


def test_project_knowledge_loads_instructions_and_memory(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use small diffs.")
    path = append_memory(str(tmp_path), "Tests use pytest")
    assert path == tmp_path / ".glm-acp" / "memory.md"
    knowledge = project_knowledge(str(tmp_path))
    assert "Use small diffs." in knowledge
    assert "Tests use pytest" in knowledge


def test_memory_is_deduplicated(tmp_path):
    append_memory(str(tmp_path), "  Stable   fact ")
    append_memory(str(tmp_path), "Stable fact")
    assert read_memory(str(tmp_path)).count("Stable fact") == 1


def test_missing_memory_is_explicit(tmp_path):
    assert "No durable" in read_memory(str(tmp_path))


def test_project_skills_are_discovered_without_loading_full_body(tmp_path):
    skill = tmp_path / ".agents" / "skills" / "review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: careful-review\ndescription: Review risky patches\n---\nSECRET BODY\n"
    )
    knowledge = project_knowledge(str(tmp_path))
    assert "careful-review" in knowledge
    assert "Review risky patches" in knowledge
    assert "SECRET BODY" not in knowledge


def test_project_memory_symlink_cannot_escape_workspace(tmp_path):
    outside = tmp_path.parent / "outside-memory.md"
    outside.write_text("outside secret")
    memory_dir = tmp_path / ".glm-acp"
    memory_dir.mkdir()
    (memory_dir / "memory.md").symlink_to(outside)
    assert "outside secret" not in project_knowledge(str(tmp_path))
    assert "outside secret" not in read_memory(str(tmp_path))


def test_verified_skill_round_trip_and_progressive_discovery(tmp_path):
    path = write_learned_skill(
        str(tmp_path),
        "Fix Async Cleanup",
        "Repair async resources when cleanup tests fail",
        "Run the focused cleanup test. Define close before calling it in finally.",
    )
    assert path == tmp_path / ".glm-acp" / "skills" / "fix-async-cleanup" / "SKILL.md"
    assert read_learned_skill(str(tmp_path), "fix-async-cleanup").startswith("---\n")
    skills = list_learned_skills(str(tmp_path))
    assert skills[0]["name"] == "fix-async-cleanup"
    assert skills[0]["description"] == "Repair async resources when cleanup tests fail"
    assert skills[0]["path"] == ".glm-acp/skills/fix-async-cleanup/SKILL.md"
    assert skills[0]["state"] == "active"
    assert skills[0]["use_count"] == 1
    assert skills[0]["revision_count"] == 1
    knowledge = project_knowledge(str(tmp_path))
    assert "fix-async-cleanup" in knowledge
    assert "Run the focused cleanup test" not in knowledge


def test_learning_rejects_credentials(tmp_path):
    with pytest.raises(ValueError, match="credential or secret"):
        write_learned_skill(
            str(tmp_path),
            "unsafe",
            "Use API access",
            "Set api_key=super-secret-value before running the tool.",
        )
    with pytest.raises(ValueError, match="credential or secret"):
        append_memory(str(tmp_path), "token=super-secret-value")


def test_forget_removes_only_agent_learned_skill(tmp_path):
    user_skill = tmp_path / ".agents" / "skills" / "keep" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text("---\nname: keep\ndescription: Keep me\n---\n")
    learned = write_learned_skill(str(tmp_path), "remove-me", "Temporary", "Do this.")

    removed = forget_learned_skill(str(tmp_path), "remove-me")

    assert removed == learned
    assert not learned.exists()
    assert user_skill.exists()


def test_learned_skill_symlink_cannot_escape_workspace(tmp_path):
    outside = tmp_path.parent / "outside-skills"
    outside.mkdir(exist_ok=True)
    learning_dir = tmp_path / ".glm-acp"
    learning_dir.mkdir()
    (learning_dir / "skills").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the workspace"):
        write_learned_skill(str(tmp_path), "unsafe", "Unsafe", "Do not write outside.")
    assert list_learned_skills(str(tmp_path)) == []
    assert not list(outside.iterdir())


def test_project_memory_can_forget_exact_entry(tmp_path):
    append_memory(str(tmp_path), "Keep this")
    append_memory(str(tmp_path), "Remove this")

    forget_memory(str(tmp_path), "Remove this")

    assert "Keep this" in read_memory(str(tmp_path))
    assert "Remove this" not in read_memory(str(tmp_path))


def test_private_user_profile_round_trip_and_forget(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_ACP_CONFIG_DIR", str(tmp_path / "private-config"))
    path = append_user_profile("Prefers concise reports", "preference")
    append_user_profile("Prefers concise reports", "preference")

    assert read_user_profile().count("Prefers concise reports") == 1
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600

    forget_user_profile("Prefers concise reports")
    assert "No durable" in read_user_profile()


def test_user_profile_rejects_sensitive_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_ACP_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="credential or secret"):
        append_user_profile("password=not-for-memory", "environment")


def test_user_profile_symlink_is_not_read_or_written(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    outside = tmp_path / "outside-user.md"
    outside.write_text("private external content")
    (config / "user.md").symlink_to(outside)
    monkeypatch.setenv("GLM_ACP_CONFIG_DIR", str(config))

    assert "private external content" not in read_user_profile()
    with pytest.raises(ValueError, match="unsafe"):
        append_user_profile("New preference", "preference")
    assert outside.read_text() == "private external content"


def test_skill_refinement_tracks_revisions_and_usage(tmp_path):
    write_learned_skill(str(tmp_path), "review", "Review changes", "Run tests.")
    read_learned_skill(str(tmp_path), "review")
    write_learned_skill(
        str(tmp_path), "review", "Review changes safely", "Run focused tests, then inspect diff."
    )

    skill = list_learned_skills(str(tmp_path))[0]
    assert skill["revision_count"] == 2
    assert skill["use_count"] == 1
    assert skill["last_used_at"]


def test_skill_lifecycle_pin_archive_and_restore(tmp_path):
    write_learned_skill(str(tmp_path), "safe-release", "Release safely", "Run checks.")
    manage_learned_skill(str(tmp_path), "safe-release", "pin")
    with pytest.raises(ValueError, match="Pinned skills"):
        manage_learned_skill(str(tmp_path), "safe-release", "archive")

    manage_learned_skill(str(tmp_path), "safe-release", "unpin")
    manage_learned_skill(str(tmp_path), "safe-release", "archive")
    assert skill_curator_status(str(tmp_path))["archived"] == 1
    assert not (tmp_path / ".glm-acp/skills/safe-release/SKILL.md").exists()

    manage_learned_skill(str(tmp_path), "safe-release", "restore")
    assert (tmp_path / ".glm-acp/skills/safe-release/SKILL.md").is_file()
    assert skill_curator_status(str(tmp_path))["active"] == 1


def test_curator_marks_stale_then_archives_without_deleting(tmp_path):
    write_learned_skill(str(tmp_path), "old-workflow", "Old workflow", "Follow the workflow.")
    usage_path = tmp_path / ".glm-acp/skills/.usage.json"
    payload = json.loads(usage_path.read_text())
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    payload["skills"]["old-workflow"]["last_used_at"] = (now - timedelta(days=31)).isoformat()
    usage_path.write_text(json.dumps(payload))

    stale = curate_learned_skills(str(tmp_path), now=now)
    assert stale["stale"] == ["old-workflow"]
    assert skill_curator_status(str(tmp_path))["stale"] == 1
    read_learned_skill(str(tmp_path), "old-workflow")
    assert skill_curator_status(str(tmp_path))["active"] == 1

    payload = json.loads(usage_path.read_text())
    payload["skills"]["old-workflow"]["last_used_at"] = (now - timedelta(days=91)).isoformat()
    usage_path.write_text(json.dumps(payload))
    archived = curate_learned_skills(str(tmp_path), now=now)

    assert archived["archived"] == ["old-workflow"]
    archived_path = tmp_path / ".glm-acp/skills/.archive/old-workflow/SKILL.md"
    assert archived_path.is_file()


def test_curator_detects_manual_drift_and_description_overlap(tmp_path):
    first = write_learned_skill(
        str(tmp_path),
        "deploy-api",
        "Deploy kubernetes service with verified checks",
        "Run the deployment checks.",
    )
    write_learned_skill(
        str(tmp_path),
        "release-api",
        "Deploy kubernetes service with release checks",
        "Run the release checks.",
    )
    first.write_text(first.read_text() + "\nManual correction.\n")

    status = skill_curator_status(str(tmp_path))

    assert status["drifted"] == ["deploy-api"]
    assert status["overlap_candidates"][0]["skills"] == ["deploy-api", "release-api"]
    curated = curate_learned_skills(str(tmp_path))
    assert curated["review"] == ["deploy-api"]
