from glm_acp.memory import append_memory, project_knowledge, read_memory


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
