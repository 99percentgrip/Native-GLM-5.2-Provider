"""Tests for glm_acp.session_store — persistence, path traversal, atomic writes."""

import json
import pytest
from pathlib import Path

os = __import__("os")
os.environ.setdefault("ZAI_API_KEY", "test-key")

from glm_acp.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(base_dir=tmp_path)


@pytest.fixture
def sample_data():
    return {
        "version": 1,
        "cwd": "/home/user/project",
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": "hello"}],
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "estimated_tokens": 150,
    }


# ============================================================
# Save / Load round-trip
# ============================================================

class TestSaveLoad:
    def test_save_and_load(self, store, sample_data):
        store.save("session-1", sample_data)
        loaded = store.load("session-1")
        assert loaded is not None
        assert loaded["cwd"] == "/home/user/project"
        assert loaded["model"] == "glm-5.2"
        assert loaded["messages"][0]["content"] == "hello"
        assert loaded["estimated_tokens"] == 150

    def test_save_injects_timestamp(self, store, sample_data):
        store.save("session-1", sample_data)
        loaded = store.load("session-1")
        assert "saved_at" in loaded

    def test_load_nonexistent(self, store):
        assert store.load("nonexistent") is None

    def test_overwrite(self, store, sample_data):
        store.save("session-1", sample_data)
        modified = {**sample_data, "model": "glm-4.7"}
        store.save("session-1", modified)
        loaded = store.load("session-1")
        assert loaded["model"] == "glm-4.7"


# ============================================================
# Path traversal protection
# ============================================================

class TestPathTraversal:
    def test_traversal_attempt_sanitized(self, store, sample_data):
        """Session IDs with path separators must not escape the base dir."""
        store.save("../../../etc/passwd", sample_data)
        # The file should be inside the store base dir, not at /etc/
        malicious_path = Path("/etc/passwd.json")
        assert not malicious_path.exists()

    def test_traversal_load_finds_sanitized(self, store, sample_data):
        """Loading with traversal chars should find the sanitized file."""
        store.save("../../etc/passwd", sample_data)
        loaded = store.load("../../etc/passwd")
        # Should load successfully — same sanitization applied to both
        assert loaded is not None
        assert loaded["model"] == "glm-5.2"

    def test_dot_dot_slash_in_session_id(self, store, sample_data):
        store.save("..%2F..%2Fevil", sample_data)
        # Verify file exists within base_dir
        files = list(store._base_dir.glob("*.json"))
        assert len(files) == 1
        # The sanitized name should be inside base_dir
        assert files[0].parent == store._base_dir

    def test_slash_in_session_id(self, store, sample_data):
        store.save("nested/path/session", sample_data)
        loaded = store.load("nested/path/session")
        assert loaded is not None


# ============================================================
# List sessions
# ============================================================

class TestListSessions:
    def test_list_empty(self, store):
        assert store.list() == []

    def test_list_returns_metadata(self, store, sample_data):
        store.save("s1", {**sample_data, "title": "First chat"})
        store.save("s2", {**sample_data, "title": "Second chat"})
        sessions = store.list()
        assert len(sessions) == 2
        titles = {s["title"] for s in sessions}
        assert "First chat" in titles
        assert "Second chat" in titles

    def test_list_sorted_by_recency(self, store, sample_data):
        import time
        store.save("old", {**sample_data, "title": "Old"})
        time.sleep(0.05)
        store.save("new", {**sample_data, "title": "New"})
        sessions = store.list()
        assert sessions[0]["title"] == "New"
        assert sessions[1]["title"] == "Old"

    def test_list_handles_corrupted_json(self, store, sample_data):
        """Corrupted session files should be silently skipped."""
        store.save("good", {**sample_data, "title": "Good"})
        # Write garbage
        (store._base_dir / "corrupt.json").write_text("not valid json{{{")
        sessions = store.list()
        assert len(sessions) == 1
        assert sessions[0]["title"] == "Good"


# ============================================================
# Delete sessions
# ============================================================

class TestDeleteSession:
    def test_delete_existing(self, store, sample_data):
        store.save("session-1", sample_data)
        store.delete("session-1")
        assert store.load("session-1") is None

    def test_delete_nonexistent_no_error(self, store):
        store.delete("nonexistent")  # should not raise


# ============================================================
# Corrupted data handling
# ============================================================

class TestCorruptedData:
    def test_load_corrupted_json(self, store):
        """Loading corrupted JSON should return None, not crash."""
        store._base_dir.mkdir(parents=True, exist_ok=True)
        (store._base_dir / "bad.json").write_text("not valid json{")
        assert store.load("bad") is None
