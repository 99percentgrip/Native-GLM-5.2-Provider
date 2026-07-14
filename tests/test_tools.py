"""Tests for glm_acp.tools — sandbox, gitignore, tool execution, ToolResult."""

import sys

import pytest

from glm_acp.tools import (
    Sandbox,
    ToolError,
    ToolResult,
    _command_environment,
    _is_ignored,
    _load_gitignore_patterns,
    execute_tool,
)


def test_command_environment_removes_inherited_credentials(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "provider-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setenv("DOCS_API_TOKEN", "docs-secret")
    monkeypatch.setenv("SAFE_SETTING", "visible")

    environment = _command_environment()

    assert environment["SAFE_SETTING"] == "visible"
    assert "ZAI_API_KEY" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert "DOCS_API_TOKEN" not in environment


class TestSandbox:
    def test_relative_path_resolves(self, tmp_path):
        sandbox = Sandbox(str(tmp_path))
        resolved = sandbox.resolve("test.py")
        assert resolved == tmp_path / "test.py"

    def test_absolute_path_inside_root(self, tmp_path):
        sandbox = Sandbox(str(tmp_path))
        resolved = sandbox.resolve(str(tmp_path / "src" / "main.py"))
        assert resolved == tmp_path / "src" / "main.py"

    def test_path_outside_root_blocked(self, tmp_path):
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="outside the workspace"):
            sandbox.resolve("/etc/passwd")

    def test_additional_dirs_allowed(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        sandbox = Sandbox(str(tmp_path), [str(other)])
        resolved = sandbox.resolve(str(other / "file.txt"))
        assert resolved == other / "file.txt"

    def test_symlink_escape_blocked(self, tmp_path):
        link = tmp_path / "escape"
        link.symlink_to("/etc")
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError):
            sandbox.resolve(str(link / "passwd"))


class TestGitignorePatterns:
    def test_direct_match(self):
        assert _is_ignored(".env", [".env"])

    def test_parent_directory_match(self):
        assert _is_ignored(".git/HEAD", [".git"])
        assert _is_ignored(".git/refs/heads/main", [".git"])

    def test_trailing_slash_pattern(self):
        assert _is_ignored(".venv/lib/x.py", [".venv/"])
        assert _is_ignored(".venv", [".venv/"])

    def test_wildcard_pattern(self):
        assert _is_ignored("app.egg-info/PKG-INFO", ["*.egg-info/"])

    def test_non_ignored_path(self):
        assert not _is_ignored("src/main.py", [".git", ".venv/"])
        assert not _is_ignored("README.md", ["*.pyc"])

    def test_node_modules(self):
        assert _is_ignored("node_modules/react/index.js", ["node_modules"])

    def test_load_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text(".venv/\n*.pyc\n# comment\n\nbuild/")
        patterns = _load_gitignore_patterns(tmp_path)
        assert ".venv/" in patterns
        assert "*.pyc" in patterns
        assert "build/" in patterns
        assert "# comment" not in patterns

    def test_load_gitignore_missing(self, tmp_path):
        assert _load_gitignore_patterns(tmp_path) == []


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("read_file", {"path": "test.txt"}, sandbox)
        assert result.output == "hello world"
        assert result.file_path is not None

    @pytest.mark.asyncio
    async def test_read_file_caps_default_output(self, tmp_path):
        from glm_acp.tools import MAX_TOOL_OUTPUT_CHARS

        (tmp_path / "large.txt").write_text("line\n" * (MAX_TOOL_OUTPUT_CHARS // 2))
        result = await execute_tool("read_file", {"path": "large.txt"}, Sandbox(str(tmp_path)))

        assert len(result.output) <= MAX_TOOL_OUTPUT_CHARS + 250
        assert "truncated" in result.output.lower()
        assert "start_line" in result.output

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path):
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="File not found"):
            await execute_tool("read_file", {"path": "nope.txt"}, sandbox)

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("write_file", {"path": "out.txt", "content": "data"}, sandbox)
        assert (tmp_path / "out.txt").read_text() == "data"
        assert result.file_path is not None
        assert result.new_text == "data"
        assert result.old_text is None  # new file

    @pytest.mark.asyncio
    async def test_write_file_overwrite(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool(
            "write_file", {"path": "existing.txt", "content": "new"}, sandbox
        )
        assert result.old_text == "old"
        assert result.new_text == "new"

    @pytest.mark.asyncio
    async def test_edit_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("old line\nnew line")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool(
            "edit_file",
            {
                "path": "code.py",
                "old_text": "old line",
                "new_text": "replaced",
            },
            sandbox,
        )
        assert f.read_text() == "replaced\nnew line"
        assert result.file_path is not None
        assert result.old_text == "old line"
        assert result.new_text == "replaced"

    @pytest.mark.asyncio
    async def test_edit_file_not_found_text(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello")
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="old_text not found"):
            await execute_tool(
                "edit_file",
                {
                    "path": "code.py",
                    "old_text": "missing",
                    "new_text": "x",
                },
                sandbox,
            )

    @pytest.mark.asyncio
    async def test_edit_file_ambiguous(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("dup\ndup")
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="appears 2 times"):
            await execute_tool(
                "edit_file",
                {
                    "path": "code.py",
                    "old_text": "dup",
                    "new_text": "x",
                },
                sandbox,
            )

    @pytest.mark.asyncio
    async def test_apply_patch(self, tmp_path):
        path = tmp_path / "code.py"
        path.write_text("one\ntwo\nthree\n")
        result = await execute_tool(
            "apply_patch",
            {
                "path": "code.py",
                "patch": "@@ -1,3 +1,3 @@\n one\n-two\n+second\n three\n",
            },
            Sandbox(str(tmp_path)),
        )
        assert path.read_text() == "one\nsecond\nthree\n"
        assert result.old_text == "one\ntwo\nthree\n"

    @pytest.mark.asyncio
    async def test_apply_patch_rejects_context_mismatch(self, tmp_path):
        (tmp_path / "code.py").write_text("actual\n")
        with pytest.raises(ToolError, match="context mismatch"):
            await execute_tool(
                "apply_patch",
                {"path": "code.py", "patch": "@@ -1 +1 @@\n-expected\n+new\n"},
                Sandbox(str(tmp_path)),
            )

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        (tmp_path / "file.py").write_text("x")
        (tmp_path / "subdir").mkdir()
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("list_directory", {"path": "."}, sandbox)
        assert "file.py" in result.output
        assert "dir subdir" in result.output

    @pytest.mark.asyncio
    async def test_grep(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\nimport sys")
        (tmp_path / "b.py").write_text("print('hello')")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("grep", {"pattern": "import"}, sandbox)
        assert "a.py" in result.output
        assert "import os" in result.output
        assert "b.py" not in result.output

    @pytest.mark.asyncio
    async def test_search_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("search_files", {"pattern": "*.py"}, sandbox)
        assert "a.py" in result.output
        assert "b.txt" not in result.output

    @pytest.mark.asyncio
    async def test_unknown_tool(self, tmp_path):
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="Unknown tool"):
            await execute_tool("nonexistent", {}, sandbox)


class TestToolResultStructure:
    """Verify ToolResult carries file path and diff info for ACP follow."""

    @pytest.mark.asyncio
    async def test_read_file_has_path(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("read_file", {"path": "main.py"}, sandbox)
        assert result.file_path is not None
        assert "main.py" in result.file_path
        assert result.old_text is None
        assert result.new_text is None

    @pytest.mark.asyncio
    async def test_write_file_has_diff(self, tmp_path):
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("write_file", {"path": "new.py", "content": "x = 1"}, sandbox)
        assert result.file_path is not None
        assert result.old_text is None  # new file
        assert result.new_text == "x = 1"

    @pytest.mark.asyncio
    async def test_write_file_overwrite_has_old(self, tmp_path):
        (tmp_path / "existing.py").write_text("old content")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool(
            "write_file", {"path": "existing.py", "content": "new"}, sandbox
        )
        assert result.old_text == "old content"
        assert result.new_text == "new"

    @pytest.mark.asyncio
    async def test_edit_file_has_diff(self, tmp_path):
        (tmp_path / "code.py").write_text("foo\nbar")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool(
            "edit_file",
            {
                "path": "code.py",
                "old_text": "foo",
                "new_text": "baz",
            },
            sandbox,
        )
        assert result.file_path is not None
        assert result.old_text == "foo"
        assert result.new_text == "baz"

    @pytest.mark.asyncio
    async def test_run_command_no_path(self, tmp_path):
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("run_command", {"command": "echo hello"}, sandbox)
        assert result.file_path is None
        assert result.old_text is None
        assert result.new_text is None

    @pytest.mark.asyncio
    async def test_run_command_reports_silent_failure(self, tmp_path):
        script = tmp_path / "fail.py"
        script.write_text("raise SystemExit(7)\n")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool(
            "run_command", {"command": f'"{sys.executable}" "{script}"'}, sandbox
        )
        assert "Exit code: 7" in result.output

    @pytest.mark.asyncio
    async def test_run_command_caps_output(self, tmp_path):
        from glm_acp.tools import MAX_TOOL_OUTPUT_CHARS

        script = tmp_path / "large_output.py"
        script.write_text("print('x' * 200000)\n")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool(
            "run_command",
            {"command": f'"{sys.executable}" "{script}"'},
            sandbox,
        )
        assert len(result.output) <= MAX_TOOL_OUTPUT_CHARS + 500
        assert "truncated" in result.output.lower()

    @pytest.mark.asyncio
    async def test_read_file_with_line(self, tmp_path):
        (tmp_path / "f.py").write_text("line1\nline2\nline3")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("read_file", {"path": "f.py", "start_line": 2}, sandbox)
        assert result.line == 2

    def test_tool_result_defaults(self):
        r = ToolResult(output="test")
        assert r.output == "test"
        assert r.file_path is None
        assert r.line is None
        assert r.old_text is None
        assert r.new_text is None


# ============================================================
# Edge cases: binary output, bad timeouts
# ============================================================


class TestToolEdgeCases:
    @pytest.mark.asyncio
    async def test_run_command_binary_output(self, tmp_path):
        """Command that outputs binary data should not crash."""
        sandbox = Sandbox(str(tmp_path))
        # printf outputs raw bytes including non-UTF8
        result = await execute_tool("run_command", {"command": "printf '\\xff\\xfe'"}, sandbox)
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_run_command_string_timeout_normalized(self, tmp_path):
        """Non-numeric timeout should fall back to default."""
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool(
            "run_command", {"command": "echo hi", "timeout": "not-a-number"}, sandbox
        )
        assert "hi" in result.output

    @pytest.mark.asyncio
    async def test_run_command_zero_timeout_normalized(self, tmp_path):
        """Zero timeout should fall back to default."""
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("run_command", {"command": "echo hi", "timeout": 0}, sandbox)
        assert "hi" in result.output

    @pytest.mark.asyncio
    async def test_run_command_negative_timeout_normalized(self, tmp_path):
        """Negative timeout should fall back to default."""
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("run_command", {"command": "echo hi", "timeout": -5}, sandbox)
        assert "hi" in result.output

    @pytest.mark.asyncio
    async def test_read_file_missing_path_key(self, tmp_path):
        """Missing path key should raise ToolError."""
        from glm_acp.tools import ToolError

        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError):
            await execute_tool("read_file", {}, sandbox)

    @pytest.mark.asyncio
    async def test_write_file_missing_content_key(self, tmp_path):
        """Missing content key should raise ToolError."""
        from glm_acp.tools import ToolError

        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError):
            await execute_tool("write_file", {"path": "x.py"}, sandbox)


# ============================================================
# Binary file robustness
# ============================================================


class TestBinaryFiles:
    @pytest.mark.asyncio
    async def test_read_file_normalizes_crlf(self, tmp_path):
        """Text reads preserve universal-newline behavior on every platform."""
        (tmp_path / "lines.txt").write_bytes(b"first\r\nsecond\r\n")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("read_file", {"path": "lines.txt"}, sandbox)
        assert result.output == "first\nsecond\n"

    @pytest.mark.asyncio
    async def test_read_file_binary(self, tmp_path):
        """Reading a binary file should give a clear error, not crash."""
        from glm_acp.tools import ToolError

        (tmp_path / "data.bin").write_bytes(b"\xff\xfe\x00\x01\x02\x03")
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="binary"):
            await execute_tool("read_file", {"path": "data.bin"}, sandbox)

    @pytest.mark.asyncio
    async def test_edit_file_binary(self, tmp_path):
        """Editing a binary file should give a clear error, not crash."""
        from glm_acp.tools import ToolError

        (tmp_path / "data.bin").write_bytes(b"\xff\xfe\x00\x01")
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="binary"):
            await execute_tool(
                "edit_file", {"path": "data.bin", "old_text": "a", "new_text": "b"}, sandbox
            )

    @pytest.mark.asyncio
    async def test_grep_skips_binary_files(self, tmp_path):
        """grep should skip binary files without crashing."""
        (tmp_path / "data.bin").write_bytes(b"\xff\xfe\x00\x01search")
        (tmp_path / "readable.py").write_text("search here\n")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("grep", {"pattern": "search"}, sandbox)
        assert "readable.py" in result.output
        assert "data.bin" not in result.output


# ============================================================
# Invalid regex / string line numbers
# ============================================================


class TestToolInputValidation:
    @pytest.mark.asyncio
    async def test_grep_invalid_regex(self, tmp_path):
        """Invalid regex should give a clear error, not crash."""
        from glm_acp.tools import ToolError

        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="Invalid regex"):
            await execute_tool("grep", {"pattern": "[unclosed"}, sandbox)

    @pytest.mark.asyncio
    async def test_read_file_string_start_line(self, tmp_path):
        """String start_line should be coerced to int, not crash."""
        (tmp_path / "f.py").write_text("line1\nline2\nline3")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("read_file", {"path": "f.py", "start_line": "2"}, sandbox)
        assert "line2" in result.output
        assert "line1" not in result.output

    @pytest.mark.asyncio
    async def test_read_file_start_beyond_file(self, tmp_path):
        """start_line beyond file length should return empty, not crash."""
        (tmp_path / "f.py").write_text("line1\nline2")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("read_file", {"path": "f.py", "start_line": 100}, sandbox)
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_read_file_line_zero(self, tmp_path):
        """start_line=0 should behave as 'from the beginning', not wrap negative."""
        (tmp_path / "f.py").write_text("line1\nline2\nline3")
        sandbox = Sandbox(str(tmp_path))
        result = await execute_tool("read_file", {"path": "f.py", "start_line": 0}, sandbox)
        assert "line1" in result.output

    @pytest.mark.asyncio
    async def test_edit_file_empty_old_text(self, tmp_path):
        """Empty old_text should give a clear error, not match everywhere."""
        from glm_acp.tools import ToolError

        (tmp_path / "f.py").write_text("hello world")
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="empty"):
            await execute_tool(
                "edit_file", {"path": "f.py", "old_text": "", "new_text": "x"}, sandbox
            )

    @pytest.mark.asyncio
    async def test_edit_file_missing_old_text_key(self, tmp_path):
        """Missing old_text key should default to empty and give clear error."""
        from glm_acp.tools import ToolError

        (tmp_path / "f.py").write_text("hello world")
        sandbox = Sandbox(str(tmp_path))
        with pytest.raises(ToolError, match="empty"):
            await execute_tool("edit_file", {"path": "f.py", "new_text": "x"}, sandbox)
