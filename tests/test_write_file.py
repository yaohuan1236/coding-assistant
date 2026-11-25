"""write_file 测试:创建/覆盖/UTF-8/父目录/参数校验/Workspace 边界。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from codecat.core.tools import (
    DefaultToolRegistry,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from codecat.core.types import ToolCall
from codecat.core.workspace import Workspace


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _tool(root: Path) -> WriteFileTool:
    return WriteFileTool(Workspace(root))


def _write(tool: WriteFileTool, path: str, content: str) -> Any:
    call = ToolCall(id="call_001", name="write_file", arguments={"path": path, "content": content})
    return _run(tool.execute(call))


# ---------------------------------------------------------------------------
# 基本写入行为
# ---------------------------------------------------------------------------

def test_write_file_creates_new_file(tmp_path: Path) -> None:
    result = _write(_tool(tmp_path), "hello.txt", "Hello CodeCat")

    assert result.is_error is False
    assert result.content == "文件写入成功: hello.txt"
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "Hello CodeCat"


def test_write_file_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("old content", encoding="utf-8")

    result = _write(_tool(tmp_path), "hello.txt", "new content")

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "new content"  # 完整覆盖,不是追加


def test_write_file_allows_empty_content(tmp_path: Path) -> None:
    result = _write(_tool(tmp_path), "empty.txt", "")

    assert result.is_error is False
    assert "缺少必需参数" not in result.content  # 空串是合法内容,不是缺参数
    assert (tmp_path / "empty.txt").exists()
    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""


def test_write_file_roundtrips_utf8(tmp_path: Path) -> None:
    content = "你好,CodeCat!日本語 🐱"

    result = _write(_tool(tmp_path), "utf8.txt", content)

    assert result.is_error is False
    assert (tmp_path / "utf8.txt").read_text(encoding="utf-8") == content


def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    result = _write(_tool(tmp_path), "a/b/c/test.txt", "nested")

    assert result.is_error is False
    assert (tmp_path / "a" / "b" / "c" / "test.txt").read_text(encoding="utf-8") == "nested"


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def test_write_file_missing_path(tmp_path: Path) -> None:
    call = ToolCall(id="call_001", name="write_file", arguments={"content": "hello"})

    result = _run(_tool(tmp_path).execute(call))

    assert result.is_error is True
    assert "缺少必需参数: path" in result.content


def test_write_file_missing_content(tmp_path: Path) -> None:
    call = ToolCall(id="call_001", name="write_file", arguments={"path": "test.txt"})

    result = _run(_tool(tmp_path).execute(call))

    assert result.is_error is True
    assert "缺少必需参数: content" in result.content


# ---------------------------------------------------------------------------
# Workspace 边界
# ---------------------------------------------------------------------------

def test_write_file_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside = tmp_path / "outside.txt"

    result = _write(WriteFileTool(Workspace(workspace_root)), "../outside.txt", "x")

    assert result.is_error is True
    assert "路径超出 Workspace 范围" in result.content
    assert not outside.exists()  # 外部文件没有被创建


def test_write_file_rejects_multi_level_traversal(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside = tmp_path.parent / "outside_multi.txt"

    result = _write(WriteFileTool(Workspace(workspace_root)), "../../outside_multi.txt", "x")

    assert result.is_error is True
    assert not outside.exists()


def test_write_file_rejects_outside_absolute_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside = tmp_path / "outside_abs.txt"

    result = _write(WriteFileTool(Workspace(workspace_root)), str(outside), "x")

    assert result.is_error is True
    assert not outside.exists()


def test_write_file_rejects_sibling_prefix_directory(tmp_path: Path) -> None:
    # workspace-other 与 workspace 是兄弟目录,字符串前缀判断会误放行
    root = tmp_path / "workspace"
    root.mkdir()
    sibling = tmp_path / "workspace-other"
    sibling.mkdir()

    result = _write(WriteFileTool(Workspace(root)), str(sibling / "test.txt"), "x")

    assert result.is_error is True
    assert "路径超出 Workspace 范围" in result.content
    assert not (sibling / "test.txt").exists()


# ---------------------------------------------------------------------------
# 写入失败与 schema
# ---------------------------------------------------------------------------

def test_write_file_reports_os_error(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()

    result = _write(_tool(tmp_path), "adir", "x")

    assert result.is_error is True
    assert "写入文件失败" in result.content


def test_write_file_schema_is_openai_compatible(tmp_path: Path) -> None:
    schema = _tool(tmp_path).schema()

    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "write_file"
    assert function["description"] == WriteFileTool.description
    assert function["parameters"]["properties"]["path"]["type"] == "string"
    assert function["parameters"]["properties"]["content"]["type"] == "string"
    assert function["parameters"]["required"] == ["path", "content"]


def test_registry_exposes_all_file_tools(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    registry = DefaultToolRegistry()
    registry.register(ReadFileTool(workspace))
    registry.register(ListFilesTool(workspace))
    registry.register(WriteFileTool(workspace))

    names = [schema["function"]["name"] for schema in registry.schemas()]

    assert names == ["read_file", "list_files", "write_file"]
