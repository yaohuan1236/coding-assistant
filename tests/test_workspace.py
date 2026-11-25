"""Workspace 边界测试:路径解析规则、越界拒绝、Tool 集成。"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codecat.core.tools import ListFilesTool, ReadFileTool
from codecat.core.types import ToolCall
from codecat.core.workspace import Workspace, WorkspaceError


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="call_001", name=name, arguments=dict(arguments))


# ---------------------------------------------------------------------------
# 路径解析规则(任务书第十五节)
# ---------------------------------------------------------------------------

def test_resolve_dot_returns_root(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    assert workspace.resolve(".") == tmp_path.resolve()


def test_resolve_relative_path(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    workspace = Workspace(tmp_path)

    assert workspace.resolve("src") == (tmp_path / "src").resolve()


def test_resolve_nested_path(tmp_path: Path) -> None:
    (tmp_path / "src" / "codecat").mkdir(parents=True)
    workspace = Workspace(tmp_path)

    assert workspace.resolve("src/codecat") == (tmp_path / "src" / "codecat").resolve()


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError):
        workspace.resolve("../")


def test_multi_parent_traversal_rejected(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError):
        workspace.resolve("../../README.md")


def test_dotdot_even_if_lands_back_inside_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError):
        workspace.resolve("src/../src")


def test_absolute_path_outside_is_rejected(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError):
        workspace.resolve(str(tmp_path.parent / "outside.txt"))


def test_absolute_path_inside_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    workspace = Workspace(tmp_path)

    assert workspace.resolve(str(tmp_path / "src")) == (tmp_path / "src").resolve()


def test_sibling_prefix_directory_is_rejected(tmp_path: Path) -> None:
    # workspace 与 workspace-other 是兄弟目录,字符串前缀判断会误放行
    root = tmp_path / "workspace"
    root.mkdir()
    sibling = tmp_path / "workspace-other"
    sibling.mkdir()
    workspace = Workspace(root)

    with pytest.raises(WorkspaceError):
        workspace.resolve(str(sibling))


# ---------------------------------------------------------------------------
# Tool 集成:越界统一转成 is_error 结果(任务书第十六、十七节)
# ---------------------------------------------------------------------------

def test_read_file_relative_path_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))

    result = asyncio.run(tool.execute(_call("read_file", path="hello.txt")))

    assert result.is_error is False
    assert result.content == "hi"


def test_read_file_rejects_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("SECRET", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))

    result = asyncio.run(tool.execute(_call("read_file", path=str(outside))))

    assert result.is_error is True
    assert "路径超出 Workspace 范围" in result.content
    assert "SECRET" not in result.content  # 没有读到外部文件内容


def test_read_file_rejects_parent_traversal(tmp_path: Path) -> None:
    (tmp_path.parent / "parent_secret.txt").write_text("SECRET", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))

    result = asyncio.run(tool.execute(_call("read_file", path="../parent_secret.txt")))

    assert result.is_error is True
    assert "路径超出 Workspace 范围" in result.content


def test_list_files_dot_lists_workspace_root(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    tool = ListFilesTool(Workspace(tmp_path))

    result = asyncio.run(tool.execute(_call("list_files", path=".")))

    assert result.is_error is False
    assert "[FILE] a.txt" in result.content


def test_list_files_rejects_parent_traversal(tmp_path: Path) -> None:
    tool = ListFilesTool(Workspace(tmp_path))

    result = asyncio.run(tool.execute(_call("list_files", path="../")))

    assert result.is_error is True
    assert "路径超出 Workspace 范围" in result.content


def test_list_files_rejects_outside_absolute_path(tmp_path: Path) -> None:
    tool = ListFilesTool(Workspace(tmp_path))

    result = asyncio.run(tool.execute(_call("list_files", path=str(tmp_path.parent))))

    assert result.is_error is True
    assert "路径超出 Workspace 范围" in result.content
