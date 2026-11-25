"""list_files 单元测试 + Runtime 多工具回归。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence

from codecat.core.loop import AgentLoop
from codecat.core.tools import DefaultToolRegistry, ListFilesTool, ReadFileTool
from codecat.core.types import LLMResponse, Message, ToolCall
from codecat.core.workspace import Workspace


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _call(**arguments: Any) -> ToolCall:
    return ToolCall(id="call_001", name="list_files", arguments=arguments)


# ---------------------------------------------------------------------------
# 单元测试(任务书第十九节)
# ---------------------------------------------------------------------------

def test_list_files_returns_sorted_entries(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    result = _run(ListFilesTool(Workspace(tmp_path)).execute(_call(path=str(tmp_path))))

    assert result.is_error is False
    assert result.content.splitlines() == ["[FILE] a.txt", "[FILE] b.py", "[DIR] subdir"]


def test_list_files_missing_directory(tmp_path: Path) -> None:
    result = _run(ListFilesTool(Workspace(tmp_path)).execute(_call(path=str(tmp_path / "not_exist"))))

    assert result.is_error is True
    assert "目录不存在" in result.content


def test_list_files_path_is_file(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")

    result = _run(ListFilesTool(Workspace(tmp_path)).execute(_call(path=str(target))))

    assert result.is_error is True
    assert "不是目录" in result.content


def test_list_files_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _run(ListFilesTool(Workspace(tmp_path)).execute(_call(path=str(empty))))

    assert result.is_error is False
    assert result.content == "目录为空"


def test_list_files_missing_path_argument(tmp_path: Path) -> None:
    result = _run(ListFilesTool(Workspace(tmp_path)).execute(_call()))

    assert result.is_error is True
    assert "缺少必需参数: path" in result.content


def test_list_files_schema_is_openai_compatible(tmp_path: Path) -> None:
    schema = ListFilesTool(Workspace(tmp_path)).schema()

    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "list_files"
    assert function["description"] == ListFilesTool.description
    assert function["parameters"]["properties"]["path"]["type"] == "string"
    assert function["parameters"]["required"] == ["path"]


# ---------------------------------------------------------------------------
# Runtime 回归:多 ToolCall 串行执行(read_file + list_files)
# ---------------------------------------------------------------------------

class ScriptedLLM:
    def __init__(self, script: Sequence[LLMResponse | Exception]) -> None:
        self._script = list(script)
        self.requests: list[list[Message]] = []
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.requests.append(list(messages))
        self.tools_seen.append(tools)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_runtime_runs_read_file_and_list_files_in_one_step(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("内容", encoding="utf-8")
    read_call = ToolCall(id="c1", name="read_file", arguments={"path": str(tmp_path / "a.txt")})
    list_call = ToolCall(id="c2", name="list_files", arguments={"path": str(tmp_path)})
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[read_call, list_call]),
            LLMResponse(content="两个工具都完成了"),
        ]
    )
    registry = DefaultToolRegistry()
    registry.register(ReadFileTool(Workspace(tmp_path)))
    registry.register(ListFilesTool(Workspace(tmp_path)))
    agent = AgentLoop(llm, registry, max_steps=3)

    answer = _run(agent.run("先读文件再列目录"))

    assert answer == "两个工具都完成了"
    # 两个工具都暴露给了 LLM
    assert [tool["function"]["name"] for tool in llm.tools_seen[0]] == [
        "read_file",
        "list_files",
    ]
    # 串行执行后,历史里两条 tool 消息各自对应一次调用
    assert [m.role for m in llm.requests[1]] == ["user", "assistant", "tool", "tool"]
    assert "内容" in llm.requests[1][2].content
    assert "[FILE] a.txt" in llm.requests[1][3].content
