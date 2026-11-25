"""工具系统第一版测试:Tool 接口 / DefaultToolRegistry / ReadFileTool / 端到端。"""
from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any, Sequence

import pytest

from codecat.core.loop import AgentLoop, ToolRegistry
from codecat.core.tools import DefaultToolRegistry, ReadFileTool, Tool
from codecat.core.tools import registry as registry_module
from codecat.core.types import LLMResponse, Message, ToolCall, ToolResult
from codecat.core.workspace import Workspace


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _call(name: str = "read_file", **arguments: Any) -> ToolCall:
    return ToolCall(id="call_001", name=name, arguments=arguments)


class EchoTool:
    """测试用最小工具:回显自己的名字。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} 测试工具"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(tool_call_id=call.id, content=f"{self.name} executed")


# ---------------------------------------------------------------------------
# read_file 单元测试
# ---------------------------------------------------------------------------

def test_read_file_returns_content(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("CodeCat 是一个轻量编程助手。", encoding="utf-8")

    result = _run(ReadFileTool(Workspace(tmp_path)).execute(_call(path=str(target))))

    assert result.is_error is False
    assert result.tool_call_id == "call_001"
    assert result.content == "CodeCat 是一个轻量编程助手。"


def test_read_file_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "not-exist.txt"

    result = _run(ReadFileTool(Workspace(tmp_path)).execute(_call(path=str(missing))))

    assert result.is_error is True
    assert "文件不存在" in result.content


def test_read_file_missing_path_argument(tmp_path: Path) -> None:
    result = _run(ReadFileTool(Workspace(tmp_path)).execute(_call()))

    assert result.is_error is True
    assert "缺少必需参数: path" in result.content


def test_read_file_directory_is_not_a_file(tmp_path: Path) -> None:
    result = _run(ReadFileTool(Workspace(tmp_path)).execute(_call(path=str(tmp_path))))

    assert result.is_error is True
    assert "目标路径不是文件" in result.content


def test_read_file_undecodable_content(tmp_path: Path) -> None:
    target = tmp_path / "binary.dat"
    target.write_bytes(b"\xff\xfe\x00\x01")

    result = _run(ReadFileTool(Workspace(tmp_path)).execute(_call(path=str(target))))

    assert result.is_error is True
    assert "读取文件失败" in result.content


def test_read_file_schema_is_openai_compatible(tmp_path: Path) -> None:
    schema = ReadFileTool(Workspace(tmp_path)).schema()

    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "read_file"
    assert function["description"] == ReadFileTool.description
    assert function["parameters"]["properties"]["path"]["type"] == "string"
    assert function["parameters"]["required"] == ["path"]


# ---------------------------------------------------------------------------
# DefaultToolRegistry 测试
# ---------------------------------------------------------------------------

def test_registry_schemas_exposes_registered_tools(tmp_path: Path) -> None:
    registry = DefaultToolRegistry()
    registry.register(ReadFileTool(Workspace(tmp_path)))

    schemas = registry.schemas()

    assert [schema["function"]["name"] for schema in schemas] == ["read_file"]


def test_registry_routes_by_tool_name() -> None:
    registry = DefaultToolRegistry()
    registry.register(EchoTool("tool_a"))
    registry.register(EchoTool("tool_b"))

    result_a = _run(registry.execute(_call("tool_a")))
    result_b = _run(registry.execute(_call("tool_b")))

    assert result_a.content == "tool_a executed"
    assert result_b.content == "tool_b executed"


def test_registry_unknown_tool_returns_error_result() -> None:
    registry = DefaultToolRegistry()

    result = _run(registry.execute(_call("unknown_tool")))

    assert result.is_error is True
    assert result.tool_call_id == "call_001"
    assert "未知工具" in result.content


def test_registry_rejects_duplicate_names() -> None:
    registry = DefaultToolRegistry()
    registry.register(EchoTool("dup"))

    with pytest.raises(ValueError, match="已注册"):
        registry.register(EchoTool("dup"))


def test_registry_and_tool_satisfy_protocols(tmp_path: Path) -> None:
    assert isinstance(DefaultToolRegistry(), ToolRegistry)  # loop 依赖的协议
    assert isinstance(ReadFileTool(Workspace(tmp_path)), Tool)
    assert not isinstance(object(), Tool)


def test_registry_does_not_depend_on_llm_layer() -> None:
    tree = ast.parse(inspect.getsource(registry_module))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    assert not any("core.llm" in name or "httpx" in name for name in imported)


# ---------------------------------------------------------------------------
# 端到端:AgentLoop + 真实 Registry + 真实 ReadFileTool
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """按预设序列返回响应,并记录每次请求的消息与 tools。"""

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


def test_end_to_end_agent_reads_file_through_registry(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("CodeCat 是一个面向中文开发者的轻量编程助手。", encoding="utf-8")

    tool_call = ToolCall(id="call_001", name="read_file", arguments={"path": str(target)})
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=[tool_call]),
            LLMResponse(content="这个项目是一个面向中文开发者的轻量编程助手。"),
        ]
    )
    registry = DefaultToolRegistry()
    registry.register(ReadFileTool(Workspace(tmp_path)))
    agent = AgentLoop(llm, registry, max_steps=3)

    answer = _run(agent.run("请读取 REQ.md 并告诉我它的内容。"))

    # 1) 第一次请求就把 read_file 的 schema 带给了 LLM
    assert llm.tools_seen[0] is not None
    assert [tool["function"]["name"] for tool in llm.tools_seen[0]] == ["read_file"]
    # 2) 历史完整:第二次请求里有 assistant(tool_calls) + tool(文件内容)
    assert [m.role for m in llm.requests[1]] == ["user", "assistant", "tool"]
    tool_message = llm.requests[1][2]
    assert tool_message.tool_call_id == "call_001"
    assert "轻量编程助手" in tool_message.content  # 真实读到的文件内容
    # 3) 最终回答基于文件内容
    assert answer == "这个项目是一个面向中文开发者的轻量编程助手。"
