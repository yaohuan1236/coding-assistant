"""ToolRegistry:工具的注册、schema 暴露与路由执行。

职责边界:
- 只做注册 / 查找 / 路由,不包含任何具体工具逻辑(无 if/elif 按名字分支)
- 不依赖 LLM 层(不 import LLMClient/LLMResponse),更不碰 httpx/Provider SDK
- 未知工具属于可恢复错误,转成 ToolResult(is_error=True) 返回给 loop
"""
from __future__ import annotations

from typing import Any

from codecat.core.tools.base import Tool
from codecat.core.types import ToolCall, ToolResult


class DefaultToolRegistry:
    """第一版注册表:显式注册、按 call.name 字典路由。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("工具必须有非空 name")
        if tool.name in self._tools:
            raise ValueError(f"工具已注册: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """返回全部已注册工具的 OpenAI 兼容 schema 列表。"""
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        """按名字查找并执行;未知工具返回 is_error 结果,不抛异常。"""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"未知工具: {call.name}",
                is_error=True,
            )
        return await tool.execute(call)
