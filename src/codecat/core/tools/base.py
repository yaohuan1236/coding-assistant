"""单个 Tool 的统一接口。

原则:
- Tool 自己提供 name/description/schema,自己执行;Registry 不碰具体工具逻辑
- Tool 只返回 ToolResult;Message 的构造由 loop 负责(边界不可越过)
- 可恢复错误由 Tool 自己转成 ToolResult(is_error=True)
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from codecat.core.types import ToolCall, ToolResult


@runtime_checkable
class Tool(Protocol):
    """Agent 可调用工具的最小契约。实现方无需继承。"""

    name: str
    description: str

    def schema(self) -> dict[str, Any]:
        """返回 OpenAI 兼容的单工具 schema(含 name/description/parameters)。"""
        ...

    async def execute(self, call: ToolCall) -> ToolResult:
        """执行一次调用。可恢复错误必须转成 ToolResult(is_error=True)。"""
        ...
