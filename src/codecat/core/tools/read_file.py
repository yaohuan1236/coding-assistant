"""read_file:读取指定文本文件的内容。

可恢复错误(文件不存在、路径不是文件、缺参数、解码失败)一律转成
ToolResult(is_error=True),让模型决定下一步,而不是炸穿 Agent Loop。
"""
from __future__ import annotations

from typing import Any

from codecat.core.types import ToolCall, ToolResult
from codecat.core.workspace import Workspace, WorkspaceError


class ReadFileTool:
    name = "read_file"
    description = "读取指定文件的文本内容"

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要读取的文件路径",
                        }
                    },
                    "required": ["path"],
                },
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        path = call.arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return ToolResult(
                tool_call_id=call.id,
                content="缺少必需参数: path",
                is_error=True,
            )

        try:
            target = self._workspace.resolve(path)
        except WorkspaceError as exc:
            return ToolResult(
                tool_call_id=call.id,
                content=str(exc),
                is_error=True,
            )

        try:
            if not target.exists():
                return ToolResult(
                    tool_call_id=call.id,
                    content=f"文件不存在: {path}",
                    is_error=True,
                )
            if not target.is_file():
                return ToolResult(
                    tool_call_id=call.id,
                    content=f"目标路径不是文件: {path}",
                    is_error=True,
                )
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult(
                tool_call_id=call.id,
                content=f"读取文件失败: {exc}",
                is_error=True,
            )

        return ToolResult(tool_call_id=call.id, content=content)
