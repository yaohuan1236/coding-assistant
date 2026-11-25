"""list_files:列出指定目录的直接子项(第一版不递归)。

可恢复错误(缺参数、目录不存在、路径不是目录、读取失败)一律转成
ToolResult(is_error=True),让模型决定下一步。
"""
from __future__ import annotations

from typing import Any

from codecat.core.types import ToolCall, ToolResult
from codecat.core.workspace import Workspace, WorkspaceError


class ListFilesTool:
    name = "list_files"
    description = "列出指定目录下的文件和子目录"

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
                            "description": "要查看的目录路径",
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
                    content=f"目录不存在: {path}",
                    is_error=True,
                )
            if not target.is_dir():
                return ToolResult(
                    tool_call_id=call.id,
                    content=f"目标路径不是目录: {path}",
                    is_error=True,
                )
            entries = sorted(target.iterdir(), key=lambda entry: entry.name)
        except OSError as exc:
            return ToolResult(
                tool_call_id=call.id,
                content=f"读取目录失败: {exc}",
                is_error=True,
            )

        if not entries:
            return ToolResult(tool_call_id=call.id, content="目录为空")

        lines = [
            f"[DIR] {entry.name}" if entry.is_dir() else f"[FILE] {entry.name}"
            for entry in entries
        ]
        return ToolResult(tool_call_id=call.id, content="\n".join(lines))
