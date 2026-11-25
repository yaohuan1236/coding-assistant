"""write_file:在 Workspace 内创建或完整覆盖 UTF-8 文本文件。

路径安全完全委托给 Workspace.resolve;本工具只负责参数校验、建父目录、写入。
可恢复错误(缺参数、越界、写入失败)一律转成 ToolResult(is_error=True)。
"""
from __future__ import annotations

from typing import Any

from codecat.core.types import ToolCall, ToolResult
from codecat.core.workspace import Workspace, WorkspaceError


class WriteFileTool:
    name = "write_file"
    description = "在 Workspace 内创建或完整覆盖 UTF-8 文本文件"

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
                            "description": "Workspace 内的目标文件路径",
                        },
                        "content": {
                            "type": "string",
                            "description": "要完整写入文件的文本内容(可为空字符串)",
                        },
                    },
                    "required": ["path", "content"],
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

        # content="" 是合法的空文件,不能用真值判断,只校验类型
        content = call.arguments.get("content")
        if not isinstance(content, str):
            return ToolResult(
                tool_call_id=call.id,
                content="缺少必需参数: content",
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
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(
                tool_call_id=call.id,
                content=f"写入文件失败: {exc}",
                is_error=True,
            )

        return ToolResult(tool_call_id=call.id, content=f"文件写入成功: {path}")
