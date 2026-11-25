"""Workspace:Agent 文件访问的统一边界。

职责:
- 把 Agent 提供的路径解析为 Workspace 内的绝对路径
- 拒绝越界路径(含 ".." 穿越、外部绝对路径、workspace2 这类前缀相似目录)

不负责读写文件——那是具体 Tool 的职责;也不属于 AgentLoop/Registry 的职责。
"""
from __future__ import annotations

from pathlib import Path


class WorkspaceError(Exception):
    """路径超出 Workspace 范围或无法解析。"""


class Workspace:
    """Agent 文件操作的根目录边界。"""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, path: str) -> Path:
        """把 Agent 给出的路径解析为 Workspace 内的绝对路径。

        前置条件:path 已由调用方(Tool 层)校验为非空字符串;
        缺失/空串/非字符串属于 Tool 的参数校验职责,本类不重复做。
        因此 resolve("") 会得到根目录,resolve(None) 会抛 TypeError——
        这两种情况在 Tool 层已被拦截。

        规则:
        - 任何包含 ".." 的路径一律拒绝(即使最终会落回 Workspace 内)
        - 相对路径按 Workspace 根解析;"." 解析为根本身
        - 绝对路径必须在 Workspace 内
        - 用 resolve() + relative_to() 判断,不做字符串前缀比较
          (因此 workspace 与 workspace-other 不会被混淆)

        越界时抛 WorkspaceError,由调用方(Tool)转成 is_error 的 ToolResult。
        """
        candidate = Path(path)

        # 严格拒绝父目录穿越,不做"绕一圈回来也算合法"的宽容
        if ".." in candidate.parts:
            raise WorkspaceError("路径超出 Workspace 范围")

        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self._root / candidate).resolve()

        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise WorkspaceError("路径超出 Workspace 范围") from None

        return resolved
