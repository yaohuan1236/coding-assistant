"""最小 Agent Loop(今日范围:控制流正确,不接真实工具)。

一轮 run 的形态:
    BUILD(历史 + 当前输入)→ CALL LLM
    → 无 tool_calls:返回最终回答,结束
    → 有 tool_calls:EXEC(注入的 execute_tool)→ 回填 ToolResult → 再 CALL
最多 max_steps 次 LLM 调用,防止模型无限调用工具。

工具注册/schema/handler 查找属于未来的 ToolRegistry,本文件只依赖注入的 execute_tool;
未来把它换成 ToolRegistry.execute 即可,不需要重写循环。
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from codecat.core.llm import LLMError
from codecat.core.types import LLMClient, Message, ToolCall, ToolResult

ToolExecutor = Callable[[ToolCall], Awaitable[ToolResult]]
"""工具执行入口契约。未来 ToolRegistry.execute 满足同一形状即可直接注入。"""


class Agent:
    """最小 Agent:持有对话历史,驱动 LLM ↔ 工具循环。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        execute_tool: ToolExecutor | None = None,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        self._llm = llm
        self._tools = tools
        self._execute_tool = execute_tool
        self._max_steps = max_steps
        self.messages: list[Message] = []
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))

    async def run(self, user_input: str) -> str:
        """执行一次完整对话回合,返回最终回答(预期错误转可读文本,不向上抛)。"""
        self.messages.append(Message(role="user", content=user_input))

        for _ in range(self._max_steps):
            try:
                response = await self._llm.complete(self.messages, tools=self._tools)
            except LLMError as exc:
                return f"LLM 调用失败:{exc}"

            self.messages.append(response.to_message())
            if not response.tool_calls:
                return response.content or ""

            for call in response.tool_calls:
                result = await self._run_tool(call)
                self.messages.append(
                    Message(role="tool", content=result.content, tool_call_id=result.tool_call_id)
                )

        return f"已达到最大步数限制(max_steps={self._max_steps}),本回合安全结束。"

    async def _run_tool(self, call: ToolCall) -> ToolResult:
        """调用注入的工具执行入口;任何异常都转成 is_error 结果(Never-Throw)。"""
        if self._execute_tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content="tool execution not implemented",
                is_error=True,
            )
        try:
            return await self._execute_tool(call)
        except Exception as exc:  # noqa: BLE001 —— 工具异常不得炸穿主循环
            return ToolResult(
                tool_call_id=call.id,
                content=f"工具执行异常:{type(exc).__name__}: {exc}",
                is_error=True,
            )
