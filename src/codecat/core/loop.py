"""CodeCat Agent Runtime 第一版:最小、稳定的执行闭环。

职责边界(见第一版任务书):
- loop:保存对话历史、调用 LLM、判断/回填 Tool Call、限制 max_steps
- llm.py:与 provider 通信(loop 不碰 httpx / JSON / SDK)
- ToolRegistry:工具调度(loop 不知道任何具体工具)
- ToolResult → Message 的转换在 loop 完成(ToolResult 属执行层数据,Message 属协议)

一个 Step = 一次 LLM 决策;同一条响应里的多个 ToolCall 仍只算一个 Step。
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from codecat.core.types import LLMClient, Message, ToolCall, ToolResult


class AgentMaxStepsError(Exception):
    """LLM 决策轮数达到 max_steps 仍未完成,Agent 终止。"""


@runtime_checkable
class ToolRegistry(Protocol):
    """loop 依赖的最小工具调度接口(第一版临时定义在 loop.py)。

    TODO(实现 core/tools/registry.py 时):把本协议迁移到 core/tools/registry.py,
    loop 改为从那里导入并删除此处定义,保证全项目只有一份 Registry 契约;
    迁移时同步更新 tests/test_loop.py 的导入。
    """

    def schemas(self) -> list[dict[str, Any]]:
        """返回 OpenAI 兼容的工具 schema 列表(无工具时返回空列表)。"""
        ...

    async def execute(self, call: ToolCall) -> ToolResult:
        """执行一次工具调用。

        可恢复的工具错误(文件不存在、命令失败等)由实现方转换为
        ToolResult(is_error=True);实现方不应把这类错误抛给 loop。
        """
        ...


class AgentLoop:
    """最小 Agent Loop:一轮 run = 若干次「LLM 决策 → 工具执行 → 回填」。"""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        max_steps: int = 20,
        system_prompt: str | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        self._llm = llm
        self._tool_registry = tool_registry
        self._max_steps = max_steps
        self._system_prompt = system_prompt

    async def run(self, user_input: str) -> str:
        """执行一轮完整对话,返回最终回答。

        - max_steps 超限:抛 AgentMaxStepsError
        - LLM 层错误(LLMError):不在此吞掉,原样抛给调用方(CLI/测试)处理
        - 工具可恢复错误:由 ToolRegistry 转成 ToolResult(is_error=True) 回填给模型
        """
        messages = self._build_initial_messages(user_input)

        for _ in range(self._max_steps):
            # 每个 Step 重新取 schema:语义上允许工具集在 run 内变化;
            # 若将来 schema 生成变重,优先在 ToolRegistry 内部缓存,而不是在 loop 里预取
            response = await self._llm.complete(
                messages=messages,
                tools=self._tool_registry.schemas(),
            )

            # 先回填 assistant 消息(含 tool_calls),再执行工具:
            # 历史不能只剩 tool 结果而丢掉对应的 tool_calls
            messages.append(response.to_message())

            if not response.tool_calls:
                return response.content or ""

            # 第一版严格串行:写类工具并发执行会引发工作区竞态
            for tool_call in response.tool_calls:
                self._log_tool_call(tool_call)
                result = await self._tool_registry.execute(tool_call)
                self._log_tool_result(result)
                messages.append(
                    Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    )
                )

        raise AgentMaxStepsError(
            f"达到最大步数限制(max_steps={self._max_steps}),Agent 已终止"
        )

    @staticmethod
    def _log_tool_call(tool_call: ToolCall) -> None:
        """终端可见的 Tool 调用日志:名称 + 参数。只做格式化,不解析 Provider JSON。"""
        print(f"[Tool] {tool_call.name}")
        print(f"[Args] {json.dumps(tool_call.arguments, ensure_ascii=False, default=str)}")

    @staticmethod
    def _log_tool_result(result: ToolResult) -> None:
        """只打印执行状态;不打印完整内容(可能是长文件)。"""
        print(f"[Tool] {'error' if result.is_error else 'success'}")

    def _build_initial_messages(self, user_input: str) -> list[Message]:
        """system prompt(如有)+ user 消息;未配置时不留空 system 消息。"""
        messages: list[Message] = []
        if self._system_prompt:
            messages.append(Message(role="system", content=self._system_prompt))
        messages.append(Message(role="user", content=user_input))
        return messages
