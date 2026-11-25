"""最小 Agent Loop 控制流测试(全部离线,不调真实 API)。

覆盖任务书要求的 4 个场景:
1. 普通对话:user → LLM → assistant
2. 一次 tool call:user → LLM(tool_call) → ToolResult → LLM → final answer
3. 连续 tool call:受 max_steps 限制,不会无限循环
4. 错误处理:LLM 调用失败 / 工具执行失败,都给出明确结果而不是崩溃或静默

另加一条转换回归:tool_call_id 能正确往返 provider 格式(第二轮请求的前提)。
"""
from __future__ import annotations

import asyncio
from typing import Any, Sequence

import pytest

from codecat.core.llm import LLMCallError, to_provider_messages
from codecat.core.types import LLMResponse, Message, ToolCall, ToolResult
from codecat.loop import Agent


class FakeLLM:
    """按预设序列返回 LLMResponse(或抛出预设异常),并记录每次收到的消息。"""

    def __init__(self, script: Sequence[LLMResponse | Exception]) -> None:
        self._script = list(script)
        self.requests: list[list[Message]] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.requests.append(list(messages))  # 快照:验证 loop 每轮发出去的完整历史
        if not self._script:
            raise AssertionError("FakeLLM 脚本已耗尽:loop 的调用次数超出预期")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------------------
# 1. 普通对话
# ---------------------------------------------------------------------------

def test_plain_chat_returns_final_content() -> None:
    fake = FakeLLM([LLMResponse(content="你好,我是码猫。")])
    agent = Agent(fake, system_prompt="你是码猫")

    answer = asyncio.run(agent.run("你好"))

    assert answer == "你好,我是码猫。"
    assert fake.call_count == 1
    assert [m.role for m in agent.messages] == ["system", "user", "assistant"]
    assert fake.requests[0][0].content == "你是码猫"  # system prompt 在第一次请求里


# ---------------------------------------------------------------------------
# 2. 一次 tool call 的完整往返
# ---------------------------------------------------------------------------

def test_single_tool_call_roundtrip() -> None:
    call = ToolCall(id="call_1", name="echo", arguments={"text": "hi"})
    fake = FakeLLM(
        [
            LLMResponse(content=None, tool_calls=[call]),
            LLMResponse(content="工具结果已拿到:hi"),
        ]
    )

    async def execute(c: ToolCall) -> ToolResult:
        return ToolResult(tool_call_id=c.id, content=c.arguments["text"])

    agent = Agent(fake, execute_tool=execute)
    answer = asyncio.run(agent.run("帮我回显 hi"))

    assert answer == "工具结果已拿到:hi"
    assert fake.call_count == 2
    assert [m.role for m in agent.messages] == ["user", "assistant", "tool", "assistant"]

    tool_message = agent.messages[2]
    assert tool_message.tool_call_id == "call_1"  # 协议要求:回填时必须带 id
    assert tool_message.content == "hi"
    # 第二次请求里能同时看到 assistant.tool_calls 与 tool 消息
    roles = [m.role for m in fake.requests[1]]
    assert roles == ["user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# 3. 连续 tool call:被 max_steps 截断,不会无限循环
# ---------------------------------------------------------------------------

def test_consecutive_tool_calls_stop_at_max_steps() -> None:
    script = [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="noop")]) for i in range(5)]
    fake = FakeLLM(script)

    async def execute(c: ToolCall) -> ToolResult:
        return ToolResult(tool_call_id=c.id, content="ok")

    agent = Agent(fake, execute_tool=execute, max_steps=3)
    answer = asyncio.run(agent.run("一直调用工具"))

    assert fake.call_count == 3  # 恰好 max_steps 次 LLM 调用
    assert "max_steps" in answer  # 安全结束,并有明确提示


# ---------------------------------------------------------------------------
# 4. 错误处理:LLM 失败 / 工具失败
# ---------------------------------------------------------------------------

def test_llm_error_returns_readable_message() -> None:
    fake = FakeLLM([LLMCallError("HTTP 500:服务端错误")])
    agent = Agent(fake)

    answer = asyncio.run(agent.run("你好"))

    assert "LLM 调用失败" in answer
    assert "HTTP 500" in answer  # 错误原因透传,不静默吞掉


def test_tool_failure_becomes_error_result_not_crash() -> None:
    call = ToolCall(id="call_1", name="boom")
    fake = FakeLLM(
        [
            LLMResponse(tool_calls=[call]),
            LLMResponse(content="工具失败了,我换个办法"),
        ]
    )

    async def execute(c: ToolCall) -> ToolResult:
        raise RuntimeError("磁盘炸了")

    agent = Agent(fake, execute_tool=execute)
    answer = asyncio.run(agent.run("执行一下"))

    assert answer == "工具失败了,我换个办法"  # loop 没崩,模型拿到错误后继续
    tool_message = agent.messages[2]
    assert tool_message.role == "tool"
    assert "磁盘炸了" in tool_message.content


# ---------------------------------------------------------------------------
# 额外回归:项目 Message ⇄ provider 格式的双向契约
# ---------------------------------------------------------------------------

def test_provider_message_conversion_keeps_tool_contract() -> None:
    messages = [
        Message(role="assistant", content=None, tool_calls=[ToolCall("c1", "echo", {"a": 1})]),
        Message(role="tool", content="ok", tool_call_id="c1"),
    ]

    converted = to_provider_messages(messages)

    assert converted[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'  # dict → JSON 串
    assert converted[0]["tool_calls"][0]["id"] == "c1"
    assert converted[1]["tool_call_id"] == "c1"  # 缺失会被 provider 拒绝

    with pytest.raises(ValueError):
        Message(role="tool", content="没有 id 的 tool 消息")
