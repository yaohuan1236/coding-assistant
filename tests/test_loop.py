"""AgentLoop 第一版控制流测试(全部离线,不调真实 API)。

覆盖任务书第 26 节的 6 个场景:
1. 直接回答
2. 一次 Tool Call(消息历史完整)
3. 一次响应多个 Tool Call(只算 1 个 Step)
4. Tool Error 回填后继续
5. 超过 max_steps
6. LLMError 不被吞掉
"""
from __future__ import annotations

import asyncio
from typing import Any, Sequence

import pytest

from codecat.core.llm import LLMCallError
from codecat.core.loop import AgentLoop, AgentMaxStepsError, ToolRegistry
from codecat.core.types import LLMResponse, Message, ToolCall, ToolResult


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


class FakeToolRegistry:
    """记录调用并按预设返回 ToolResult 的最小注册表替身。"""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.results: dict[str, ToolResult] = {}

    def schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "fake"}}]

    async def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if call.name in self.results:
            return self.results[call.name]
        return ToolResult(tool_call_id=call.id, content=f"{call.name} ok")


# ---------------------------------------------------------------------------
# 1. 直接回答
# ---------------------------------------------------------------------------

def test_direct_answer_returns_content() -> None:
    fake = FakeLLM([LLMResponse(content="你好,我是码猫。")])
    registry = FakeToolRegistry()
    agent = AgentLoop(fake, registry, system_prompt="你是码猫")

    answer = asyncio.run(agent.run("你好"))

    assert answer == "你好,我是码猫。"
    assert fake.call_count == 1
    assert registry.calls == []  # 没有调用任何工具
    assert [m.role for m in fake.requests[0]] == ["system", "user"]


def test_no_system_prompt_does_not_create_empty_system_message() -> None:
    fake = FakeLLM([LLMResponse(content="ok")])
    agent = AgentLoop(fake, FakeToolRegistry())

    asyncio.run(agent.run("hi"))

    assert [m.role for m in fake.requests[0]] == ["user"]


# ---------------------------------------------------------------------------
# 2. 一次 Tool Call:消息历史完整
# ---------------------------------------------------------------------------

def test_single_tool_call_keeps_history_complete() -> None:
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})
    fake = FakeLLM([LLMResponse(tool_calls=[call]), LLMResponse(content="读到了")])
    registry = FakeToolRegistry()

    agent = AgentLoop(fake, registry, max_steps=3)
    answer = asyncio.run(agent.run("读 a.txt"))

    assert answer == "读到了"
    assert registry.calls == [call]
    # 第二次请求必须同时包含 assistant(tool_calls) 与 tool 消息
    assert [m.role for m in fake.requests[1]] == ["user", "assistant", "tool"]
    tool_message = fake.requests[1][2]
    assert tool_message.tool_call_id == "call_1"


# ---------------------------------------------------------------------------
# 3. 一次响应多个 Tool Call:全部执行,仍只算 1 个 Step
# ---------------------------------------------------------------------------

def test_multiple_tool_calls_in_one_response() -> None:
    calls = [
        ToolCall(id="c1", name="read_file", arguments={"path": "a"}),
        ToolCall(id="c2", name="read_file", arguments={"path": "b"}),
    ]
    fake = FakeLLM([LLMResponse(tool_calls=calls), LLMResponse(content="两个都读了")])
    registry = FakeToolRegistry()

    agent = AgentLoop(fake, registry, max_steps=2)
    answer = asyncio.run(agent.run("读两个文件"))

    assert answer == "两个都读了"
    assert [c.id for c in registry.calls] == ["c1", "c2"]  # 两个都执行
    assert fake.call_count == 2  # 2 个 Tool Call 仍然只占 1 个 Step


def test_tool_calls_do_not_consume_extra_steps() -> None:
    calls = [
        ToolCall(id="c1", name="a"),
        ToolCall(id="c2", name="b"),
        ToolCall(id="c3", name="c"),
    ]
    fake = FakeLLM([LLMResponse(tool_calls=calls)])  # 只有 1 次 LLM 决策
    registry = FakeToolRegistry()

    agent = AgentLoop(fake, registry, max_steps=1)  # 预算刚好用完
    with pytest.raises(AgentMaxStepsError):
        asyncio.run(agent.run("一次调三个"))

    assert fake.call_count == 1
    assert len(registry.calls) == 3  # 三个工具全部执行,但没有偷偷多调 LLM


# ---------------------------------------------------------------------------
# 4. Tool Error:回填 is_error 结果,Agent 继续
# ---------------------------------------------------------------------------

def test_tool_error_is_passed_back_to_llm() -> None:
    call = ToolCall(id="c1", name="read_file", arguments={"path": "missing.py"})
    fake = FakeLLM(
        [LLMResponse(tool_calls=[call]), LLMResponse(content="文件不存在,我换个办法")]
    )
    registry = FakeToolRegistry()
    registry.results["read_file"] = ToolResult(
        tool_call_id="c1", content="文件不存在: missing.py", is_error=True
    )

    agent = AgentLoop(fake, registry, max_steps=3)
    answer = asyncio.run(agent.run("读 missing.py"))

    assert answer == "文件不存在,我换个办法"
    tool_message = fake.requests[1][2]
    assert tool_message.content == "文件不存在: missing.py"


# ---------------------------------------------------------------------------
# 5. 超过 max_steps:不会无限循环
# ---------------------------------------------------------------------------

def test_max_steps_stops_infinite_tool_calls() -> None:
    script = [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="noop")]) for i in range(10)]
    fake = FakeLLM(script)
    registry = FakeToolRegistry()

    agent = AgentLoop(fake, registry, max_steps=3)
    with pytest.raises(AgentMaxStepsError, match="max_steps=3"):
        asyncio.run(agent.run("一直调用工具"))

    assert fake.call_count == 3  # 恰好 3 次 LLM 决策,没有第 4 次
    assert len(registry.calls) == 3


# ---------------------------------------------------------------------------
# 6. LLMError:不被吞掉
# ---------------------------------------------------------------------------

def test_llm_error_is_not_swallowed() -> None:
    fake = FakeLLM([LLMCallError("HTTP 500:服务端错误")])
    registry = FakeToolRegistry()
    agent = AgentLoop(fake, registry)

    with pytest.raises(LLMCallError, match="HTTP 500"):
        asyncio.run(agent.run("你好"))

    assert registry.calls == []


# ---------------------------------------------------------------------------
# Tool 日志:旁路观察,不改变控制流
# ---------------------------------------------------------------------------

def test_direct_answer_has_no_tool_logs(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeLLM([LLMResponse(content="你好,我是码猫。")])
    agent = AgentLoop(fake, FakeToolRegistry())

    asyncio.run(agent.run("你好"))

    out = capsys.readouterr().out
    assert "[Tool]" not in out
    assert "[Args]" not in out


def test_tool_call_logs_name_args_and_success(capsys: pytest.CaptureFixture[str]) -> None:
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})
    fake = FakeLLM([LLMResponse(tool_calls=[call]), LLMResponse(content="完成")])
    agent = AgentLoop(fake, FakeToolRegistry(), max_steps=3)

    answer = asyncio.run(agent.run("读文件"))

    assert answer == "完成"
    out = capsys.readouterr().out
    assert "[Tool] read_file" in out
    assert '[Args] {"path": "a.txt"}' in out
    assert "[Tool] success" in out
    assert "[Tool] error" not in out


def test_tool_error_logs_error_and_agent_continues(capsys: pytest.CaptureFixture[str]) -> None:
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "missing.txt"})
    fake = FakeLLM(
        [LLMResponse(tool_calls=[call]), LLMResponse(content="文件不存在,我换个办法")]
    )
    registry = FakeToolRegistry()
    registry.results["read_file"] = ToolResult(
        tool_call_id="call_1", content="文件不存在: missing.txt", is_error=True
    )
    agent = AgentLoop(fake, registry, max_steps=3)

    answer = asyncio.run(agent.run("读 missing.txt"))

    assert answer == "文件不存在,我换个办法"
    out = capsys.readouterr().out
    assert "[Tool] error" in out
    assert "Traceback" not in out


def test_multiple_tool_calls_log_separately(capsys: pytest.CaptureFixture[str]) -> None:
    calls = [ToolCall(id="c1", name="tool_a"), ToolCall(id="c2", name="tool_b")]
    fake = FakeLLM([LLMResponse(tool_calls=calls), LLMResponse(content="完成")])
    agent = AgentLoop(fake, FakeToolRegistry(), max_steps=2)

    asyncio.run(agent.run("调两个工具"))

    out = capsys.readouterr().out
    assert "[Tool] tool_a" in out
    assert "[Tool] tool_b" in out
    assert out.count("[Tool] success") == 2  # 每个 ToolCall 各一条状态


# ---------------------------------------------------------------------------
# 契约细节
# ---------------------------------------------------------------------------

def test_invalid_max_steps_rejected() -> None:
    with pytest.raises(ValueError):
        AgentLoop(FakeLLM([]), FakeToolRegistry(), max_steps=0)


def test_fake_registry_satisfies_tool_registry_protocol() -> None:
    assert isinstance(FakeToolRegistry(), ToolRegistry)
    assert not isinstance(object(), ToolRegistry)
