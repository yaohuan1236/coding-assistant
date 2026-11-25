"""Message 契约约束测试:非法字段组合必须在构造时就失败。"""
from __future__ import annotations

import pytest

from codecat.core.types import Message, ToolCall


# ---------------------------------------------------------------------------
# role="tool" 的约束
# ---------------------------------------------------------------------------

def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValueError, match="必须携带 tool_call_id"):
        Message(role="tool", content="结果")


def test_tool_message_rejects_tool_calls() -> None:
    with pytest.raises(ValueError, match="不能携带 tool_calls"):
        Message(
            role="tool",
            content="结果",
            tool_call_id="c1",
            tool_calls=[ToolCall(id="c2", name="echo")],
        )


def test_tool_message_requires_content() -> None:
    with pytest.raises(ValueError, match="必须有 content"):
        Message(role="tool", content=None, tool_call_id="c1")


# ---------------------------------------------------------------------------
# 镜像约束:tool_calls 只在 assistant;tool_call_id 只在 tool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["system", "user"])
def test_only_assistant_can_carry_tool_calls(role: str) -> None:
    with pytest.raises(ValueError, match="只有 role='assistant'"):
        Message(role=role, content="x", tool_calls=[ToolCall(id="c1", name="echo")])


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_only_tool_can_carry_tool_call_id(role: str) -> None:
    with pytest.raises(ValueError, match="只有 role='tool'"):
        Message(role=role, content="x", tool_call_id="c1")


# ---------------------------------------------------------------------------
# 合法形状不受影响
# ---------------------------------------------------------------------------

def test_valid_message_shapes_are_accepted() -> None:
    Message(role="system", content="你是助手")
    Message(role="user", content="问题")
    Message(role="assistant", content="回答")
    Message(role="assistant", content=None, tool_calls=[ToolCall(id="c1", name="echo")])
    Message(role="tool", content="结果", tool_call_id="c1")
