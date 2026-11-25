"""parse_response() 对 provider 畸形响应的严格性测试。"""
from __future__ import annotations

from typing import Any

import pytest

from codecat.core.llm import LLMCallError, ToolCallParseError, parse_response


def _payload(tool_calls: Any) -> dict[str, Any]:
    return {"choices": [{"message": {"content": None, "tool_calls": tool_calls}}]}


# ---------------------------------------------------------------------------
# 合法响应
# ---------------------------------------------------------------------------

def test_valid_tool_call_is_parsed() -> None:
    response = parse_response(
        _payload(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ]
        )
    )

    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.txt"}


def test_plain_text_response_without_tool_calls() -> None:
    data = {
        "choices": [{"message": {"role": "assistant", "content": "你好,我是码猫。"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }

    response = parse_response(data)

    assert response.content == "你好,我是码猫。"
    assert response.tool_calls == []
    assert response.usage is not None
    assert response.usage.total_tokens == 5


# ---------------------------------------------------------------------------
# 缺 id / 空 id / 结构非法
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        {"function": {"name": "read_file", "arguments": "{}"}},            # 缺 id
        {"id": "", "function": {"name": "read_file", "arguments": "{}"}},  # 空 id
        {"id": "   ", "function": {"name": "read_file", "arguments": "{}"}},  # 空白 id
    ],
)
def test_missing_or_empty_id_raises(raw: dict[str, Any]) -> None:
    with pytest.raises(ToolCallParseError, match="合法 id"):
        parse_response(_payload([raw]))


def test_non_dict_tool_call_raises() -> None:
    with pytest.raises(LLMCallError, match="tool_call 不是对象"):
        parse_response(_payload(["not-a-dict"]))


def test_missing_function_object_raises() -> None:
    with pytest.raises(LLMCallError, match="function 不是对象"):
        parse_response(_payload([{"id": "c1"}]))


# ---------------------------------------------------------------------------
# 外层信封结构:一律 LLMCallError,不允许 AttributeError 穿出
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "data",
    [
        ["not-a-dict"],                                     # data 不是 dict
        {},                                                 # 缺 choices
        {"choices": "nope"},                                # choices 不是 list
        {"choices": []},                                    # choices 为空
        {"choices": ["not-a-dict"]},                        # choices[0] 不是对象
        {"choices": [{}]},                                  # 缺 message
        {"choices": [{"message": "nope"}]},                 # message 不是对象
        {"choices": [{"message": {"tool_calls": "abc"}}]},  # tool_calls 不是 list
    ],
)
def test_invalid_envelope_raises_llm_call_error(data: Any) -> None:
    with pytest.raises(LLMCallError):
        parse_response(data)


# ---------------------------------------------------------------------------
# 缺 name / 空 name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        {"id": "c1", "function": {"arguments": "{}"}},             # 缺 name
        {"id": "c1", "function": {"name": "", "arguments": "{}"}},  # 空 name
    ],
)
def test_missing_or_empty_name_raises(raw: dict[str, Any]) -> None:
    with pytest.raises(ToolCallParseError, match="function.name"):
        parse_response(_payload([raw]))


# ---------------------------------------------------------------------------
# arguments 非法(已有逻辑,补测试锁定)
# ---------------------------------------------------------------------------

def test_invalid_arguments_json_raises() -> None:
    with pytest.raises(ToolCallParseError, match="合法 JSON"):
        parse_response(_payload([{"id": "c1", "function": {"name": "n", "arguments": "{oops"}}]))


def test_non_object_arguments_raises() -> None:
    with pytest.raises(ToolCallParseError, match="JSON 对象"):
        parse_response(_payload([{"id": "c1", "function": {"name": "n", "arguments": "[1, 2]"}}]))
