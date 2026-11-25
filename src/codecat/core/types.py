"""协议层:CodeCat 内部统一的数据契约。

约束:
- 纯数据结构 + 接口协议,不放业务逻辑
- 不 import 任何 provider SDK;SDK 类型不得越过 llm.py
- loop.py 只依赖本文件的类型,换模型供应商时不改动上层
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]
"""消息角色,与 OpenAI 兼容协议一致。"""


@dataclass
class ToolCall:
    """模型请求执行一次工具调用。arguments 已由 llm.py 解析为 dict。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行结果。失败时 is_error=True,content 放可读原因(Never-Throw)。"""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """统一消息结构。

    - role="tool" 时必须携带 tool_call_id:OpenAI 兼容协议要求,
      缺失会导致工具结果的回填请求被 provider 拒绝
    - role="assistant" 且要求调用工具时,tool_calls 非空,content 可为 None
    """

    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("role='tool' 的 Message 必须携带 tool_call_id")


@dataclass
class Usage:
    """token 用量。第一版只记录,不参与任何决策。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """一次 LLM 调用的结构化结果(content 与 tool_calls 至少一个非空)。"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None

    def to_message(self) -> Message:
        """转成可追加进历史消息的 assistant 消息。"""
        return Message(role="assistant", content=self.content, tool_calls=self.tool_calls or None)


@dataclass
class LLMConfig:
    """模型调用配置。api_key 不写死、不进日志(dataclass repr 中隐藏)。"""

    model: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float = 1.0
    max_tokens: int = 4096
    request_timeout: float = 120.0


class LLMClient(Protocol):
    """LLM 客户端接口。loop.py 只依赖此协议,不依赖任何具体实现。"""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """发送一次对话补全请求。

        tools 为 OpenAI 兼容的工具 schema 列表;第一版不解释其内容,原样透传 provider。
        """
        ...
