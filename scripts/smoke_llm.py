"""冒烟 + 流程演示:用真实 DeepSeek API 跑一次最小 Agent 回合,并打印每一环。

用法:python scripts/smoke_llm.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from codecat.core.llm import LLMError, OpenAICompatibleClient, load_llm_config
from codecat.core.loop import AgentLoop
from codecat.core.types import LLMResponse, Message, ToolCall, ToolResult


class TracingLLM:
    """透明代理:打印每次请求/响应,然后原样转发给真实客户端。"""

    def __init__(self, inner: OpenAICompatibleClient) -> None:
        self._inner = inner

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        print(f"    → 发出请求:{len(messages)} 条消息,roles={[m.role for m in messages]}")
        started = time.monotonic()
        response = await self._inner.complete(messages, tools=tools)
        elapsed = time.monotonic() - started
        usage = response.usage
        tokens = (
            f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
            f"total={usage.total_tokens}"
            if usage
            else "无"
        )
        print(f"    ← 收到响应({elapsed:.2f}s):content={response.content!r}")
        print(f"      tool_calls={len(response.tool_calls)} 个  usage:{tokens}")
        return response


class NoToolsRegistry:
    """演示脚本用:没有注册任何工具。"""

    def schemas(self) -> list[dict[str, Any]]:
        return []

    async def execute(self, call: ToolCall) -> ToolResult:
        raise RuntimeError(f"演示环境没有注册工具,无法执行 {call.name!r}")


async def main() -> None:
    print("[1] 加载配置:config/config.json + .env")
    config = load_llm_config()
    print(f"    model={config.model}  base_url={config.base_url}  "
          f"api_key=已读取({len(config.api_key)} 字符,不回显)")

    print("[2] 构造 AgentLoop(空工具注册表 + max_steps=4)")
    agent = AgentLoop(
        TracingLLM(OpenAICompatibleClient(config)),
        NoToolsRegistry(),
        max_steps=4,
        system_prompt="你是简洁的中文编程助手,回答不超过 30 字。",
    )

    print("[3] agent.run() → 追加 user 消息 → BUILD 完成 → CALL")
    answer = await agent.run("用一句话说明什么是 Agent Loop")

    print("[4] 没有 tool_calls → loop 直接返回最终回答,本回合结束")
    print("    最终回答:", answer)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(main())
    except LLMError as exc:
        print(f"冒烟失败({type(exc).__name__}):{exc}")
        sys.exit(1)
