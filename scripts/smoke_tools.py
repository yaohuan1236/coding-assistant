"""工具系统端到端冒烟:真实 LLM 决定调用 read_file,再基于文件内容回答。

用法(在项目根目录):python scripts/smoke_tools.py
"""
from __future__ import annotations

import asyncio
import sys

from codecat.core.llm import LLMError, OpenAICompatibleClient, load_llm_config
from codecat.core.loop import AgentLoop
from codecat.core.tools import DefaultToolRegistry, ReadFileTool
from codecat.core.types import ToolCall, ToolResult


class TracingReadFileTool(ReadFileTool):
    """在真实工具外面加一行打印,方便观察工具是否真的被调用。"""

    async def execute(self, call: ToolCall) -> ToolResult:
        result = await super().execute(call)
        status = "错误" if result.is_error else f"{len(result.content)} 字符"
        print(f"    [tool] read_file({call.arguments}) → is_error={result.is_error}, {status}")
        return result


async def main() -> None:
    config = load_llm_config()
    registry = DefaultToolRegistry()
    registry.register(TracingReadFileTool())

    agent = AgentLoop(
        OpenAICompatibleClient(config),
        registry,
        max_steps=6,
        system_prompt=(
            "你是简洁的中文编程助手。需要了解项目时,必须先用 read_file 读取文件,"
            "不要凭空回答。最终回答不超过 60 字。"
        ),
    )

    print("[1] 用户:请读取 REQ.md,并告诉我这个项目主要是做什么的。")
    print("[2] AgentLoop 运行中(是否调用 read_file 由 LLM 决定)…")
    answer = await agent.run("请读取 REQ.md,并告诉我这个项目主要是做什么的。")
    print("[3] 最终回答:", answer)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(main())
    except LLMError as exc:
        print(f"冒烟失败({type(exc).__name__}):{exc}")
        sys.exit(1)
