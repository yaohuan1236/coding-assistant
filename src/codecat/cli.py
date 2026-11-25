"""CLI 入口:装配运行时并进入交互循环。

职责仅限 wiring:
- 加载配置 → 创建 LLMClient
- 创建 Workspace(文件访问边界)
- 创建 ToolRegistry 并注册内置 Tool
- 创建 AgentLoop
- 读取用户输入、展示结果与可读错误

推理、工具路由、Provider 通信全部在各自模块,CLI 不重复实现任何一层。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from codecat.core.llm import LLMError, OpenAICompatibleClient, load_llm_config
from codecat.core.loop import AgentLoop, AgentMaxStepsError
from codecat.core.tools import (
    DefaultToolRegistry,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from codecat.core.workspace import Workspace

BANNER = "码猫 CodeCat v0.1.0"
SYSTEM_PROMPT = (
    "你是码猫 CodeCat,一个简洁的中文编程助手。"
    "所有文件路径都相对于当前 Workspace 根目录,不能访问范围之外的路径。"
    "当需要查看文件内容或目录结构时,使用 read_file / list_files 工具;"
    "需要创建或修改文件时,使用 write_file 工具。不要凭空猜测。"
    "回答尽量简短。"
)
MAX_STEPS = 20
EXIT_COMMANDS = {"exit", "quit", "退出"}
# 默认 Workspace:项目根目录下的 workspace/(src/codecat/cli.py 往上三级)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "workspace"


def build_registry(workspace: Workspace) -> DefaultToolRegistry:
    """注册内置工具,统一注入 Workspace 边界。"""
    registry = DefaultToolRegistry()
    registry.register(ReadFileTool(workspace))
    registry.register(ListFilesTool(workspace))
    registry.register(WriteFileTool(workspace))
    return registry


def build_agent() -> AgentLoop:
    """按依赖顺序装配运行时:配置 → LLMClient → Workspace → ToolRegistry → AgentLoop。"""
    config = load_llm_config()
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        llm=OpenAICompatibleClient(config),
        tool_registry=build_registry(Workspace(WORKSPACE_ROOT)),
        max_steps=MAX_STEPS,
        system_prompt=SYSTEM_PROMPT,
    )


async def chat_loop() -> None:
    agent = build_agent()
    print(BANNER)
    print("输入问题开始对话;输入 exit / quit 退出。\n")

    while True:
        try:
            raw = input("你:")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        user_input = raw.strip()
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("再见。")
            return

        try:
            answer = await agent.run(user_input)
        except AgentMaxStepsError:
            print("CodeCat:本次任务执行步数超过限制,已终止。\n")
            continue
        except LLMError as exc:
            print(f"CodeCat:LLM 调用失败:{exc}\n")
            continue

        print(f"CodeCat:{answer}\n")


def main() -> None:
    # Windows 控制台默认 GBK,强制 UTF-8 输出,避免中文乱码
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\n再见。")
    except LLMError as exc:  # 启动期错误:配置缺失、API key 缺失等
        print(f"启动失败:{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
