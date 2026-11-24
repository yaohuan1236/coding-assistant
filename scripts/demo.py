"""Day-1 demo:跑通 nanobot SDK 的最小示例。

用法(在项目根目录):
    .venv/Scripts/python.exe scripts/demo.py "用 Python 写一个快速排序"
"""
import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


async def main() -> None:
    _load_env()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY:")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 到 platform.deepseek.com 创建 API key 并填入 .env")
        sys.exit(1)

    from nanobot import Nanobot

    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "用一句话解释:什么是 agent 的工具调用循环?"
    )

    async with Nanobot.from_config(
        config_path=PROJECT_ROOT / "config" / "config.json",
        workspace=PROJECT_ROOT / "workspace",
    ) as bot:
        print(f"模型: {bot.runtime.model}")
        print(f"工作区: {bot.runtime.workspace}")
        print("-" * 60)
        result = await bot.run(prompt)
        print("-" * 60)
        print(result.content)
        print()
        print(f"[使用工具] {result.tools_used}")
        print(f"[停止原因] {result.stop_reason}")
        if result.usage:
            print(f"[token 用量] {result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
