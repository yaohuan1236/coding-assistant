"""CLI 接线测试:注册表装配、交互循环、退出与错误呈现。"""
from __future__ import annotations

import asyncio
from typing import Iterator

import pytest

from codecat.cli import build_registry, chat_loop
from codecat.core.llm import LLMCallError
from codecat.core.loop import AgentMaxStepsError
from codecat.core.workspace import Workspace


class FakeAgent:
    """替代真实 AgentLoop:记录输入,按预设返回或抛错。"""

    def __init__(self, answers: list[str] | None = None, error: Exception | None = None) -> None:
        self.answers = list(answers or [])
        self.error = error
        self.inputs: list[str] = []

    async def run(self, user_input: str) -> str:
        self.inputs.append(user_input)
        if self.error is not None:
            raise self.error
        return self.answers.pop(0)


def _feed_inputs(monkeypatch: pytest.MonkeyPatch, values: list[str]) -> None:
    iterator: Iterator[str] = iter(values)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)


def _patch_agent(monkeypatch: pytest.MonkeyPatch, agent: FakeAgent) -> None:
    monkeypatch.setattr("codecat.cli.build_agent", lambda: agent)


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------

def test_build_registry_registers_builtin_tools(tmp_path) -> None:
    registry = build_registry(Workspace(tmp_path))

    schemas = registry.schemas()

    assert [schema["function"]["name"] for schema in schemas] == [
        "read_file",
        "list_files",
        "write_file",
    ]


# ---------------------------------------------------------------------------
# 交互循环
# ---------------------------------------------------------------------------

def test_chat_loop_prints_answer(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    agent = FakeAgent(answers=["你好,我是码猫。"])
    _patch_agent(monkeypatch, agent)
    _feed_inputs(monkeypatch, ["你好", "exit"])

    asyncio.run(chat_loop())

    out = capsys.readouterr().out
    assert "CodeCat:你好,我是码猫。" in out
    assert agent.inputs == ["你好"]


def test_chat_loop_ignores_blank_input(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = FakeAgent(answers=["ok"])
    _patch_agent(monkeypatch, agent)
    _feed_inputs(monkeypatch, ["   ", "exit"])

    asyncio.run(chat_loop())

    assert agent.inputs == []  # 空输入不调用 Agent


def test_chat_loop_exits_on_exit_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_agent(monkeypatch, FakeAgent())
    _feed_inputs(monkeypatch, ["exit"])

    asyncio.run(chat_loop())

    assert "再见" in capsys.readouterr().out


def test_chat_loop_exits_on_eof(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_agent(monkeypatch, FakeAgent())
    _feed_inputs(monkeypatch, [])  # 第一次 input 直接 EOF

    asyncio.run(chat_loop())

    assert "再见" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 错误呈现:不暴露 traceback
# ---------------------------------------------------------------------------

def test_chat_loop_reports_max_steps_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    agent = FakeAgent(error=AgentMaxStepsError("max_steps=20"))
    _patch_agent(monkeypatch, agent)
    _feed_inputs(monkeypatch, ["算一下", "exit"])

    asyncio.run(chat_loop())

    out = capsys.readouterr().out
    assert "步数超过限制" in out
    assert "Traceback" not in out


def test_chat_loop_reports_llm_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    agent = FakeAgent(error=LLMCallError("HTTP 500: boom"))
    _patch_agent(monkeypatch, agent)
    _feed_inputs(monkeypatch, ["你好", "exit"])

    asyncio.run(chat_loop())

    out = capsys.readouterr().out
    assert "LLM 调用失败" in out
    assert "HTTP 500" in out
    assert "Traceback" not in out
