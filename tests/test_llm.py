"""llm.py 测试:配置加载 + HTTP 错误边界。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from codecat.core.llm import (
    LLMCallError,
    MissingAPIKeyError,
    OpenAICompatibleClient,
    load_llm_config,
    to_provider_messages,
)
from codecat.core.types import LLMConfig, Message, ToolCall


def _write_config(path: Path, preset_extra: dict) -> Path:
    preset = {"provider": "p", "model": "m1", "maxTokens": 1024, **preset_extra}
    data = {
        "providers": {"p": {"apiKey": "test-key", "baseUrl": "https://example.invalid"}},
        "modelPresets": {"m": preset},
        "agents": {"defaults": {"modelPreset": "m"}},
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_llm_config_reads_generation_params(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "config.json", {"temperature": 0.2, "requestTimeout": 60}
    )

    config = load_llm_config(path)

    assert config.model == "m1"
    assert config.api_key == "test-key"
    assert config.max_tokens == 1024
    assert config.temperature == 0.2   # 配置里写了就必须生效
    assert config.request_timeout == 60.0


def test_load_llm_config_falls_back_to_defaults(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "config.json", {})

    config = load_llm_config(path)

    assert config.temperature == 1.0
    assert config.request_timeout == 120.0


# ---------------------------------------------------------------------------
# HTTP 错误边界:401/500 一律转成 LLMCallError(不抛裸 httpx 异常)
# ---------------------------------------------------------------------------

def _client_returning(status: int) -> OpenAICompatibleClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "boom"}})

    config = LLMConfig(model="m", base_url="https://example.invalid", api_key="test-key")
    return OpenAICompatibleClient(config, transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("status", [401, 500])
def test_http_error_becomes_llm_call_error(status: int) -> None:
    client = _client_returning(status)

    with pytest.raises(LLMCallError, match=f"HTTP {status}"):
        asyncio.run(client.complete([Message(role="user", content="hi")]))


# ---------------------------------------------------------------------------
# 配置缺少 API key:启动即报 MissingAPIKeyError
# ---------------------------------------------------------------------------

def test_missing_api_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODECAT_TEST_KEY", raising=False)
    data = {
        "providers": {
            "p": {"apiKey": "${CODECAT_TEST_KEY}", "baseUrl": "https://example.invalid"}
        },
        "modelPresets": {"m": {"provider": "p", "model": "m1", "maxTokens": 1024}},
        "agents": {"defaults": {"modelPreset": "m"}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MissingAPIKeyError, match="CODECAT_TEST_KEY"):
        load_llm_config(path)


# ---------------------------------------------------------------------------
# 出向转换:tool_call_id / tool_calls 契约(第二轮请求的前提)
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
