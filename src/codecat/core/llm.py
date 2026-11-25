"""LLM 适配层:provider 边界的唯一入口。

职责(严格限定,见今日任务书第四节):
1. 加载配置:config/config.json + 环境变量(.env 兜底),产出 LLMConfig
2. 出向转换:项目 Message → provider 请求格式
3. 调用 provider:httpx 直连 OpenAI 兼容 /chat/completions(今日非流式)
4. 入向转换:provider 响应 → LLMResponse(arguments 的 JSON 解析在此完成)

loop.py 及上层不得出现 httpx / JSON / SDK 细节。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv

from codecat.core.types import LLMConfig, LLMResponse, Message, ToolCall, Usage

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/codecat/core/llm.py → 项目根目录

_DEFAULT_BASE_URLS = {"deepseek": "https://api.deepseek.com"}


# ---------------------------------------------------------------------------
# 异常:llm.py 抛出的都是「预期内」错误,loop.py 捕获 LLMError 并转为可读结果
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """LLM 层预期错误基类。"""


class ConfigError(LLMError):
    """配置文件缺失或格式错误。"""


class MissingAPIKeyError(LLMError):
    """API key 缺失。不允许带着空 key 悄悄发请求。"""


class LLMCallError(LLMError):
    """网络失败、HTTP 错误、响应结构不可解析。"""


class ToolCallParseError(LLMError):
    """模型给出的 tool arguments 不是合法 JSON 对象。"""


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_llm_config(config_path: str | Path | None = None) -> LLMConfig:
    """读取 config/config.json 的默认模型预设,产出 LLMConfig。

    ${VAR} 形式的 apiKey 从环境变量解析;项目根目录 .env 作为兜底填充。
    """
    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    path = Path(config_path) if config_path else _PROJECT_ROOT / "config" / "config.json"
    if not path.is_file():
        raise ConfigError(f"配置文件不存在:{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件不是合法 JSON:{path}") from exc

    try:
        preset_name = raw["agents"]["defaults"]["modelPreset"]
        preset = raw["modelPresets"][preset_name]
        provider_name = preset["provider"]
        provider = raw["providers"][provider_name]
        model = preset["model"]
    except KeyError as exc:
        raise ConfigError(f"配置文件缺少字段:{exc}") from exc

    raw_key = provider.get("apiKey", "")
    api_key = _resolve_env_ref(raw_key)
    if not api_key:
        env_name = raw_key[2:-1] if raw_key.startswith("${") and raw_key.endswith("}") else "API key"
        raise MissingAPIKeyError(
            f"缺少 API key:请设置环境变量 {env_name}(或写入项目根目录的 .env 文件)"
        )

    base_url = (provider.get("baseUrl") or _DEFAULT_BASE_URLS.get(provider_name, "")).rstrip("/")
    if not base_url:
        raise ConfigError(f"provider {provider_name!r} 未配置 baseUrl")

    return LLMConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=int(preset.get("maxTokens", 4096)),
    )


def _resolve_env_ref(value: str) -> str:
    """把 "${VAR}" 全串引用解析为环境变量值;普通字符串原样返回。"""
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


# ---------------------------------------------------------------------------
# Provider 客户端
# ---------------------------------------------------------------------------

class OpenAICompatibleClient:
    """基于 httpx 的 OpenAI 兼容 Chat Completions 客户端(DeepSeek 使用同一协议)。

    第一版只实现非流式 complete();流式接口留给接 TUI 时再扩展。
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": to_provider_messages(messages),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
        data = await self._post_chat_completions(payload)
        return parse_response(data)

    async def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self._config.request_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMCallError(f"LLM 请求失败:{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            raise LLMCallError(f"LLM 返回 HTTP {response.status_code}:{response.text[:500]}")
        try:
            return response.json()
        except ValueError as exc:
            raise LLMCallError(f"LLM 响应不是合法 JSON:{response.text[:200]!r}") from exc


# ---------------------------------------------------------------------------
# 双向转换:项目类型 ⇄ provider 格式
# ---------------------------------------------------------------------------

def to_provider_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """项目 Message → OpenAI 兼容消息列表(含 assistant.tool_calls / tool.tool_call_id)。

    第二轮及以后的请求要把带 tool_calls 的 assistant 消息和 tool 消息发回去,
    因此这是 loop 每轮都要走的路径,不只是测试用。
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            item["content"] = message.content
        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        result.append(item)
    return result


def parse_response(data: dict[str, Any]) -> LLMResponse:
    """provider 响应 dict → LLMResponse。tool arguments 在此解析为 dict。"""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMCallError(f"LLM 响应缺少 choices[0].message:{str(data)[:300]}") from exc

    tool_calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        name = function.get("name", "")
        arguments = _parse_tool_arguments(name, function.get("arguments", "{}"))
        tool_calls.append(ToolCall(id=raw.get("id", ""), name=name, arguments=arguments))

    usage = None
    raw_usage = data.get("usage")
    if raw_usage:
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            total_tokens=int(raw_usage.get("total_tokens", 0)),
        )

    return LLMResponse(content=message.get("content"), tool_calls=tool_calls, usage=usage)


def _parse_tool_arguments(name: str, raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):  # 个别 provider 直接给 dict
        return raw_args
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except (json.JSONDecodeError, TypeError) as exc:
        raise ToolCallParseError(
            f"工具 {name!r} 的 arguments 不是合法 JSON:{str(raw_args)[:200]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ToolCallParseError(
            f"工具 {name!r} 的 arguments 应为 JSON 对象,实际是 {type(parsed).__name__}"
        )
    return parsed
