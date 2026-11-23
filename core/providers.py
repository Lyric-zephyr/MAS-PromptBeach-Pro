"""Endpoint-specific request capabilities; no global SDK monkey patches."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from core import profiles


def provider(base_url: str, model: str = "") -> str:
    explicit = profiles.value("MODEL_PROVIDER", "")
    if explicit:
        if explicit not in {"qwen", "deepseek", "openai", "vllm", "compatible"}:
            raise ValueError(f"unknown MODEL_PROVIDER: {explicit}")
        return explicit
    host = (urlsplit(base_url).hostname or "").lower()
    if host.endswith(("qianwenaiapi.com", "dashscope.aliyuncs.com")):
        return "qwen"
    if host.endswith("deepseek.com"):
        return "deepseek"
    if host == "api.openai.com":
        return "openai"
    return "vllm" if host in {"localhost", "127.0.0.1", "0.0.0.0"} else "compatible"


def limit_field(base_url: str, model: str = "") -> str:
    field = profiles.value("TOKEN_LIMIT_FIELD", "")
    field = field or ("max_completion_tokens" if provider(base_url, model) == "openai" else "max_tokens")
    if field not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("TOKEN_LIMIT_FIELD must be max_tokens or max_completion_tokens")
    return field


def body(base_url: str, model: str, thinking: bool, repetition_penalty: float) -> dict:
    kind = provider(base_url, model)
    extras = {}
    if kind == "qwen":
        extras["enable_thinking"] = thinking
    elif kind == "deepseek":
        extras["thinking"] = {"type": "enabled" if thinking else "disabled"}
    elif kind == "vllm":
        extras = {"repetition_penalty": repetition_penalty, "chat_template_kwargs": {"enable_thinking": thinking}}
    configured = json.loads(profiles.value("MODEL_EXTRA_BODY", "") or "{}")
    if not isinstance(configured, dict):
        raise ValueError("MODEL_EXTRA_BODY must be a JSON object")
    reserved = {
        "max_tokens",
        "max_completion_tokens",
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "seed",
    }
    if reserved.intersection(configured):
        raise ValueError("MODEL_EXTRA_BODY must not override standard request parameters")
    extras.update(configured)
    return extras


def adapt_payload(payload: dict, base_url: str, model: str) -> dict:
    """Applied *after* LangChain renames max_tokens. Respect each call's actual cap."""
    payload = dict(payload)
    target = limit_field(base_url, model)
    other = "max_tokens" if target == "max_completion_tokens" else "max_completion_tokens"
    if other in payload:
        if target in payload and payload[target] != payload[other]:
            raise ValueError("conflicting output token limits")
        payload[target] = payload.pop(other)
    if not payload.get("tools"):
        payload.pop("tool_choice", None)
    for key in profiles.value("MODEL_OMIT_PARAMS", "").split(","):
        payload.pop(key.strip(), None)
    return payload
