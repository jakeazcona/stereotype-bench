"""OpenRouter adapter — one key reaches GPT, Claude, Gemini, Llama, Grok, Kimi, ..."""
from __future__ import annotations

import os

import httpx

from ..types import Message
from .base import GenerationResult


class OpenRouterProvider:
    provider_id = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 120.0,
        referer: str = "https://github.com/jakeazcona/stereotype-bench",
        title: str = "stereotype-bench",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Export it, copy .env.example to .env, "
                "or pass api_key=... explicitly."
            )
        self.timeout = timeout
        self.referer = referer
        self.title = title

    def generate(
        self,
        model: str,
        messages: list[Message],
        **kwargs,
    ) -> GenerationResult:
        body = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "usage": {"include": True},  # ask OpenRouter to include cost
            **kwargs,
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.referer,
                "X-Title": self.title,
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            # Surface the response body — OpenRouter returns actionable JSON
            # like {"error":{"message":"...","code":402}}; httpx's default
            # raise_for_status() hides it.
            body_preview = resp.text[:500] if resp.text else "(empty body)"
            raise httpx.HTTPStatusError(
                f"OpenRouter {resp.status_code} for model={model!r}: {body_preview}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {}) or {}
        return GenerationResult(
            text=choice.get("content", ""),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            cost_usd=usage.get("cost"),
            raw=data,
        )
