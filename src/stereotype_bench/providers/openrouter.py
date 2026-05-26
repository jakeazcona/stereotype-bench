"""OpenRouter adapter — one key reaches GPT, Claude, Gemini, Llama, Grok, Kimi, ..."""
from __future__ import annotations

import logging
import os
import random
import time

import httpx

from ..types import Message
from .base import GenerationResult

_log = logging.getLogger("stereotype_bench.provider")


class OpenRouterProvider:
    provider_id = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 120.0,
        referer: str = "https://github.com/jakeazcona/stereotype-bench",
        title: str = "stereotype-bench",
        max_retries_5xx: int = 3,
        max_retries_429: int = 6,
        backoff_base: float = 1.0,
        backoff_cap_5xx: float = 16.0,
        backoff_cap_429: float = 60.0,
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
        self.max_retries_5xx = max_retries_5xx
        self.max_retries_429 = max_retries_429
        self.backoff_base = backoff_base
        self.backoff_cap_5xx = backoff_cap_5xx
        self.backoff_cap_429 = backoff_cap_429

    @staticmethod
    def _parse_retry_after(resp: httpx.Response) -> float | None:
        """Return Retry-After in seconds if present and parseable, else None."""
        val = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None  # HTTP-date form — skip, fall back to exponential

    def _sleep_for_retry(
        self, status: int, attempt: int, resp: httpx.Response
    ) -> float:
        """Compute and perform the retry sleep; returns the actual sleep seconds."""
        server_hint = self._parse_retry_after(resp)
        if status == 429:
            cap = self.backoff_cap_429
        else:
            cap = self.backoff_cap_5xx
        backoff = min(cap, self.backoff_base * (2 ** attempt))
        # Full jitter helps de-synchronize concurrent workers all hitting the
        # same 429 wall; without it they retry in lockstep and re-collide.
        backoff = random.uniform(0.0, backoff)
        delay = max(backoff, server_hint) if server_hint is not None else backoff
        time.sleep(delay)
        return delay

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
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
            "Content-Type": "application/json",
        }
        # Retry policy: 429 (rate limit) gets more attempts with a longer cap
        # than 5xx (server error), counted independently. Retry-After honored.
        resp: httpx.Response | None = None
        tried_429 = 0
        tried_5xx = 0
        while True:
            resp = httpx.post(url, headers=headers, json=body, timeout=self.timeout)
            status = resp.status_code
            if status < 500 and status != 429:
                break  # success or non-retryable 4xx
            if status == 429:
                if tried_429 >= self.max_retries_429:
                    break
                delay = self._sleep_for_retry(status, tried_429, resp)
                _log.warning(
                    "rate_limit model=%s attempt=%d/%d sleep=%.1fs",
                    model, tried_429 + 1, self.max_retries_429, delay,
                )
                tried_429 += 1
            else:  # 5xx
                if tried_5xx >= self.max_retries_5xx:
                    break
                delay = self._sleep_for_retry(status, tried_5xx, resp)
                _log.warning(
                    "server_error model=%s status=%d attempt=%d/%d sleep=%.1fs",
                    model, status, tried_5xx + 1, self.max_retries_5xx, delay,
                )
                tried_5xx += 1
        assert resp is not None
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
