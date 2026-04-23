from unittest.mock import MagicMock, patch

import pytest

from stereotype_bench.providers.openrouter import OpenRouterProvider
from stereotype_bench.types import Message


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider()


def test_openrouter_generate(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake = MagicMock()
    fake.json.return_value = {
        "choices": [{"message": {"content": "hello world"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.0001},
    }
    fake.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=fake) as mock_post:
        p = OpenRouterProvider()
        result = p.generate("foo/bar", [Message("user", "hi")])
    assert result.text == "hello world"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.cost_usd == pytest.approx(0.0001)
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "foo/bar"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
