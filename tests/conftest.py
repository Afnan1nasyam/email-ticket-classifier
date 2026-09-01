"""Shared pytest fixtures, including an offline fake LLM provider.

The whole test suite runs with no network and no API key. `FakeProvider`
satisfies the same `LLMProvider` interface as `GroqProvider` but returns canned
responses, so the classifier and API can be exercised deterministically.
"""

from __future__ import annotations

from typing import Callable, Sequence

import pytest

from app.llm_provider import LLMProvider, LLMResponse

# A minimal valid classifier payload, used as the default fake completion.
DEFAULT_FAKE_COMPLETION = '{"label": "general", "confidence": 0.5, "reasoning": "fake"}'

# Type of a per-call response source: a fixed string, a sequence of strings
# returned in order (last one repeats), or a callable of (system, user).
ResponseSpec = str | Sequence[str] | Callable[[str, str], str]


class FakeProvider(LLMProvider):
    """An `LLMProvider` that returns canned responses, for offline tests.

    Configure with `responses`:
      - a `str` — returned for every call;
      - a sequence of `str` — returned in order, the last repeating once
        exhausted;
      - a callable `(system_prompt, user_message) -> str`.

    Every call is recorded in `self.calls` as `(system_prompt, user_message)`.
    """

    def __init__(
        self,
        responses: ResponseSpec | None = None,
        *,
        prompt_tokens: int = 100,
        completion_tokens: int = 20,
        latency_ms: float = 0.0,
        model_id: str = "fake-model",
    ) -> None:
        self._responses = responses if responses is not None else DEFAULT_FAKE_COMPLETION
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._latency_ms = latency_ms
        self._model_id = model_id
        self._index = 0
        self.calls: list[tuple[str, str]] = []

    def _resolve(self, system_prompt: str, user_message: str) -> str:
        r = self._responses
        if callable(r):
            return r(system_prompt, user_message)
        if isinstance(r, str):
            return r
        # Sequence of strings: return in order, last one repeats.
        seq = list(r)
        text = seq[self._index] if self._index < len(seq) else seq[-1]
        self._index += 1
        return text

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        self.calls.append((system_prompt, user_message))
        text = self._resolve(system_prompt, user_message)
        return LLMResponse(
            text=text,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            latency_ms=self._latency_ms,
            model_id=self._model_id,
        )


@pytest.fixture
def fake_provider() -> FakeProvider:
    """A FakeProvider returning the default valid completion."""
    return FakeProvider()


@pytest.fixture
def make_fake_provider() -> Callable[..., FakeProvider]:
    """Factory fixture: build a FakeProvider with a custom response spec."""

    def _make(responses: ResponseSpec | None = None, **kwargs) -> FakeProvider:
        return FakeProvider(responses, **kwargs)

    return _make
