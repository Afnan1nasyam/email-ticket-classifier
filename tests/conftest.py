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

# A valid, high-confidence payload for a concrete category, used by the API and
# security fixtures where a non-fallback result is wanted.
VALID_BILLING_COMPLETION = (
    '{"label": "billing", "confidence": 0.95, "reasoning": "About an invoice charge."}'
)

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


class RaisingProvider(LLMProvider):
    """An `LLMProvider` whose `complete` always raises, to exercise error paths.

    The exception message deliberately embeds a fake key-shaped string so tests
    can assert that provider failures never leak anything matching the Groq API
    key prefix into logs or responses.
    """

    def __init__(self, message: str = "provider boom gsk_FAKELEAKEDKEY0000") -> None:
        self.message = message
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        self.calls.append((system_prompt, user_message))
        raise RuntimeError(self.message)


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


@pytest.fixture
def make_classifier() -> Callable[..., "TicketClassifier"]:
    """Factory fixture: build a TicketClassifier around a given provider.

    Defaults to a FakeProvider returning a valid, high-confidence billing
    payload so callers get a non-fallback result unless they say otherwise.
    """
    from app.classifier import TicketClassifier
    from app.preprocessor import EmailPreprocessor
    from app.prompt_builder import PromptBuilder

    def _make(provider: LLMProvider | None = None) -> "TicketClassifier":
        if provider is None:
            provider = FakeProvider(VALID_BILLING_COMPLETION)
        return TicketClassifier(EmailPreprocessor(), provider, PromptBuilder())

    return _make


@pytest.fixture
def classifier(make_classifier) -> "TicketClassifier":
    """A TicketClassifier wrapping a FakeProvider (valid billing completion)."""
    return make_classifier()


@pytest.fixture
def make_app() -> Callable[..., "Flask"]:
    """Factory fixture: build a Flask app around an injected classifier."""
    from app import create_app

    def _make(classifier) -> "Flask":
        return create_app(classifier=classifier)

    return _make


@pytest.fixture
def app(classifier):
    """A Flask app with an injected FakeProvider-backed classifier."""
    from app import create_app

    return create_app(classifier=classifier)


@pytest.fixture
def client(app):
    """Flask test client for the injected app."""
    return app.test_client()
