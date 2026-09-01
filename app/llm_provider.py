"""LLM provider abstraction.

`LLMProvider` is the seam that keeps the rest of the codebase ignorant of which
LLM vendor is in use. `GroqProvider` is the only implementation that talks to a
real API, and this module is the ONLY place in the codebase that imports the
`groq` SDK — swapping providers is meant to be a change confined to this file
plus the model ID in `app/config.py`.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable

from app import config

logger = logging.getLogger(__name__)

# Trust the OS certificate store so TLS-inspecting corporate proxies (e.g.
# Zscaler on the build network) are trusted without disabling verification. This
# is a no-op on machines with no custom root CA (such as the deployment host),
# so it is safe to run unconditionally. Guarded so a missing optional dependency
# never blocks import.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - truststore is in requirements.txt
    logger.debug("truststore not available; using default certifi trust store")


@dataclass(frozen=True)
class LLMResponse:
    """A single completion plus the metadata the rest of the system needs.

    Attributes:
        text: the raw completion text (message content only).
        prompt_tokens: tokens billed for the prompt, as reported by the API.
        completion_tokens: tokens billed for the completion (includes any
            reasoning tokens), as reported by the API.
        latency_ms: wall-clock latency of the successful call, in milliseconds.
        model_id: the model that produced the completion.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model_id: str

    @property
    def total_tokens(self) -> int:
        """Total tokens billed for the call (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """Abstract interface for a chat-completion provider."""

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Return a completion for the given system prompt and user message.

        Args:
            system_prompt: the system role content.
            user_message: the user role content.

        Returns:
            An `LLMResponse` with the completion text and token/latency metadata.
        """
        raise NotImplementedError


class TokenRateLimiter:
    """Adaptive throttle for a tokens-per-minute (TPM) ceiling.

    Maintains a rolling window of the token counts actually observed on recent
    calls and blocks until the next call is expected to fit under the ceiling.
    This is the mechanism the eval runner uses to respect Groq's 6,000 TPM free-
    tier limit without a brittle fixed delay.

    The window is seeded with a conservative first-call estimate so an early
    burst cannot trip HTTP 429 before any real usage data has accumulated.
    """

    def __init__(
        self,
        tpm_limit: int = config.TPM_LIMIT,
        window_seconds: float = 60.0,
        seed_tokens: int = config.THROTTLE_SEED_TOKENS,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a rate limiter.

        Args:
            tpm_limit: token ceiling per rolling window.
            window_seconds: length of the rolling window.
            seed_tokens: synthetic first-call cost used to prime the window.
            sleep: sleep function (injectable for testing).
            now: monotonic clock (injectable for testing).
        """
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        self._sleep = sleep
        self._now = now
        self._events: deque[tuple[float, int]] = deque()
        if seed_tokens > 0:
            self._events.append((self._now(), seed_tokens))

    def _prune(self, t: float) -> None:
        while self._events and self._events[0][0] <= t - self.window_seconds:
            self._events.popleft()

    def _used(self, t: float) -> int:
        self._prune(t)
        return sum(tokens for _, tokens in self._events)

    def acquire(self, estimated_tokens: int = 0) -> None:
        """Block until `estimated_tokens` is expected to fit under the ceiling.

        Args:
            estimated_tokens: expected token cost of the imminent call.
        """
        if estimated_tokens >= self.tpm_limit:
            logger.warning(
                "Estimated tokens (%d) >= TPM limit (%d); proceeding without wait",
                estimated_tokens,
                self.tpm_limit,
            )
            return
        while True:
            t = self._now()
            if self._used(t) + estimated_tokens <= self.tpm_limit or not self._events:
                return
            wait = (self._events[0][0] + self.window_seconds) - t
            if wait <= 0:
                continue
            logger.debug("TPM throttle: sleeping %.1fs", wait)
            self._sleep(wait)

    def record(self, tokens: int) -> None:
        """Record the actual token cost of a completed call.

        Args:
            tokens: total tokens the call consumed (prompt + completion).
        """
        self._events.append((self._now(), tokens))


class GroqProvider(LLMProvider):
    """`LLMProvider` backed by Groq's chat-completions API.

    Owns retry/backoff and rate-limit awareness. Reads the classification JSON
    only from `message.content`; reasoning is suppressed via
    `include_reasoning=False`, and `reasoning_format` is never set (it is
    unsupported on gpt-oss-120b).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = config.MODEL_ID,
        temperature: float = config.TEMPERATURE,
        reasoning_effort: str = config.REASONING_EFFORT,
        max_retries: int = config.MAX_RETRIES,
        backoff_base: float = config.BACKOFF_BASE_SECONDS,
        timeout: float = config.REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """Create a Groq-backed provider.

        Args:
            api_key: Groq API key; falls back to `config.get_api_key()`.
            model_id: model to call.
            temperature: sampling temperature.
            reasoning_effort: gpt-oss reasoning effort ("low"|"medium"|"high").
            max_retries: max retry attempts on 429/5xx/connection errors.
            backoff_base: base (seconds) for exponential backoff.
            timeout: per-request timeout in seconds.
        """
        # Imported here (module-local to this file only) so this stays the single
        # point of contact with the groq SDK.
        from groq import Groq

        self.model_id = model_id
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = Groq(api_key=api_key or config.get_api_key(), timeout=timeout)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True for transient errors worth retrying (429, 5xx, network)."""
        from groq import APIConnectionError, APIStatusError, RateLimitError

        if isinstance(exc, (RateLimitError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code >= 500
        return False

    def _backoff_seconds(self, attempt: int, exc: Exception) -> float:
        """Compute the backoff delay for a retry, honoring Retry-After if given."""
        retry_after = getattr(getattr(exc, "response", None), "headers", {})
        if retry_after:
            value = retry_after.get("retry-after")
            if value:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        # Exponential backoff with jitter.
        return self.backoff_base ** attempt + random.uniform(0.0, 0.5)

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        """See `LLMProvider.complete`. Retries transient failures with backoff.

        Raises:
            groq.APIError: if the call fails non-transiently or retries are
                exhausted.
        """
        from groq import APIConnectionError, APIStatusError

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        attempt = 0
        while True:
            try:
                started = time.perf_counter()
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    include_reasoning=False,
                    messages=messages,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                break
            except (APIStatusError, APIConnectionError) as exc:
                if not self._is_retryable(exc) or attempt >= self.max_retries:
                    logger.error(
                        "Groq call failed (%s); not retrying", type(exc).__name__
                    )
                    raise
                attempt += 1
                delay = self._backoff_seconds(attempt, exc)
                logger.warning(
                    "Groq call failed (%s); retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)

        message = resp.choices[0].message
        usage = resp.usage
        response = LLMResponse(
            text=message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            model_id=resp.model or self.model_id,
        )
        logger.info(
            "Groq completion: model=%s prompt_tokens=%d completion_tokens=%d latency_ms=%.0f",
            response.model_id,
            response.prompt_tokens,
            response.completion_tokens,
            response.latency_ms,
        )
        return response
