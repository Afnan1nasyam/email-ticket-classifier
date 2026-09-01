"""Typed data structures for classification results.

`ClassificationResult` is the validated output of `TicketClassifier`: the label
is always one of the configured categories (never a hallucinated one), and the
metadata needed for logging, eval metrics, and the API response travels with it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ClassificationResult:
    """The result of classifying one email.

    Attributes:
        label: the final category — guaranteed to be one of `config.CATEGORIES`.
        confidence: the model's self-reported confidence in [0.0, 1.0].
        reasoning: a short natural-language justification (may be empty).
        fallback_used: True if the result was forced to the fallback category
            (`general`) because the model output was invalid, named an unknown
            label, or fell below the confidence threshold.
        latency_ms: wall-clock latency of the provider call, in milliseconds.
        prompt_tokens: prompt tokens billed for the call.
        completion_tokens: completion tokens billed (includes reasoning tokens).
        model_id: the model that produced the classification.
    """

    label: str
    confidence: float
    reasoning: str
    fallback_used: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    model_id: str

    @property
    def total_tokens(self) -> int:
        """Total tokens billed for the call (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (for the API response)."""
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data
