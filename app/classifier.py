"""Ticket classification service.

`TicketClassifier` orchestrates the full path: preprocess -> build prompt ->
call the provider -> parse and validate. It is constructor-injected with its
collaborators so it can be tested offline with a fake provider, and it
guarantees the returned label is always one of the configured categories.
"""

from __future__ import annotations

import json
import logging

from app import config
from app.llm_provider import LLMProvider
from app.preprocessor import EmailPreprocessor
from app.prompt_builder import PromptBuilder
from app.schemas import ClassificationResult

logger = logging.getLogger(__name__)

_JSON_DECODER = json.JSONDecoder()


class TicketClassifier:
    """Classify a raw email into one of the configured categories."""

    def __init__(
        self,
        preprocessor: EmailPreprocessor,
        provider: LLMProvider,
        prompt_builder: PromptBuilder,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        template: str = config.ACTIVE_PROMPT_TEMPLATE,
    ) -> None:
        """Create a classifier.

        Args:
            preprocessor: cleans/truncates the raw email.
            provider: LLM provider used to obtain a completion.
            prompt_builder: renders the prompt template.
            confidence_threshold: below this self-reported confidence, the
                result is downgraded to the fallback category.
            template: prompt template filename to render.
        """
        self.preprocessor = preprocessor
        self.provider = provider
        self.prompt_builder = prompt_builder
        self.confidence_threshold = confidence_threshold
        self.template = template

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract the first JSON object from a model completion.

        The parser is deliberately tolerant. There are Groq community reports of
        gpt-oss-120b leaking reasoning text into `message.content`, so
        prose-before-JSON (and markdown code fences) are real, documented
        behaviors, not hypothetical — we scan for the first `{` that begins a
        valid JSON object and ignore any surrounding prose or fences.

        Recovering these two cases is strictly better than degrading them to the
        fallback category: degradation would discard an otherwise-correct answer
        just because it arrived wrapped in a fence or prose. Output that contains
        no valid JSON object at all still falls back to `general` (see
        `classify`), so tolerance never masks a genuinely unparseable response.

        Args:
            text: raw completion text.

        Returns:
            The parsed dict, or None if no valid JSON object is present.
        """
        if not text:
            return None
        start = text.find("{")
        while start != -1:
            try:
                obj, _ = _JSON_DECODER.raw_decode(text[start:])
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
                continue
            if isinstance(obj, dict):
                return obj
            start = text.find("{", start + 1)
        return None

    @staticmethod
    def _coerce_confidence(value: object) -> float:
        """Coerce a raw confidence value to a float in [0.0, 1.0].

        Returns 0.0 for anything non-numeric (which then trips the low-confidence
        fallback).
        """
        try:
            conf = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, conf))

    def classify(self, text: str) -> ClassificationResult:
        """Classify a raw email.

        Args:
            text: raw email body.

        Returns:
            A `ClassificationResult` whose label is always one of
            `config.CATEGORIES`.
        """
        cleaned = self.preprocessor.preprocess_one(text)
        system_prompt, user_message = self.prompt_builder.build(cleaned, template=self.template)
        response = self.provider.complete(system_prompt, user_message)

        parsed = self._extract_json(response.text)
        fallback_used = False

        if parsed is None:
            label = config.FALLBACK_LABEL
            confidence = 0.0
            reasoning = "Model response could not be parsed as JSON."
            fallback_used = True
        else:
            raw_label = str(parsed.get("label", "")).strip().lower()
            confidence = self._coerce_confidence(parsed.get("confidence"))
            reasoning = str(parsed.get("reasoning", "")).strip()

            if not config.is_valid_label(raw_label):
                label = config.FALLBACK_LABEL
                fallback_used = True
                reasoning = reasoning or f"Model returned an unknown label: {raw_label!r}."
            elif confidence < self.confidence_threshold:
                label = config.FALLBACK_LABEL
                fallback_used = True
            else:
                label = raw_label

        result = ClassificationResult(
            label=label,
            confidence=confidence,
            reasoning=reasoning,
            fallback_used=fallback_used,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model_id=response.model_id,
        )
        # Structured logging — never the API key or its prefix.
        logger.info(
            "classified label=%s confidence=%.2f fallback=%s latency_ms=%.0f "
            "prompt_tokens=%d completion_tokens=%d",
            result.label,
            result.confidence,
            result.fallback_used,
            result.latency_ms,
            result.prompt_tokens,
            result.completion_tokens,
        )
        return result
