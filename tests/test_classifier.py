"""Offline tests for `TicketClassifier` LLM-response parsing and fallback.

Every unexpected model shape must degrade to `general` (fallback_used=True)
rather than raise. The classifier is built with an injected FakeProvider; no
real provider is ever constructed.
"""

from __future__ import annotations

import pytest

from app import config
from app.classifier import TicketClassifier
from app.preprocessor import EmailPreprocessor
from app.prompt_builder import PromptBuilder
from app.schemas import ClassificationResult

from tests.conftest import FakeProvider


def _classifier(canned_text) -> TicketClassifier:
    return TicketClassifier(
        EmailPreprocessor(),
        FakeProvider(canned_text),
        PromptBuilder(),
    )


# --------------------------------------------------------------------------- #
# Happy paths — valid JSON recovered, no fallback
# --------------------------------------------------------------------------- #

def test_clean_json_returns_correct_label_without_fallback():
    clf = _classifier('{"label": "billing", "confidence": 0.95, "reasoning": "x"}')
    result = clf.classify("My invoice is wrong.")
    assert isinstance(result, ClassificationResult)
    assert result.label == "billing"
    assert result.fallback_used is False


def test_markdown_fenced_json_is_recovered_without_fallback():
    canned = '```json\n{"label": "technical", "confidence": 0.9}\n```'
    clf = _classifier(canned)
    result = clf.classify("The app crashes on login.")
    assert result.label == "technical"
    assert result.fallback_used is False


def test_prose_surrounding_json_is_recovered_without_fallback():
    canned = 'Sure, here is my answer: {"label": "complaint", "confidence": 0.8} hope this helps!'
    clf = _classifier(canned)
    result = clf.classify("Your agent was rude to me.")
    assert result.label == "complaint"
    assert result.fallback_used is False


# --------------------------------------------------------------------------- #
# Degrade-to-general paths — must never raise
# --------------------------------------------------------------------------- #

def test_malformed_json_falls_back_to_general():
    clf = _classifier('{"label": "billing", confidence: }')
    result = clf.classify("anything")
    assert result.label == config.FALLBACK_LABEL == "general"
    assert result.fallback_used is True


def test_truncated_json_falls_back_to_general():
    clf = _classifier('{"label": "billing", "confidence": 0.9')  # no closing brace
    result = clf.classify("anything")
    assert result.label == "general"
    assert result.fallback_used is True


def test_empty_response_falls_back_to_general():
    clf = _classifier("")
    result = clf.classify("anything")
    assert result.label == "general"
    assert result.fallback_used is True


def test_missing_label_field_falls_back_to_general():
    clf = _classifier('{"confidence": 0.9, "reasoning": "x"}')
    result = clf.classify("anything")
    assert result.label == "general"
    assert result.fallback_used is True


def test_unknown_label_falls_back_to_general():
    clf = _classifier('{"label": "spam", "confidence": 0.99}')
    result = clf.classify("Buy cheap pills now")
    assert result.label == "general"
    assert result.fallback_used is True


def test_confidence_below_threshold_falls_back_to_general():
    clf = _classifier('{"label": "billing", "confidence": 0.3}')
    result = clf.classify("My invoice is wrong.")
    assert result.label == "general"
    assert result.fallback_used is True


@pytest.mark.parametrize(
    "canned",
    [
        '{"label": "billing", "confidence": 1.5}',   # above 1.0
        '{"label": "billing", "confidence": -0.5}',  # below 0.0
        '{"label": "billing", "confidence": "high"}',  # non-numeric
        '{"label": "billing"}',  # missing confidence entirely
    ],
)
def test_out_of_range_or_missing_confidence_never_raises(canned):
    clf = _classifier(canned)
    result = clf.classify("My invoice is wrong.")
    assert 0.0 <= result.confidence <= 1.0
    assert result.label in config.CATEGORIES


def test_confidence_above_one_is_clamped_and_label_kept():
    clf = _classifier('{"label": "billing", "confidence": 1.5}')
    result = clf.classify("My invoice is wrong.")
    # Clamped to 1.0 which is >= threshold, so the valid label survives.
    assert result.confidence == 1.0
    assert result.label == "billing"
    assert result.fallback_used is False


# --------------------------------------------------------------------------- #
# Input edge cases
# --------------------------------------------------------------------------- #

def test_empty_input_string_does_not_raise_and_returns_result():
    clf = _classifier('{"label": "general", "confidence": 0.5}')
    result = clf.classify("")
    assert isinstance(result, ClassificationResult)
    assert result.label in config.CATEGORIES


def test_result_label_always_in_configured_categories():
    for canned in [
        '{"label": "billing", "confidence": 0.9}',
        '{"label": "spam", "confidence": 0.9}',
        "total nonsense",
        "",
    ]:
        clf = _classifier(canned)
        result = clf.classify("some email text")
        assert result.label in config.CATEGORIES
