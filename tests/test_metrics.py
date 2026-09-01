"""Tests for evals/metrics.py against a small hand-computed fixture.

Runs fully offline (no API key, no network). The fixture has five records over
labels billing/technical/urgent/general with one fallback, chosen so every
metric can be checked by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable regardless of how pytest is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evals import metrics  # noqa: E402

LABELS = ["billing", "technical", "complaint", "urgent", "feedback", "general"]


@pytest.fixture
def records() -> list[dict]:
    base = {"fallback_used": False, "latency_ms": 100.0, "prompt_tokens": 50, "completion_tokens": 10}
    return [
        {**base, "predicted": "billing", "actual": "billing"},      # correct
        {**base, "predicted": "billing", "actual": "technical"},    # billing FP / technical FN
        {**base, "predicted": "technical", "actual": "technical"},  # correct
        {**base, "predicted": "general", "actual": "urgent", "fallback_used": True},  # fallback
        {**base, "predicted": "billing", "actual": "billing"},      # correct
    ]


def test_accuracy(records):
    assert metrics.accuracy(records) == pytest.approx(0.6)  # 3/5


def test_fallback_rate(records):
    assert metrics.fallback_rate(records) == pytest.approx(0.2)  # 1/5


def test_mean_latency(records):
    assert metrics.mean_latency_ms(records) == pytest.approx(100.0)


def test_token_totals(records):
    assert metrics.token_totals(records) == {
        "prompt_tokens": 250,
        "completion_tokens": 50,
        "total_tokens": 300,
    }


def test_per_class_metrics(records):
    pc = metrics.per_class_metrics(records, LABELS)
    # billing: TP=2, FP=1, FN=0
    assert pc["billing"]["precision"] == pytest.approx(2 / 3)
    assert pc["billing"]["recall"] == pytest.approx(1.0)
    assert pc["billing"]["f1"] == pytest.approx(0.8)
    assert pc["billing"]["support"] == 2
    # technical: TP=1, FP=0, FN=1
    assert pc["technical"]["precision"] == pytest.approx(1.0)
    assert pc["technical"]["recall"] == pytest.approx(0.5)
    assert pc["technical"]["f1"] == pytest.approx(2 / 3)
    assert pc["technical"]["support"] == 2
    # urgent: never predicted -> recall 0, support 1
    assert pc["urgent"]["f1"] == 0.0
    assert pc["urgent"]["support"] == 1
    # general: predicted once but never the true label -> precision 0, support 0
    assert pc["general"]["precision"] == 0.0
    assert pc["general"]["support"] == 0
    # unused labels are present and zeroed
    assert pc["complaint"]["f1"] == 0.0
    assert pc["feedback"]["support"] == 0


def test_confusion_matrix_layout(records):
    cm = metrics.confusion_matrix(records, LABELS)
    # dict-of-dicts keyed by category names, fully square over LABELS
    assert set(cm.keys()) == set(LABELS)
    for row in cm.values():
        assert set(row.keys()) == set(LABELS)
    assert cm["billing"]["billing"] == 2
    assert cm["technical"]["billing"] == 1
    assert cm["technical"]["technical"] == 1
    assert cm["urgent"]["general"] == 1
    assert sum(cm["complaint"].values()) == 0


def test_top_confusions(records):
    cm = metrics.confusion_matrix(records, LABELS)
    top = metrics.top_confusions(cm, n=3)
    assert {(a, p, c) for a, p, c in top} == {
        ("technical", "billing", 1),
        ("urgent", "general", 1),
    }


def test_summarize(records):
    s = metrics.summarize(records, LABELS)
    assert s["sample_size"] == 5
    assert s["accuracy"] == pytest.approx(0.6)
    assert s["fallback_rate"] == pytest.approx(0.2)
    assert s["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 6)
    assert s["total_tokens"] == 300
    assert set(s["confusion_matrix"].keys()) == set(LABELS)
