"""Pure metric functions for classifier evaluation.

No I/O, no globals, no config import — every function takes its data as
arguments and returns plain values, so the whole module is trivially testable.

A "record" is a mapping with at least these keys:
    predicted:          str   - the predicted label
    actual:             str   - the ground-truth label
    fallback_used:      bool  - whether the result was forced to the fallback
    latency_ms:         float - provider latency for the call
    prompt_tokens:      int
    completion_tokens:  int

Extra keys are ignored, so per-row records carrying `id`, `confidence`, etc.
can be passed straight through.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

Record = Mapping[str, Any]


def _labels_from(records: Sequence[Record], labels: Sequence[str] | None) -> list[str]:
    """Return the label set to report over: explicit `labels`, else observed."""
    if labels is not None:
        return list(labels)
    seen: list[str] = []
    for r in records:
        for key in ("actual", "predicted"):
            value = r[key]
            if value not in seen:
                seen.append(value)
    return sorted(seen)


def accuracy(records: Sequence[Record]) -> float:
    """Fraction of records where predicted == actual (0.0 if empty)."""
    if not records:
        return 0.0
    correct = sum(1 for r in records if r["predicted"] == r["actual"])
    return correct / len(records)


def fallback_rate(records: Sequence[Record]) -> float:
    """Fraction of records where the fallback category was used (0.0 if empty)."""
    if not records:
        return 0.0
    return sum(1 for r in records if r["fallback_used"]) / len(records)


def confusion_matrix(
    records: Sequence[Record], labels: Sequence[str] | None = None
) -> dict[str, dict[str, int]]:
    """Confusion matrix as ``matrix[actual][predicted] = count``.

    Rows are the true labels, columns the predicted labels. Every label in
    `labels` appears as both a row and a column (zero-filled), so the layout is
    stable across runs.
    """
    label_list = _labels_from(records, labels)
    matrix = {a: {p: 0 for p in label_list} for a in label_list}
    for r in records:
        actual, predicted = r["actual"], r["predicted"]
        # Defensively include any label not in the provided set.
        matrix.setdefault(actual, {p: 0 for p in label_list})
        if predicted not in matrix[actual]:
            matrix[actual][predicted] = 0
        matrix[actual][predicted] += 1
    return matrix


def per_class_metrics(
    records: Sequence[Record], labels: Sequence[str] | None = None
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1, and support.

    Precision/recall/F1 are 0.0 when their denominator is 0 (e.g. a label the
    model never predicts has precision 0.0). `support` is the count of records
    whose true label is that class.
    """
    label_list = _labels_from(records, labels)
    predicted_counts = Counter(r["predicted"] for r in records)
    actual_counts = Counter(r["actual"] for r in records)
    tp_counts = Counter(r["predicted"] for r in records if r["predicted"] == r["actual"])

    out: dict[str, dict[str, float]] = {}
    for label in label_list:
        tp = tp_counts.get(label, 0)
        fp = predicted_counts.get(label, 0) - tp
        fn = actual_counts.get(label, 0) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_counts.get(label, 0),
        }
    return out


def top_confusions(
    matrix: Mapping[str, Mapping[str, int]], n: int = 3
) -> list[tuple[str, str, int]]:
    """Return the `n` most frequent off-diagonal (actual, predicted, count)."""
    pairs = [
        (actual, predicted, count)
        for actual, row in matrix.items()
        for predicted, count in row.items()
        if actual != predicted and count > 0
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:n]


def mean_latency_ms(records: Sequence[Record]) -> float:
    """Mean provider latency in ms (0.0 if empty)."""
    if not records:
        return 0.0
    return sum(float(r["latency_ms"]) for r in records) / len(records)


def token_totals(records: Sequence[Record]) -> dict[str, int]:
    """Summed prompt, completion, and total tokens across records."""
    prompt = sum(int(r["prompt_tokens"]) for r in records)
    completion = sum(int(r["completion_tokens"]) for r in records)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def summarize(
    records: Sequence[Record], labels: Sequence[str] | None = None
) -> dict[str, Any]:
    """Aggregate every metric into one JSON-serializable dict."""
    per_class = per_class_metrics(records, labels)
    matrix = confusion_matrix(records, labels)
    f1s = [m["f1"] for m in per_class.values()]
    tokens = token_totals(records)
    return {
        "sample_size": len(records),
        "accuracy": accuracy(records),
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "fallback_rate": fallback_rate(records),
        "mean_latency_ms": mean_latency_ms(records),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "top_confusions": top_confusions(matrix),
        **tokens,
    }
