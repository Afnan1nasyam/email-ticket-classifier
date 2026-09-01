#!/usr/bin/env python3
"""Evaluation runner: classify a labeled dataset and record accuracy metrics.

Reuses the same `TicketClassifier` the API uses (so the measured path is the
production path), throttles adaptively with `TokenRateLimiter` to respect Groq's
per-minute token ceiling, and writes a full per-run JSON plus a human-readable
entry appended to EVAL_LOG.md. Results are accumulated in memory and written
once, so an interruption or a provider failure never leaves a half-written file.

Example:
    python evals/run_eval.py --prompt prompts/v1_zero_shot.txt
    python evals/run_eval.py --prompt prompts/v3_few_shot_with_rules.txt \\
        --limit 40 --reasoning-effort medium
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from app import config
from app.classifier import TicketClassifier
from app.llm_provider import GroqProvider, TokenRateLimiter
from app.preprocessor import EmailPreprocessor
from app.prompt_builder import PromptBuilder
from evals import metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Run the classifier against the labeled dataset and record metrics.",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Path to a prompt template file (e.g. prompts/v1_zero_shot.txt).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of rows evaluated (default: all rows).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Minimum seconds between calls (safety floor; the real throttle is "
             "the adaptive TokenRateLimiter).",
    )
    parser.add_argument(
        "--reasoning-effort", default=config.REASONING_EFFORT,
        choices=["low", "medium", "high"],
        help=f"Reasoning effort passed to the provider (default: {config.REASONING_EFFORT}).",
    )
    parser.add_argument(
        "--dataset", default=str(_PROJECT_ROOT / "data" / "test_dataset.csv"),
        help="Path to the dataset CSV (default: data/test_dataset.csv).",
    )
    parser.add_argument(
        "--output-dir", default=str(_PROJECT_ROOT / "evals" / "results"),
        help="Directory for the JSON result and EVAL_LOG.md (default: evals/results/).",
    )
    return parser.parse_args(argv)


def _format_eval_log_entry(header: dict, summary: dict, partial: bool) -> str:
    """Render a Markdown EVAL_LOG.md entry from a run header and metrics."""
    lines: list[str] = []
    flag = " — PARTIAL" if partial else ""
    lines.append(
        f"## {header['timestamp_utc']} — {header['prompt_name']} "
        f"(reasoning_effort={header['reasoning_effort']}){flag}"
    )
    lines.append("")
    lines.append(f"- status: {header['status']}")
    lines.append(f"- prompt: `{header['prompt']}`")
    lines.append(
        f"- dataset: `{header['dataset']}` "
        f"({header['sample_size']}/{header['dataset_size']} rows)"
    )
    lines.append(
        f"- model: `{header['model_id']}`, temperature={header['temperature']}, "
        f"reasoning_effort={header['reasoning_effort']}"
    )
    lines.append(
        f"- **accuracy: {summary['accuracy'] * 100:.1f}%** "
        f"| macro-F1: {summary['macro_f1']:.3f} "
        f"| fallback rate: {summary['fallback_rate'] * 100:.1f}%"
    )
    lines.append(
        f"- mean latency: {summary['mean_latency_ms']:.0f} ms "
        f"| tokens: prompt={summary['prompt_tokens']}, "
        f"completion={summary['completion_tokens']}, total={summary['total_tokens']}"
    )
    lines.append("")
    lines.append("Per-class metrics:")
    lines.append("")
    lines.append("| label | precision | recall | f1 | support |")
    lines.append("|---|---|---|---|---|")
    for label, m in summary["per_class"].items():
        lines.append(
            f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} "
            f"| {m['f1']:.2f} | {m['support']} |"
        )
    lines.append("")
    lines.append("Top confusions:")
    top = summary["top_confusions"]
    if top:
        for actual, predicted, count in top:
            lines.append(f"- {actual} misread as {predicted}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("notes: ")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation. Returns a process exit code."""
    args = parse_args(argv)

    prompt_path = Path(args.prompt).resolve()
    if not prompt_path.is_file():
        print(f"error: prompt template not found: {prompt_path}", file=sys.stderr)
        return 2
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.is_file():
        print(f"error: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    dataset_size = len(df)
    if args.limit is not None:
        df = df.head(args.limit)

    preprocessor = EmailPreprocessor()
    df = preprocessor.preprocess_batch(df)  # adds clean_text via the pandas path

    provider = GroqProvider(reasoning_effort=args.reasoning_effort)
    builder = PromptBuilder(prompts_dir=prompt_path.parent)
    classifier = TicketClassifier(
        preprocessor, provider, builder, template=prompt_path.name
    )

    limiter = TokenRateLimiter()
    records: list[dict] = []
    partial = False
    status = "complete"
    exit_code = 0
    last_total: int | None = None
    last_call_ts: float | None = None
    total = len(df)
    started = time.perf_counter()

    try:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            if args.delay > 0 and last_call_ts is not None:
                elapsed = time.monotonic() - last_call_ts
                if elapsed < args.delay:
                    time.sleep(args.delay - elapsed)

            estimate = last_total if last_total is not None else config.THROTTLE_SEED_TOKENS
            limiter.acquire(estimate)
            result = classifier.classify_preprocessed(row.clean_text)
            last_call_ts = time.monotonic()
            limiter.record(result.total_tokens)
            last_total = result.total_tokens

            records.append(
                {
                    "id": int(row.id),
                    "email_preview": str(row.email_text)[:100],
                    "true_label": row.true_label,
                    "predicted_label": result.label,
                    # metrics-facing keys:
                    "actual": row.true_label,
                    "predicted": result.label,
                    "fallback_used": result.fallback_used,
                    "confidence": result.confidence,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_ms": result.latency_ms,
                }
            )
            print(
                f"[{i}/{total}] id={int(row.id)} pred={result.label} "
                f"true={row.true_label} ({result.latency_ms:.0f}ms, "
                f"{result.total_tokens} tok)",
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        partial, status, exit_code = True, "INTERRUPTED (Ctrl+C)", 130
        print("\ninterrupted; writing partial results...", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - record failure, write partial, exit non-zero
        partial, status, exit_code = True, f"ABORTED: {type(exc).__name__}: {exc}", 1
        print(f"\nrun aborted ({type(exc).__name__}); writing partial results...", file=sys.stderr)

    if not records:
        print("no records produced; nothing written.", file=sys.stderr)
        return exit_code or 1

    summary = metrics.summarize(records, labels=config.CATEGORY_LABELS)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    header = {
        "timestamp_utc": timestamp,
        "prompt": str(prompt_path),
        "prompt_name": prompt_path.name,
        "dataset": str(dataset_path),
        "dataset_size": dataset_size,
        "sample_size": len(records),
        "model_id": config.MODEL_ID,
        "reasoning_effort": args.reasoning_effort,
        "temperature": config.TEMPERATURE,
        "partial": partial,
        "status": status,
        "wall_clock_seconds": round(time.perf_counter() - started, 1),
    }

    rows_out = [
        {
            "id": r["id"],
            "email_preview": r["email_preview"],
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "fallback_used": r["fallback_used"],
            "confidence": r["confidence"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "latency_ms": r["latency_ms"],
        }
        for r in records
    ]

    basename = prompt_path.stem
    json_path = output_dir / f"{timestamp}_{basename}_{args.reasoning_effort}.json"
    json_path.write_text(
        json.dumps({"run": {**header, "metrics": summary}, "records": rows_out}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {json_path}", file=sys.stderr)

    log_path = output_dir / "EVAL_LOG.md"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(_format_eval_log_entry(header, summary, partial))
    print(f"appended entry to {log_path}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
