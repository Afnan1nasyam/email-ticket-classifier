#!/usr/bin/env python3
"""Verify the environment can reach Groq and get a completion — Phase 1 checkpoint.

Routes through the same `GroqProvider` the application uses (so it also
validates the provider layer and its TLS/retry setup), rather than calling the
`groq` SDK directly. Importing `app.llm_provider` injects the OS trust store,
which handles corporate TLS interception (e.g. Zscaler).

Run after completing the Setup steps in the top-level README:

    python scripts/verify_setup.py

Exit codes:
  0  success
  1  unexpected error
  2  GROQ_API_KEY not set
  3  network / TLS failure (could not reach Groq)
  4  request rejected (invalid key, or blocked by a corporate web filter)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.llm_provider import GroqProvider  # noqa: E402


def main() -> int:
    """Run the connectivity check and return a process exit code."""
    try:
        provider = GroqProvider()
    except RuntimeError as exc:  # missing/empty GROQ_API_KEY
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    try:
        resp = provider.complete(
            system_prompt="You are a terse assistant. Reply with exactly one word.",
            user_message="Reply with exactly one word: working",
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic maps errors to exit codes
        # Avoid importing groq here so this stays the only-in-llm_provider rule.
        # groq's HTTP errors expose a `status_code`; connection errors don't.
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__
        if status in (401, 403):
            print(
                f"FAIL: Groq returned HTTP {status} — invalid API key, or the URL "
                "is blocked by a web filter (e.g. corporate proxy).",
                file=sys.stderr,
            )
            return 4
        if status is not None:
            print(f"FAIL: Groq returned HTTP {status} ({name}).", file=sys.stderr)
            return 4
        if any(k in name for k in ("Connection", "Timeout")):
            print(
                f"FAIL: could not reach Groq ({name}). Check network/TLS connectivity.",
                file=sys.stderr,
            )
            return 3
        print(f"FAIL: unexpected error: {exc}", file=sys.stderr)
        return 1

    print("OK: Groq call succeeded")
    print(f"  model             : {resp.model_id}")
    print(f"  response          : {resp.text!r}")
    print(f"  prompt_tokens     : {resp.prompt_tokens}")
    print(f"  completion_tokens : {resp.completion_tokens}")
    print(f"  total_tokens      : {resp.total_tokens}")
    print(f"  latency_ms        : {resp.latency_ms:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
