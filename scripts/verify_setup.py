#!/usr/bin/env python3
"""Verify the environment can reach Groq and get a completion — Phase 1 checkpoint.

Loads configuration, makes ONE real Groq call, and reports the result. This is a
low-level connectivity diagnostic: it deliberately exercises the Groq SDK
*directly* to confirm the key, model ID, SDK, and network path all work.
Application code classifies through the LLMProvider abstraction in
app/llm_provider.py, not through this script.

Run after completing the Setup steps in the top-level README:

    python scripts/verify_setup.py

Exit codes:
  0  success
  1  unexpected error (or SDK not installed)
  2  GROQ_API_KEY not set
  3  network / TLS failure (could not reach Groq)
  4  request rejected (e.g. invalid key, or blocked by a corporate web filter)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import truststore

# Trust the OS certificate store so corporate TLS interception (e.g. Zscaler) is
# handled without disabling verification. A no-op on machines with no custom
# root CA. Must run before any HTTPS client is created.
truststore.inject_into_ssl()

# Make the project importable regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app import config  # noqa: E402


def main() -> int:
    """Run the connectivity check and return a process exit code."""
    try:
        api_key = config.get_api_key()
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    try:
        from groq import APIConnectionError, APIStatusError, Groq
    except ImportError as exc:
        print(
            f"FAIL: groq SDK not importable ({exc}). "
            "Did you run: pip install -r requirements.txt ?",
            file=sys.stderr,
        )
        return 1

    client = Groq(api_key=api_key, timeout=config.REQUEST_TIMEOUT_SECONDS)

    try:
        started = time.perf_counter()
        resp = client.chat.completions.create(
            model=config.MODEL_ID,
            temperature=config.TEMPERATURE,
            reasoning_effort=config.REASONING_EFFORT,
            include_reasoning=False,  # gpt-oss-120b defaults to True; suppress it
            messages=[{"role": "user", "content": "Reply with exactly one word: working"}],
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
    except APIConnectionError as exc:
        print(
            f"FAIL: could not reach Groq ({exc}). Check network/TLS connectivity.",
            file=sys.stderr,
        )
        return 3
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            hint = "invalid API key, or the URL is blocked by a web filter (e.g. corporate proxy)"
        else:
            hint = "unexpected API status"
        print(f"FAIL: Groq returned HTTP {exc.status_code} — {hint}.", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001 — diagnostic script: report anything
        print(f"FAIL: unexpected error: {exc}", file=sys.stderr)
        return 1

    message = resp.choices[0].message
    usage = resp.usage
    print("OK: Groq call succeeded")
    print(f"  model             : {resp.model}")
    print(f"  response          : {message.content!r}")
    print(f"  prompt_tokens     : {usage.prompt_tokens}")
    print(f"  completion_tokens : {usage.completion_tokens}")
    print(f"  total_tokens      : {usage.total_tokens}")
    print(f"  latency_ms        : {latency_ms:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
