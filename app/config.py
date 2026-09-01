"""Central configuration for the email/ticket classifier.

All tunable values, category definitions, the model ID, and inference parameters
live here — nowhere else in the codebase. Values are loaded from the environment
via python-dotenv; everything else is a named constant so there are no magic
values scattered through the code.

The category descriptions in this file are load-bearing: the Phase 2 dataset
labels are derived from them, and the Phase 5 prompt is built from them. Each
description states what the category *is* and, where it could compete with a
neighbour on the same email, which one wins and why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root on import, regardless of the current working
# directory. Values already present in the real environment (e.g. set in a
# deployment dashboard) take precedence and are not overridden.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #

# Precedence rule for the deliberate `urgent` overlap. This exact wording must
# appear verbatim — no paraphrasing — in the prompt template and in
# data/README.md. Do not edit it in one place without editing the others.
URGENT_PRECEDENCE_RULE: str = (
    "Label as urgent only when the email states time-critical business impact "
    "or an explicit deadline within roughly 24-48 hours. Frustrated or emphatic "
    "tone alone is not urgency. Otherwise apply the topical category."
)

# Six predefined classes, defined once, here. `general` doubles as the
# low-confidence fallback label. Descriptions draw the boundary against the
# nearest competing category explicitly.
CATEGORIES: dict[str, str] = {
    "billing": (
        "Money and the account ledger: invoices, incorrect or unexpected "
        "charges, refunds, failed or duplicate payments, subscription pricing, "
        "and plan up/downgrades. Wins over technical when the core request is "
        "about money owed, paid, or refunded — even if a software glitch caused "
        "the charge. The remedy the sender wants is financial, not a fix."
    ),
    "technical": (
        "The product not behaving as intended: bugs, errors, crashes, outages, "
        "API/integration failures, and setup or configuration problems. Wins "
        "over complaint when the email reports a product defect rather than "
        "dissatisfaction with a person or process. The remedy the sender wants "
        "is a fix that makes the product work."
    ),
    "complaint": (
        "Dissatisfaction with the service, staff, or how a situation was "
        "handled, seeking acknowledgement, apology, or redress. Wins over "
        "feedback when the dominant intent is a grievance about treatment "
        "received rather than a constructive idea. Wins over technical when the "
        "problem is with people or process, not a product defect. About how the "
        "sender was treated, not about a broken feature."
    ),
    # Base definition only — kept free of the precedence-rule wording so v1 (which
    # renders these definitions) has no tiebreak rule. The overlap tiebreak lives
    # in URGENT_PRECEDENCE_RULE and enters the prompt at v3.
    "urgent": (
        "Time-critical business impact or an explicit near-term deadline — a "
        "request that genuinely cannot wait. Can overlap the topical categories "
        "when a message is both time-critical and about a specific topic."
    ),
    "feedback": (
        "Suggestions, feature requests, and praise, offered constructively — no "
        "defect to fix and no grievance to resolve. Wins over complaint when the "
        "intent is to improve the product or to compliment, rather than to seek "
        "redress for something that went wrong. Forward-looking or positive, not "
        "a demand for resolution."
    ),
    # Last-resort fallback, not a tiebreaker: genuine overlaps are resolved by
    # the topical definitions and the urgent rule, not by falling back to general.
    "general": (
        "Everything that does not clearly fit another category: greetings, "
        "account or policy questions with no billing angle, ambiguous or "
        "mixed-topic messages, and the low-confidence fallback. Use only when no "
        "other category clearly applies."
    ),
}

# Ordered list of valid labels, for validation and prompt rendering.
CATEGORY_LABELS: list[str] = list(CATEGORIES)

# The label used both as the catch-all category and as the fallback when the
# model's confidence is below CONFIDENCE_THRESHOLD or its output is invalid.
FALLBACK_LABEL: str = "general"


def is_valid_label(label: str) -> bool:
    """Return True if `label` is one of the configured categories."""
    return label in CATEGORIES


# --------------------------------------------------------------------------- #
# Model, provider, and inference parameters
# --------------------------------------------------------------------------- #

# The model ID lives here and nowhere else. Swapping providers/models is a change
# to this file only. (The previous model, llama-3.3-70b-versatile, was shut down
# on Groq on 2026-08-16; the abstraction made that a one-line change.)
MODEL_ID: str = "openai/gpt-oss-120b"

# Deterministic decoding, so the eval is as reproducible as a hosted LLM allows.
TEMPERATURE: float = 0.0

# Reasoning effort is a *measured eval variable*, not a fixed constant. `low` is
# the default because reasoning tokens count against the TPM budget; the eval
# runner can override it via --reasoning-effort. Accepts "low" | "medium" | "high".
# NOTE: reasoning_format is NOT supported on gpt-oss-120b and must never be set.
REASONING_EFFORT: str = os.environ.get("REASONING_EFFORT", "low")

# gpt-oss models default to returning reasoning in a separate `reasoning` field;
# suppress it and read the classification JSON only from message.content.
INCLUDE_REASONING: bool = False


# --------------------------------------------------------------------------- #
# Classification behavior
# --------------------------------------------------------------------------- #

# Below this self-reported confidence, a result is downgraded to `general` with
# fallback_used=True. Self-reported confidence is poorly calibrated, so this is a
# value to tune against eval data, not a hard truth. The eval reports fallback
# rate separately from accuracy.
CONFIDENCE_THRESHOLD: float = 0.6

# Reject API payloads whose raw email text exceeds this many characters, before
# any expensive work happens.
MAX_INPUT_CHARS: int = 10_000

# Truncation budget for the email body, in (approximate) tokens. The preprocessor
# approximates ~4 characters per token with a safety margin.
MAX_EMAIL_TOKENS: int = 512

# Prompt template the classifier uses by default (the API path). The eval runner
# overrides this per run via --prompt. Updated to the winning version (copied to
# ACTIVE.txt) at the end of Phase 8; until then the v1 baseline.
ACTIVE_PROMPT_TEMPLATE: str = "v1_zero_shot.txt"


# --------------------------------------------------------------------------- #
# Rate limiting (Groq free tier)
# --------------------------------------------------------------------------- #

# The binding constraint is TOKENS per minute, not requests per minute.
TPM_LIMIT: int = 6_000
RPM_LIMIT: int = 30
RPD_LIMIT: int = 14_400

# The rolling token window has no history on the first call of a run; seed it
# with a conservative estimate so an early burst cannot trip HTTP 429.
THROTTLE_SEED_TOKENS: int = 1_200

# Provider retry policy for 429 / 5xx responses.
MAX_RETRIES: int = 5
BACKOFF_BASE_SECONDS: float = 2.0
REQUEST_TIMEOUT_SECONDS: float = 60.0


# --------------------------------------------------------------------------- #
# App metadata
# --------------------------------------------------------------------------- #

APP_VERSION: str = "0.1.0"


def get_api_key() -> str:
    """Return the Groq API key from the environment.

    Raises:
        RuntimeError: if GROQ_API_KEY is not set. The key value itself is never
            included in the error message.
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime settings, for injection and /health output.

    Deliberately excludes the API key — settings objects get logged and returned
    in responses, and the key must never appear there.
    """

    model_id: str = MODEL_ID
    temperature: float = TEMPERATURE
    reasoning_effort: str = REASONING_EFFORT
    include_reasoning: bool = INCLUDE_REASONING
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    max_input_chars: int = MAX_INPUT_CHARS
    max_email_tokens: int = MAX_EMAIL_TOKENS
    app_version: str = APP_VERSION


SETTINGS = Settings()
