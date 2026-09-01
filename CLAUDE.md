# CLAUDE.md

Persistent context for this repository. Read this before doing anything.

---

## What this project is

An AI-powered email/ticket classification system. A Flask REST API accepts raw
customer email text and returns a category label, using a hosted LLM for
classification.

### Engineering standards this repo holds to

| Standard | Where it holds |
|---|---|
| Built with Python and Flask | `app/`, `run.py` |
| Integrates an LLM API (Groq) | `app/llm_provider.py` |
| Categorizes emails into 6+ predefined classes | `app/config.py` |
| `/classify` and `/health` REST endpoints | `app/routes.py` |
| Object-oriented design | classes, not loose functions |
| Prompt engineering, versioned | `prompts/` |
| 90%+ classification accuracy, measured | `evals/` |
| pandas-based preprocessing, cleaning, tokenization | `app/preprocessor.py` |
| Complete lifecycle from design to deployment | `Dockerfile`, deployed URL in README |

**On accuracy.** The accuracy figure must come from an actual eval run against a
held-out labeled dataset, and the run must be reproducible by anyone who clones
the repo. It is measured, never asserted — a hardcoded or hand-waved number is
not acceptable. The target is 90%+. If the measured number lands below that,
report the real number and explain the ceiling; never adjust the dataset or the
eval to reach the target. See the Phase 8 note in `PLAN.md`.

---

## Environment constraints

- **OS:** Windows. Use PowerShell-compatible commands. No bash-isms in docs.
- **Python:** 3.14 only. No other version is installed and none may be installed.
  If a dependency does not support 3.14, find an alternative — do not suggest
  downgrading Python.
- **Virtual env:** `venv/` in project root, always activated before any pip or
  python command.
- **Serving:** `gunicorn` is used in the `Dockerfile` only (Linux). For local
  Windows runs use the Flask dev server (`run.py`); add `waitress` only if a
  production-like local server is needed. Do not put `gunicorn` in the local run
  path — it does not run on Windows.
- **Deployment target:** Render free tier. Cold starts (~30-50s after idle) are
  documented in the README. The repo and `EVAL_LOG.md` are the durable artifacts;
  the live URL is a bonus.

### Dependency compatibility (verified 2026-08-31, Python 3.14 / win_amd64)

All five deps have installable cp314/pure-Python wheels — no source build needed:
`groq 1.7.0`, `flask 3.1.3`, `python-dotenv 1.2.3`, `pandas 3.0.5`
(native `cp314-cp314-win_amd64` wheel), `pytest 9.1.1`. Note pandas resolves to
the **3.x** line, not 2.x; pin `pandas>=3,<4` and avoid 2.x-only idioms.

---

## Model and provider

- **Current model:** `openai/gpt-oss-120b` on Groq.
- **Do not use** `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`. Both were
  shut down on Groq on 2026-08-16. Requests to them return errors.
- The model ID lives in `app/config.py` and **nowhere else**. No hardcoded model
  strings anywhere in the codebase.
- **Inference parameters, also in `app/config.py`:** `temperature=0` (reproducible
  eval) and `reasoning_effort` (default `"low"`, accepts `low`/`medium`/`high`).
  `reasoning_effort` is a **measured eval variable, not a fixed constant** — low
  is the iteration default because reasoning tokens count against the 6,000 TPM
  budget, but the eval can raise it (see Phase 8).
- **Reasoning payload handling:** `reasoning_format` is **not supported** on
  gpt-oss-120b — do not set it anywhere. gpt-oss models return reasoning in a
  separate `reasoning` field by default; set `include_reasoning=false` in
  `GroqProvider` to suppress it. Parse the classification JSON only from
  `message.content`. Reasoning tokens still count toward `completion_tokens` and
  therefore TPM; the adaptive window handles this since it uses returned counts.
- All LLM calls go through the `LLMProvider` abstraction in
  `app/llm_provider.py`. Nothing else imports the `groq` SDK directly. This is a
  hard rule — the whole point is that swapping providers is a config change.

### Groq free tier rate limits — read this before writing the eval loop

- 30 requests/minute
- **6,000 tokens/minute** ← this is the binding constraint, not RPM
- 14,400 requests/day

With a few-shot prompt of roughly 700-900 tokens per call, the TPM ceiling caps
you at about **6-8 requests per minute**, not 30. A naive eval loop over 80
emails will hit HTTP 429 within seconds.

Requirements:
- The eval runner must throttle **adaptively**: `GroqProvider` records the actual
  `prompt_tokens` and `completion_tokens` returned by each call, and the runner
  maintains a rolling 60-second token window, sleeping only until the next call
  fits under the 6,000 TPM ceiling. Do not pre-estimate token counts with a fixed
  delay — completion length varies and a fixed sleep is either too slow or trips
  429.
- The provider must retry 429s with exponential backoff.
- Expect a full 80-email eval run to take 10-15 minutes. That is normal. Do not
  "optimize" it by removing the throttle.
- During prompt iteration, use a 40-email subset. Use the full set only for
  numbers that go in the README.

---

## Code conventions

- OOP where it earns its place. `EmailPreprocessor`, `TicketClassifier`,
  `LLMProvider` / `GroqProvider` are classes. Utility helpers can be functions.
- Type hints on all public methods.
- Docstrings on all classes and public methods.
- Config via `python-dotenv` and `app/config.py`. No magic values scattered
  through the code.
- Structured logging (`logging` module, not `print`) in application code.
- Every LLM response is validated before use. Assume the model will occasionally
  return malformed output and handle it explicitly.

---

## Security rules

- `.env` is in `.gitignore` **before the first commit**. Never after.
- Never log, print, echo, or include the API key in any output, error message,
  traceback, commit, or test fixture.
- `.env.example` ships with placeholder values only.
- If a key is ever committed, say so immediately and clearly — it must be
  rotated, not just removed from the working tree.

---

## Things not to do

- Do not build a UI. This is an API; a UI is out of scope.
- Do not add a database. Stateless classification, no persistence needed.
- Do not add scikit-learn, transformers, or any local model. The system is
  LLM-API-based classification.
- Do not add a tokenizer dependency. Tokenization is a regex word-boundary
  tokenizer in `app/preprocessor.py`. Do not imply anywhere — README included —
  that it is the model's own tokenizer.
- Do not invent accuracy numbers, even as placeholders in the README. Leave
  `TBD` until a real eval run produces a figure.
- Do not write the eval so that the same prompt that classifies also produced
  the ground-truth labels. See `data/README.md` for why this matters.
- Do not skip the eval log. The record of prompt v1 → v2 → v3 with measured
  deltas is the primary artifact documenting the prompt-engineering process.

---

## Local-only working docs

`CLAUDE.local.md` and `PLAN.local.md` hold additional local working context and
are gitignored (Claude Code does not auto-load `CLAUDE.local.md`). Reference them
explicitly when local context matters. `PROMPTS.md` is local build scaffolding,
not part of the deliverable.
