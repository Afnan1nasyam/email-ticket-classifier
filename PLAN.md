# Build Plan

Ten phases. Each has a checkpoint that must pass before moving on. Do not batch
phases together — the checkpoints exist because a failure caught in phase 3 is
cheap and the same failure caught in phase 9 is not.

Mark each box as it completes.

> **Ordering note (why the dataset comes early).** The labeled test set is built
> in Phase 2, immediately after config and *before any prompt exists*. This makes
> label independence structural rather than a matter of instruction: the ground
> truth is derived from the category definitions alone, so the prompt that later
> classifies cannot have shaped the labels it is scored against. A manual
> hand-review remains as a second layer on top of that.

---

## Phase 0 — Scaffold and environment

- [ ] `venv` created and activated, Python 3.14 confirmed
- [ ] `requirements.txt`: `groq`, `flask`, `python-dotenv`, `pandas>=3,<4`,
      `pytest`. No tokenizer dependency. `gunicorn` is **not** here — it goes in
      the `Dockerfile` only (Linux); local runs use the Flask dev server.
- [ ] Dependencies installed (all confirmed to have cp314/pure-Python wheels on
      2026-08-31 — no source build required)
- [ ] `.gitignore` created **before** any commit (`venv/`, `.env`, `__pycache__/`, `*.pyc`, `evals/results/*.json`)
- [ ] `.env` with real key, `.env.example` with placeholder
- [ ] Full folder tree from ARCHITECTURE.md created, `__init__.py` files in place
- [ ] `git init`, first commit

**Checkpoint:** `git status` shows no `.env` and no `venv/`. If either appears,
stop and fix `.gitignore` before committing anything.

---

## Phase 1 — Config and connectivity

- [ ] `app/config.py`: loads env, defines the six categories with descriptions,
      model ID, confidence threshold (**0.6**, tunable against eval data),
      max input length, `temperature=0`, `reasoning_effort` (default `"low"`, but
      an eval variable — see Phase 8), and the TPM limit (6000) the eval window
      throttles against. Do **not** set `reasoning_format` — it is unsupported on
      gpt-oss-120b.
- [ ] The `urgent` precedence rule appears in `config.py` **verbatim** (identical
      wording to the prompt template and `data/README.md`, no paraphrasing):
      "Label as urgent only when the email states time-critical business impact
      or an explicit deadline within roughly 24-48 hours. Frustrated or emphatic
      tone alone is not urgency. Otherwise apply the topical category."
- [ ] Throwaway smoke test: one real call to `openai/gpt-oss-120b`, print the reply
- [ ] Smoke test deleted after it passes

**Checkpoint:** a real API call succeeds. Nothing else gets built until the key,
the model string, and the SDK are all confirmed working. If this fails, every
later phase fails too, and the cause will be much harder to see.

---

## Phase 2 — Test dataset

Built here, before any prompt exists, so label independence is structural (see
the ordering note at the top). Use the `dataset-builder` subagent.

- [ ] `data/test_dataset.csv` with columns `id, email_text, true_label, difficulty`
- [ ] ~80 rows, roughly balanced across the six categories
- [ ] Difficulty mix: about 50% `easy`, 35% `medium`, 15% `hard`
- [ ] `hard` cases include: billing-and-urgent overlap, complaint-vs-feedback
      ambiguity, very short emails, emails with no clear category
- [ ] Labels derived from the category definitions in `config.py` alone — never
      from a classification prompt
- [ ] Every label hand-reviewed, not accepted on trust
- [ ] `data/README.md` documenting provenance and the synthetic-data caveat, and
      quoting the `urgent` precedence rule **verbatim** (same wording as
      `config.py` and the prompt template)

**Checkpoint:** read all ~80 rows. Disagreeing with a label means the label is
wrong, or the category definition is unclear and needs fixing in `config.py`. Do
not skip this — the entire accuracy figure rests on these labels being
defensible.

---

## Phase 3 — Preprocessor

- [ ] `EmailPreprocessor` class
- [ ] `clean(text) -> str`: strip HTML, signatures, quoted replies, normalize
      whitespace, collapse repeated punctuation
- [ ] `tokenize(text) -> list[str]`: **regex word-boundary tokenizer**, no
      external dependency; not the model's tokenizer
- [ ] `truncate(text, max_tokens) -> str`: approximates the budget at ~4
      characters per token with a safety margin
- [ ] `preprocess_batch(df) -> DataFrame`: pandas path used by the eval runner
- [ ] `preprocess_one(text) -> str`: single-item path used by the API

**Checkpoint:** feed it three deliberately ugly emails — HTML tags, a `>` quoted
reply chain, a long signature block — and confirm the output is clean.

---

## Phase 4 — Provider layer

- [ ] `LLMProvider` ABC with `complete(system_prompt, user_message) -> LLMResponse`
- [ ] `LLMResponse`: text, **prompt tokens, completion tokens** (both required for
      adaptive throttling), latency ms, model ID
- [ ] `GroqProvider` implementation; passes `temperature` and `reasoning_effort`
      from config, sets `include_reasoning=false`, never sets `reasoning_format`,
      and reads the classification JSON only from `message.content`
- [ ] Retry with exponential backoff on 429 and 5xx; max attempts configurable
- [ ] Rolling-window helper seeds the first call with a conservative ~1,200-token
      estimate so an early burst cannot trip 429 before real counts accumulate
- [ ] `FakeProvider` in `tests/conftest.py` returning canned responses

**Checkpoint:** `GroqProvider` returns a real completion. `FakeProvider`
satisfies the same interface. Confirm no file outside `llm_provider.py` imports
`groq`.

---

## Phase 5 — Prompt builder and v1 prompt

- [ ] `PromptBuilder` loads a template from `prompts/` and injects email text
- [ ] `prompts/v1_zero_shot.txt`: category definitions, JSON output instruction,
      no examples
- [ ] JSON output contract: `{"label": ..., "confidence": 0.0-1.0, "reasoning": ...}`
- [ ] The `urgent` precedence rule, when it enters a prompt (v3 / ACTIVE), is the
      **verbatim** wording shared with `config.py` and `data/README.md`

**Checkpoint:** the built prompt renders correctly with a sample email and the
model returns parseable JSON.

---

## Phase 6 — Classifier

- [ ] `TicketClassifier`, constructor-injected preprocessor + provider + builder
- [ ] `classify(text) -> ClassificationResult`
- [ ] JSON parsing that survives markdown fences and leading prose. Comment in
      the parser that reasoning text leaking into `message.content` is a real,
      community-reported behavior on this model — prose-before-JSON is not
      hypothetical
- [ ] Label validation against `config.CATEGORIES`; unknown label → `general`
- [ ] Confidence below threshold (0.6) → `general`, `fallback_used=True`
- [ ] Structured logging of label, confidence, latency, tokens — never the key

**Checkpoint:** classify five hand-written emails and confirm sensible labels.
Then feed the parser a deliberately malformed response and confirm it degrades to
`general` instead of raising.

---

## Phase 7 — Flask API

- [ ] App factory in `app/__init__.py`
- [ ] `GET /health` → status, model ID, version. No LLM call.
- [ ] `POST /classify` → accepts `{"email_text": "..."}`, returns the result
- [ ] Input validation: missing field, empty string, oversized payload
- [ ] Error handling: 400 for bad input, 502 for provider failure, 500 otherwise
- [ ] JSON error bodies, never HTML tracebacks
- [ ] `run.py` entrypoint (Flask dev server; Windows-friendly)

**Checkpoint:** both endpoints respond correctly via `curl` or Postman, including
all three failure cases.

---

## Phase 8 — Eval harness and prompt iteration

This phase is the core of the project.

- [ ] `evals/metrics.py`: accuracy, per-class precision/recall/F1, confusion
      matrix, fallback rate, mean latency, total tokens
- [ ] `evals/run_eval.py`: takes `--prompt`, `--limit`, `--delay`,
      `--reasoning-effort`; throttles **adaptively** with a rolling 60-second
      window over the actual `prompt_tokens` + `completion_tokens` returned per
      call, sleeping until the next call fits under the 6,000 TPM ceiling
      (`--delay` is a minimum floor, not the mechanism); writes JSON to
      `evals/results/` and appends to `EVAL_LOG.md`
- [ ] Run v1 on the full dataset. Record the number, whatever it is.
- [ ] Read every misclassification. Look for patterns, not individual mistakes.
- [ ] `prompts/v2_few_shot.txt`: add 2-3 examples per category, chosen to target
      the confusions v1 actually made
- [ ] Run v2. Record.
- [ ] `prompts/v3_few_shot_with_rules.txt`: add explicit precedence rules for the
      overlaps still failing — especially `urgent` vs everything else, using the
      verbatim rule wording
- [ ] Run v3. Record.
- [ ] If accuracy stalls in the high 80s, run the same prompt at
      `--reasoning-effort medium` **before** touching the prompt or the dataset,
      and log it as its own `EVAL_LOG.md` entry. Isolating a model-parameter
      effect from a prompt effect is a finding worth recording either way.
- [ ] Iterate until accuracy clears 90%
- [ ] Copy the winner to `prompts/ACTIVE.txt`

Use a 40-email subset during iteration; the full 80 only for numbers that go in
the README. Expect a full run to take 10-15 minutes — that is the TPM ceiling,
not a bug.

**Checkpoint:** `EVAL_LOG.md` contains a dated entry per run with the prompt
version, accuracy, confusion matrix, and — most importantly — a note on *what was
changed and why*. That narrative documents the prompt-engineering process.

If accuracy stalls below 90%, the honest options are: fix genuinely wrong labels,
tighten ambiguous category definitions, or report the real number and explain the
ceiling. Do not quietly delete hard examples to inflate the score, and do not
adjust the dataset or the eval to reach the target — doing so invalidates the
measurement entirely.

---

## Phase 9 — Tests, docs, deployment

Use the `test-writer` subagent for the test suite.

- [ ] `tests/test_preprocessor.py`: cleaning, tokenizing, truncation, edge cases
- [ ] `tests/test_classifier.py`: uses `FakeProvider` — malformed JSON, unknown
      label, low confidence, empty input
- [ ] `tests/test_api.py`: Flask test client, both endpoints, all error paths
- [ ] `pytest` passes with no API key set in the environment
- [ ] `Dockerfile` (python:3.14-slim, gunicorn)
- [ ] Deploy to Render free tier; set `GROQ_API_KEY` as an env var in the
      dashboard
- [ ] `README.md`: what it does, architecture diagram, setup steps, API examples
      with real request/response, **eval results table with sample size**,
      limitations section, live URL. States the accuracy is a point estimate on
      ~80 samples at temperature 0 (may vary a few points on re-run), documents
      the Render cold start (~30-50s), and describes the regex word tokenizer
      honestly (not the model's tokenizer)
- [ ] Push to GitHub as a public repo

**Checkpoint:** clone the repo into a fresh folder, follow only the README, and
get it running. Anything that required knowledge not in the README is a README
gap.

---

## Definition of done

- `/health` and `/classify` both live at a public URL
- `pytest` green with no API key present
- `EVAL_LOG.md` shows at least three prompt versions with measured deltas
- README states the accuracy figure, the sample size, and the limitations
- No secret anywhere in git history
- Every design decision in `ARCHITECTURE.md` is documented and defensible
