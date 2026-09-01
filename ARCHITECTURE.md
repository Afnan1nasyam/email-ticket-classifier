# Architecture

## Overview

A stateless Flask service that classifies customer emails into one of six
support categories using a hosted LLM, with a separate eval harness that
measures classification accuracy against a labeled dataset.

Request path:

```
POST /classify
      |
      v
routes.py            HTTP layer: parse, validate, serialize, error mapping
      |
      v
TicketClassifier     Service layer: orchestrates preprocess -> prompt -> parse
      |
      +--> EmailPreprocessor    pandas cleaning, normalization, tokenization
      |
      +--> PromptBuilder        loads versioned prompt template, injects email
      |
      +--> LLMProvider          abstract interface
                |
                v
           GroqProvider         Groq SDK, retries, backoff, rate-limit handling
                |
                v
           openai/gpt-oss-120b
      |
      v
ClassificationResult  validated: label, confidence, reasoning, latency, tokens
```

The eval harness reuses the same `TicketClassifier` the API uses. It does not
have its own classification path. This matters: if the eval measured a different
code path than production, the accuracy number would be meaningless.

---

## Layers

**HTTP layer** — `app/routes.py`
Flask blueprint. Knows about requests and responses, nothing about LLMs. Maps
domain exceptions to status codes. Validates input shape and size before any
expensive work happens.

**Service layer** — `app/classifier.py`
`TicketClassifier` owns the classification workflow. Injected with a preprocessor
and a provider, so both can be swapped or mocked. Handles the low-confidence
fallback rule and guarantees the returned label is always one of the configured
categories — never a hallucinated one.

**Provider layer** — `app/llm_provider.py`
`LLMProvider` is an abstract base class defining `complete(system, user) -> str`.
`GroqProvider` implements it. Retry logic, backoff, and rate-limit awareness live
here, not in the service layer.

This is the layer that exists because of a real event: the model this project was
originally specced against (`llama-3.3-70b-versatile`) was deprecated by Groq on
2026-08-16, mid-project. The abstraction means that swap was a one-line config
change. Any future model change is the same.

**Data layer** — `app/preprocessor.py`
`EmailPreprocessor` uses pandas for batch operations and exposes single-item
methods for the API path. Strips signatures, quoted reply chains, HTML remnants,
and excess whitespace; normalizes case for tokenization; truncates to a token
budget. Tokenization is used for length control and for the dataset statistics in
the README — it is not a feature-extraction step, since the LLM does the
classification.

Tokenization is a **regex word-boundary tokenizer**, not a model tokenizer, and
adds no external dependency. `truncate()` approximates the model's token budget
at roughly 4 characters per token with a safety margin, since the true tokenizer
for `openai/gpt-oss-120b` is not exposed locally. The README must describe this
honestly and must **not** imply the word tokenizer is the model's own tokenizer.

---

## Folder structure

```
email-ticket-classifier/
├── .claude/
│   └── agents/
│       ├── dataset-builder.md      subagent: generate labeled test data
│       ├── prompt-optimizer.md     subagent: iterate prompts against evals
│       └── test-writer.md          subagent: write pytest suites
├── app/
│   ├── __init__.py                 app factory
│   ├── config.py                   settings, category definitions, model ID
│   ├── schemas.py                  request/response dataclasses, validation
│   ├── preprocessor.py             EmailPreprocessor
│   ├── prompt_builder.py           PromptBuilder, loads from prompts/
│   ├── llm_provider.py             LLMProvider ABC + GroqProvider
│   ├── classifier.py               TicketClassifier
│   └── routes.py                   Flask blueprint: /classify, /health
├── prompts/
│   ├── v1_zero_shot.txt
│   ├── v2_few_shot.txt
│   ├── v3_few_shot_with_rules.txt
│   └── ACTIVE.txt                  copy of the winning version
├── data/
│   ├── test_dataset.csv            email_text, true_label, difficulty
│   └── README.md                   provenance and known limitations
├── evals/
│   ├── run_eval.py                 CLI eval runner
│   ├── metrics.py                  accuracy, per-class P/R/F1, confusion matrix
│   └── results/
│       ├── EVAL_LOG.md             human-readable history of every run
│       └── *.json                  raw per-run output
├── tests/
│   ├── conftest.py                 fixtures, fake provider
│   ├── test_preprocessor.py
│   ├── test_classifier.py
│   └── test_api.py
├── run.py                          dev entrypoint
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── CLAUDE.md
├── ARCHITECTURE.md
├── PLAN.md
└── README.md
```

---

## Categories

Six classes, defined once in `app/config.py`:

| Label | Meaning |
|---|---|
| `billing` | invoices, charges, refunds, payment failures, pricing |
| `technical` | bugs, errors, outages, integration and setup problems |
| `complaint` | dissatisfaction with service, staff, or handling |
| `urgent` | explicit time pressure or business-critical impact |
| `feedback` | suggestions, feature requests, praise |
| `general` | everything else, and the low-confidence fallback |

`urgent` deliberately overlaps the others, because that reflects real support
inboxes. A billing email can also be urgent. The prompt must define a precedence
rule and the dataset must contain cases that test it. Documenting how this
ambiguity is resolved is worth more than a clean-looking accuracy number.

**Precedence rule (canonical).** This exact wording appears verbatim — no
paraphrasing — in `app/config.py`, the active prompt template, and
`data/README.md`:

> "Label as urgent only when the email states time-critical business impact or
> an explicit deadline within roughly 24-48 hours. Frustrated or emphatic tone
> alone is not urgency. Otherwise apply the topical category."

Single-label output is what forces a precedence decision at all: a message can be
both billing and urgent, but the schema emits one label. In production, priority
would be a separate dimension from topic rather than a competing category.

---

## Design decisions

**1. Provider abstraction**
Nothing outside `llm_provider.py` imports the Groq SDK. Driven by a real
deprecation event. Cost: one extra file. Benefit: model swaps are config changes.

**2. Prompts as versioned files, not inline strings**
Each prompt version is a file in `prompts/`. The eval runner takes a prompt path
as an argument. This is what makes "v1 scored 71%, v3 scored 93%" a reproducible
claim rather than a story.

**3. Structured JSON output with a validation layer**
The model is instructed to return JSON with `label`, `confidence`, and
`reasoning`. The parser handles markdown fences, leading prose, and malformed
output. If the returned label is not in the configured set, it is rejected and
falls back to `general` — the API never emits a category that does not exist.

**4. Confidence threshold with fallback**
Below the configured threshold, the result becomes `general` with a flag
indicating fallback occurred. The threshold starts at **0.6** and is treated as a
value to tune against eval data, not a fixed constant. The confidence is
**model-self-reported** and therefore poorly calibrated — which is precisely why
the eval reports fallback rate as a separate metric from accuracy. A system that
answers `general` to everything can look deceptively stable.

**5. Eval harness separate from unit tests**
`pytest` runs offline against a fake provider, in under a second, with no API
key. `evals/run_eval.py` hits the real API, costs time and rate-limit budget, and
is run deliberately. Conflating them means CI either burns quota or tests
nothing.

**6. Confusion matrix, not just accuracy**
Aggregate accuracy hides the interesting failures. Per-class precision and recall
tell you *which* pairs the model confuses, which is what actually drives the next
prompt revision.

**7. Adaptive token-window throttling, not a fixed sleep**
The binding Groq limit is 6,000 tokens/minute, and completion length varies per
call, so a fixed inter-call delay is either too slow or trips HTTP 429.
`GroqProvider` records the actual `prompt_tokens` and `completion_tokens` the API
returns on every call. The eval runner maintains a rolling 60-second window of
those observed token counts and sleeps only until the next call fits under the
6,000 TPM ceiling. This adapts to real usage instead of guessing token counts up
front. The window has no history on the first call of a run, so it is seeded with
a conservative estimate (~1,200 tokens) to keep an early burst from tripping 429
before real data accumulates. The provider still retries 429/5xx with exponential
backoff as a backstop.

**8. Deterministic inference; reasoning effort as a measured variable**
`temperature` is set to **0** in `app/config.py`. Temperature 0 makes the eval as
reproducible as a hosted LLM allows; the README states the accuracy is a point
estimate on ~80 samples and may vary by a few points on re-run.

`reasoning_effort` defaults to `low` but is a **measured tradeoff, not an
assumption**. Low effort protects the 6,000 TPM budget and keeps iteration fast,
but plausibly costs accuracy on exactly the hard/ambiguous cases that decide
whether we clear 90%. It is therefore a `run_eval.py` flag (`--reasoning-effort`),
and Phase 8 calls for measuring `medium` against the same prompt if accuracy
stalls — isolating a model-parameter effect from a prompt effect is a finding
worth recording either way.

`reasoning_format` is **not supported** on gpt-oss-120b and is never set. The
model returns reasoning in a separate `reasoning` field by default; `GroqProvider`
sets `include_reasoning=false` to suppress it and reads the classification JSON
only from `message.content`. Reasoning tokens still count toward
`completion_tokens` (and thus TPM), which the adaptive window handles correctly
because it uses the counts the API actually returns.

**9. Prompt-instructed JSON over structured outputs**
`response_format` with a `json_schema` was considered and rejected. There are
Groq community reports of `json_schema` structured outputs being ignored by
gpt-oss-120b and the model returning free-form text anyway. Relying on it would
add a failure mode we cannot see in tests. The deliberate choice is to instruct
the JSON contract in the prompt and parse defensively — handling markdown fences,
leading prose, and malformed output — with the test suite covering those failure
modes explicitly. This is an evidence-based tradeoff, not an oversight.

---

## Known limitations

State these in the README. Volunteering them is stronger than being caught by
them.

- **The test dataset is synthetic.** No public dataset maps cleanly onto these six
  categories. Labels were hand-reviewed, and the set deliberately includes
  ambiguous and adversarial cases, but it is generated data and the README says
  so plainly.
- **Accuracy is measured on ~80 examples.** That is enough to be indicative, not
  enough for tight confidence intervals. Report the sample size next to the
  number.
- **Single-turn, single-label.** No conversation context, no multi-label output.
  Real support systems often need both, and would model priority as a separate
  dimension from topic rather than as the competing `urgent` category.
- **No cost tracking in the API response.** Token counts are captured per call
  and logged, but there is no aggregate spend dashboard.
- **Deployed on a free tier (Render).** Cold starts take roughly 30-50 seconds
  after idle; the README documents this. The GitHub repo and `EVAL_LOG.md` are
  the durable artifacts — the live URL is a bonus and may sleep or lapse.
