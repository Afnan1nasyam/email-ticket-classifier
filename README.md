# Email / Ticket Classifier

An AI-powered email/ticket classification system: a Flask REST API that accepts
raw customer email text and returns a support category, using a hosted LLM
(Groq, `openai/gpt-oss-120b`) for classification. Stateless, no database, no UI.

**Live URL:** _TBD — deployed to Render after the evaluation runs._

Categories: `billing`, `technical`, `complaint`, `urgent`, `feedback`,
`general` (the last also serves as the low-confidence fallback).

---

## Architecture

```
POST /classify
      |
      v
routes.py            HTTP layer: parse, validate, serialize, error mapping
      |
      v
TicketClassifier     Service layer: preprocess -> prompt -> parse -> validate
      |
      +--> EmailPreprocessor    pandas cleaning, normalization, tokenization
      +--> PromptBuilder        loads a versioned prompt template, injects email
      +--> LLMProvider / GroqProvider
                |               Groq SDK, retries, backoff, rate-limit handling
                v
           openai/gpt-oss-120b
      |
      v
ClassificationResult  validated: label, confidence, reasoning, latency, tokens
```

The eval harness reuses the same `TicketClassifier` the API uses, so the
accuracy number reflects the production path. Nothing outside
`app/llm_provider.py` imports the Groq SDK — swapping providers is a config
change. See `ARCHITECTURE.md` for the full design rationale.

---

## Setup

Requires **Python 3.14**. Paths below use forward slashes and work on Windows,
macOS, and Linux. Use `python3` if `python` is not Python 3 on your system.

1. **Clone and enter the project:**
   ```
   git clone <repo-url>
   cd email-ticket-classifier
   ```

2. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   ```
   Activate it — pick the line for your shell:
   - Windows PowerShell: `venv\Scripts\Activate.ps1`
   - Windows cmd.exe: `venv\Scripts\activate.bat`
   - macOS / Linux (bash/zsh): `source venv/bin/activate`

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Configure your API key:**
   - Copy the template to `.env`
     (macOS/Linux: `cp .env.example .env` — Windows: `copy .env.example .env`).
   - Set `GROQ_API_KEY` in `.env` to a real key from
     https://console.groq.com/keys. `.env` is gitignored; never commit it.

5. **Verify connectivity:**
   ```
   python scripts/verify_setup.py
   ```
   Prints the model response, token counts, and latency; exits 0 on success.

---

## Running the API

```
python run.py
```

Starts the Flask development server on `http://127.0.0.1:5000`. For a
production-style server, use the Docker image (below), which runs gunicorn.

---

## API

### `GET /health`

Liveness check. Makes no LLM call.

```
$ curl http://127.0.0.1:5000/health
{"model_id":"openai/gpt-oss-120b","status":"ok","version":"0.1.0"}
```

### `POST /classify`

Request:

```
$ curl -X POST http://127.0.0.1:5000/classify \
    -H "Content-Type: application/json" \
    -d '{"email_text": "I was double charged for my June invoice, please refund the extra $49."}'
```

Response (`200`):

```json
{
  "label": "billing",
  "confidence": 0.95,
  "reasoning": "duplicate charge, customer requests a refund",
  "fallback_used": false,
  "latency_ms": 640,
  "prompt_tokens": 512,
  "completion_tokens": 28,
  "total_tokens": 540,
  "model_id": "openai/gpt-oss-120b"
}
```

> The exact `latency_ms`, token counts, and `reasoning` come from the live
> model; the shape above is representative. All endpoints were verified with the
> Flask test client (see the tests). `label` is always one of the six
> categories; if the model is unparseable, returns an unknown label, or reports
> confidence below the threshold (0.6), the result degrades to `general` with
> `"fallback_used": true`.

### Error responses

All errors are JSON (never an HTML traceback):

| Situation | Status | Body |
|---|---|---|
| Missing `email_text` | 400 | `{"error":"Missing required field: 'email_text'.","status":400}` |
| Empty / whitespace `email_text` | 400 | `{"error":"'email_text' must be a non-empty string.","status":400}` |
| Oversized `email_text` | 400 | `{"error":"'email_text' exceeds the maximum length of 10000 characters.","status":400}` |
| Malformed JSON body | 400 | `{"error":"Request body must be a JSON object.","status":400}` |
| Provider failure | 502 | `{"error":"Classification provider error.","status":502}` |

---

## Evaluation

Accuracy is measured against a held-out labeled dataset
(`data/test_dataset.csv`, 81 synthetic emails; see `data/README.md` for
provenance and caveats), using the same classifier the API uses. Runs are
deliberately throttled to respect Groq's free-tier token limit and take
10–15 minutes for the full set.

```
python evals/run_eval.py --prompt prompts/v1_zero_shot.txt
python evals/run_eval.py --prompt prompts/v3_few_shot_with_rules.txt --reasoning-effort medium
```

Each run writes a JSON record to `evals/results/` and appends a dated entry to
`evals/results/EVAL_LOG.md` (the durable, human-readable history of every prompt
version and its measured accuracy).

### Results

Measured at `temperature=0` on the ~81-example set; a point estimate that may
vary a few points on re-run. **Numbers are filled in after the eval runs.**

| Prompt version | Sample size | Accuracy | Macro F1 | Notes |
|---|---|---|---|---|
| v1 (zero-shot baseline) | TBD | TBD | TBD | category definitions only |
| v2 (few-shot) | TBD | TBD | TBD | + examples targeting v1's confusions |
| v3 (few-shot + rules) | TBD | TBD | TBD | + explicit urgent precedence rule |

---

## Testing

```
python -m pytest
```

The suite runs fully offline against a fake provider — **no API key and no
network required** — in about a second. It covers preprocessing, the classifier
(including malformed-output fallback), the API endpoints and error paths, the
metric functions, and a check that the API key never appears in logs or
responses.

---

## Docker

```
docker build -t email-ticket-classifier .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key email-ticket-classifier
```

The image runs gunicorn as a non-root user on port 8000 and serves the app
factory. `.env` is never copied into the image; the key is supplied at runtime
via an environment variable (on Render, set it in the dashboard).

---

## Limitations

- **Synthetic dataset.** No public dataset maps onto these six categories; the
  set was written and hand-labeled for this project, and deliberately includes
  ambiguous and adversarial cases. It approximates, but is not, real support mail.
- **Small sample (~81 rows).** Accuracy is indicative, not statistically tight;
  per-category slices have wide confidence intervals.
- **Single-label, single-turn.** One label per email, no conversation context,
  no multi-label output. Priority would be a separate dimension in production.
- **Model-self-reported confidence.** The confidence value is the model's own
  and is poorly calibrated; the eval reports fallback rate as a separate metric.
- **No cost dashboard.** Token counts are captured and logged per call, but there
  is no aggregate spend tracking.
- **Render free tier cold start.** After idle, the first request can take
  roughly 30–50 seconds to wake the service.
