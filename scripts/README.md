# scripts/

Permanent operational and diagnostic scripts — committed artifacts, not
throwaway checks.

## verify_setup.py

Verifies that the current environment can reach Groq and get a completion. This
is the Phase 1 connectivity checkpoint. Run it after following the Setup steps
in the top-level `README.md`:

```
python scripts/verify_setup.py
```

It builds a `GroqProvider` (the same abstraction the application uses, so it
also validates the provider layer, its OS-trust-store TLS handling, and retry
config) and makes one real call with reasoning suppressed, then prints the
response, token counts, and latency.

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success — a completion was returned |
| 1 | Unexpected error, or the `groq` SDK is not installed |
| 2 | `GROQ_API_KEY` is not set (copy `.env.example` to `.env` and add a key) |
| 3 | Network / TLS failure — could not reach Groq |
| 4 | Request rejected — invalid key, or the URL is blocked by a web filter |

Note: this script goes through `GroqProvider` in `app/llm_provider.py` — the
only module that imports the `groq` SDK. It does not import the SDK itself.
