# Eval Log

Durable, human-readable record of every evaluation run. Each `run_eval.py`
invocation appends a dated (UTC) entry below with: prompt version, reasoning
effort, sample size, accuracy, per-class metrics, top confusions, fallback rate,
mean latency, tokens, and an editable `notes:` line.

Accuracy is only comparable within a single dataset version (see
`data/README.md`). The per-run raw JSON lives beside this file and is gitignored;
this log is committed as the durable record.

> No runs recorded yet. Evaluation requires live Groq access, which the corporate
> network blocks on the build machine. Runs are executed on a personal laptop
> after `git pull` — a single command, e.g.
> `python evals/run_eval.py --prompt prompts/v1_zero_shot.txt`.

---
## 20260903T185710Z — v1_zero_shot.txt (reasoning_effort=low)

- status: complete
- prompt: `C:\Users\afnan\Downloads\email ticket classifier\email-ticket-classifier\prompts\v1_zero_shot.txt`
- dataset: `C:\Users\afnan\Downloads\email ticket classifier\email-ticket-classifier\data\test_dataset.csv` (40/81 rows)
- model: `openai/gpt-oss-120b`, temperature=0.0, reasoning_effort=low
- **accuracy: 95.0%** | macro-F1: 0.481 | fallback rate: 0.0%
- mean latency: 698 ms | tokens: prompt=24101, completion=3077, total=27178

Per-class metrics:

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| billing | 1.00 | 1.00 | 1.00 | 14 |
| technical | 0.93 | 0.93 | 0.93 | 14 |
| complaint | 1.00 | 0.92 | 0.96 | 12 |
| urgent | 0.00 | 0.00 | 0.00 | 0 |
| feedback | 0.00 | 0.00 | 0.00 | 0 |
| general | 0.00 | 0.00 | 0.00 | 0 |

Top confusions:
- technical misread as urgent: 1
- complaint misread as technical: 1

notes: 

---
## 20260904T111341Z — v1_zero_shot.txt (reasoning_effort=low)

- status: complete
- prompt: `C:\Users\afnan\Downloads\email ticket classifier\email-ticket-classifier\prompts\v1_zero_shot.txt`
- dataset: `C:\Users\afnan\Downloads\email ticket classifier\email-ticket-classifier\data\test_dataset.csv` (81/81 rows)
- model: `openai/gpt-oss-120b`, temperature=0.0, reasoning_effort=low
- **accuracy: 95.1%** | macro-F1: 0.952 | fallback rate: 0.0%
- mean latency: 919 ms | tokens: prompt=48722, completion=6375, total=55097

Per-class metrics:

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| billing | 0.93 | 0.93 | 0.93 | 15 |
| technical | 0.93 | 0.93 | 0.93 | 14 |
| complaint | 1.00 | 0.92 | 0.96 | 13 |
| urgent | 0.87 | 1.00 | 0.93 | 13 |
| feedback | 1.00 | 1.00 | 1.00 | 13 |
| general | 1.00 | 0.92 | 0.96 | 13 |

Top confusions:
- billing misread as urgent: 1
- technical misread as urgent: 1
- complaint misread as technical: 1

notes: 

---
## 20260904T114513Z — v2_few_shot.txt (reasoning_effort=low)

- status: complete
- prompt: `C:\Users\afnan\Downloads\email ticket classifier\email-ticket-classifier\prompts\v2_few_shot.txt`
- dataset: `C:\Users\afnan\Downloads\email ticket classifier\email-ticket-classifier\data\test_dataset.csv` (81/81 rows)
- model: `openai/gpt-oss-120b`, temperature=0.0, reasoning_effort=low
- **accuracy: 98.8%** | macro-F1: 0.988 | fallback rate: 0.0%
- mean latency: 777 ms | tokens: prompt=87440, completion=5311, total=92751

Per-class metrics:

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| billing | 1.00 | 0.93 | 0.97 | 15 |
| technical | 1.00 | 1.00 | 1.00 | 14 |
| complaint | 1.00 | 1.00 | 1.00 | 13 |
| urgent | 0.93 | 1.00 | 0.96 | 13 |
| feedback | 1.00 | 1.00 | 1.00 | 13 |
| general | 1.00 | 1.00 | 1.00 | 13 |

Top confusions:
- billing misread as urgent: 1

notes: 

---
## 20260904T125645Z — v3_precedence.txt (reasoning_effort=low)

- status: complete
- prompt: `C:\Users\afnan\Downloads\email ticket classifier\email-ticket-classifier\prompts\v3_precedence.txt`
- dataset: `C:\Users\afnan\Downloads\email ticket classifier\email-ticket-classifier\data\test_dataset.csv` (81/81 rows)
- model: `openai/gpt-oss-120b`, temperature=0.0, reasoning_effort=low
- **accuracy: 100.0%** | macro-F1: 1.000 | fallback rate: 0.0%
- mean latency: 888 ms | tokens: prompt=91247, completion=5763, total=97010

Per-class metrics:

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| billing | 1.00 | 1.00 | 1.00 | 15 |
| technical | 1.00 | 1.00 | 1.00 | 14 |
| complaint | 1.00 | 1.00 | 1.00 | 13 |
| urgent | 1.00 | 1.00 | 1.00 | 13 |
| feedback | 1.00 | 1.00 | 1.00 | 13 |
| general | 1.00 | 1.00 | 1.00 | 13 |

Top confusions:
- none

notes: 

---
