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
