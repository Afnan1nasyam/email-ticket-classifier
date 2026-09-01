# Test Dataset (`test_dataset.csv`)

## What this is

`test_dataset.csv` is the held-out labeled dataset used to measure the
classifier's accuracy. It contains **81 rows**, one synthetic customer
email per row, each hand-labeled with one of the six categories.

Columns (exact order):

| column | meaning |
|---|---|
| `id` | sequential integer, 1..81 |
| `email_text` | the raw email body (may contain newlines, typos, quoted replies, signatures) |
| `true_label` | the ground-truth category: one of `billing`, `technical`, `complaint`, `urgent`, `feedback`, `general` |
| `label_rationale` | one concise sentence justifying `true_label` from the `config.py` definitions (states *why*, does not restate the email) |
| `difficulty` | `easy`, `medium`, or `hard` |

## The data is SYNTHETIC

Every email in this file was **written for this project**. None of it is real
customer correspondence. Names, companies, invoice numbers, ticket IDs, dollar
amounts, and addresses are all invented. There is no real PII and no secrets in
this file.

## How the labels were produced (and why that matters)

Labels were assigned **by hand, directly from the category definitions in
`app/config.py`** (`CATEGORIES`) plus the `URGENT_PRECEDENCE_RULE` — and from
nothing else. In particular:

- **No classification prompt and no LLM was used to assign these labels.** The
  dataset was built before any prompt was written. This structural independence
  is deliberate: if the same prompt that classifies emails had also produced the
  ground truth, the accuracy number would be circular and meaningless.
- Each label is intended to be **hand-reviewable**: a person reading the email
  alongside the `config.py` definitions should be able to agree with the label
  (or dispute it) without running any code.

### The urgent precedence rule (quoted verbatim from `app/config.py`)

> Label as urgent only when the email states time-critical business impact or an explicit deadline within roughly 24-48 hours. Frustrated or emphatic tone alone is not urgency. Otherwise apply the topical category.

`urgent` deliberately overlaps the topical categories (a billing email can also
be urgent). Every such overlap in this dataset was resolved with the rule above,
not with a judgment call about tone. Concretely:

- A billing/technical problem that states an explicit ~24-48h deadline or
  time-critical business impact is labeled **`urgent`**.
- A billing/technical problem that is merely angry, emphatic, or says "ASAP"
  with **no stated deadline or business impact** keeps its **topical** label.
- `general` is used only when no other category clearly applies. Per
  `config.py`, `general` is the last-resort fallback, **not** a tie resolver;
  genuine overlaps are resolved by topical precedence and the urgent rule first.

## Distribution

### By category (target within +/-1)

| category | count |
|---|---|
| billing | 15 |
| technical | 14 |
| complaint | 13 |
| urgent | 13 |
| feedback | 13 |
| general | 13 |
| **total** | **81** |

### By difficulty

| difficulty | count | intent |
|---|---|---|
| easy | 40 | one unmistakable signal; establishes a floor |
| medium | 28 | correct label is clear on a careful read, but a competing signal is present |
| hard | 13 | genuinely contested edge cases (see below) |
| **total** | **81** | |

### Category x difficulty

| category | easy | medium | hard |
|---|---|---|---|
| billing | 8 | 5 | 2 |
| technical | 6 | 5 | 3 |
| complaint | 7 | 4 | 2 |
| urgent | 6 | 5 | 2 |
| feedback | 7 | 4 | 2 |
| general | 6 | 5 | 2 |

## The 13 hard rows (what each is designed to test)

These are the rows that separate a well-engineered prompt from a lucky one. The
`id`s below have been checked against the CSV.

- **Billing vs. urgent, precedence tested both ways:**
  - `id 53` — invoice error (charged for 500 seats vs 50) **with** an explicit
    9am-tomorrow auto-pay deadline and a $22,000 erroneous-debit impact →
    **`urgent`** (topic is billing, but the precedence rule promotes it because of
    the stated sub-48h deadline + business impact).
  - `id 14` — repeated overcharge, angry and emphatic, but **no** deadline or
    business impact → **`billing`** (per the precedence rule, emphatic tone alone
    is not urgency).
- **Emphatic-tone-only foil (inverse of `id 54`):** `id 81` — an ALL-CAPS
  "URGENT!!! ... ASAP ... NOW" demand to resend an invoice and explain a charge,
  with **no** stated deadline and **no** business impact → **`billing`**. This is
  the deliberate mirror of `id 54`: it lets the eval distinguish *correctly
  ignoring urgency tone* from *correctly detecting real urgency*.
- **Polite phrasing over a business-critical outage:** `id 54` — a very courteous
  email describing a hospital records system down since morning, blocking
  admissions → **`urgent`** (the precedence rule keys on substance/impact, not
  tone).
- **Complaint vs. feedback:**
  - `id 40` — "I'm furious ... you should show clearer errors, but I just want
    someone to acknowledge how much time this cost me": contains a suggestion but
    the dominant intent is a grievance seeking redress → **`complaint`**.
  - `id 66` — harsh criticism of the reporting module whose actual core is a
    concrete product-improvement suggestion, with no defect and no demand for
    redress → **`feedback`**.
- **Technical vs. complaint:**
  - `id 27` — dissatisfied, buggy-product email ("auto-save fails ... I just need
    it to work") whose core is a defect needing a fix → **`technical`**.
  - `id 28` — an angry ALL-CAPS bug report about a broken PDF export; anger is
    tone, the substance is a product defect → **`technical`**.
  - `id 41` — explicitly "not a bug"; the grievance is about staff repeatedly
    rescheduling a migration and disrespecting the customer's time →
    **`complaint`**.
- **Very short / minimal-signal:** `id 26` ("app keeps crashing everytime i open
  the reports tab. pls fix asap" → `technical`; note "asap" is emphatic tone, not
  a stated deadline) and `id 67` ("love the new update, keep it coming!" →
  `feedback`).
- **No clear category → `general`:** `id 79` (a vague office-relocation heads-up)
  and `id 80` (a request to confirm mailed documents arrived) — neither fits any
  topical category.

## Realism

The emails intentionally include typos, all-lowercase and ALL-CAPS messages,
non-native English phrasing, quoted reply chains (`> On ...`), signature
fragments, embedded stack traces, mixed-in irrelevant detail, and lengths
ranging from a few words to several sentences. This is so the dataset can
actually fail a brittle prompt rather than rewarding keyword matching.

## Known limitations

- **Synthetic.** Real support mail has distributions, edge cases, and noise this
  set only approximates. Accuracy here is **indicative, not a guarantee** of
  real-world performance.
- **Small sample (81 rows).** The number is indicative, not statistically
  tight. A single misclassification moves overall accuracy by ~1.2 points, and
  per-category slices (13-15 rows each) have wide confidence intervals. Treat
  category-level numbers as directional only.
- **Single-label, single-turn.** Each email gets exactly one label; real tickets
  can be multi-topic, and real threads span multiple turns. This dataset models
  neither.
- **Author-labeled.** Ground truth reflects one author's reading of the
  `config.py` definitions. It has not been adjudicated by multiple independent
  annotators, so inter-annotator agreement is unmeasured.

## Reproducing / regenerating

The CSV was written with Python's `csv` module (`csv.writer`, default
`QUOTE_MINIMAL`) so all quoting and escaping of embedded commas, quotes, and
newlines is correct. Read it back with `pandas.read_csv`, which round-trips the
quoting.

## Change log (dataset versioning)

Accuracy figures are only comparable **within** a single dataset version. If
rows are later appended (for example, to target a confusion-matrix weakness
exposed by an eval run), record here which `id` ranges were added and when, and
note that accuracy across versions is not directly comparable.

- **2026-08-31 — v1:** initial 80 rows (`id 1..80`), four columns
  (`id, email_text, true_label, difficulty`). Category balance billing 14,
  technical 14, complaint/urgent/feedback/general 13 each; difficulty 40/28/12.
- **2026-08-31 — v2:** added the `label_rationale` column (now five columns);
  appended `id 81`, a hard emphatic-tone-only billing foil (the inverse of the
  polite-outage `id 54`), so the eval can distinguish ignoring urgency *tone*
  from detecting real urgency. New totals: 81 rows; billing 15; difficulty
  40/28/13. Rows `1..80` are otherwise unchanged in text and label.
  Accuracy on v2 is **not** directly comparable to any v1 run.
