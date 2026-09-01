# Email / Ticket Classifier

An AI-powered email/ticket classification system: a Flask REST API that accepts
raw customer email text and returns a category label, using a hosted LLM (Groq)
for classification.

> **Status:** in development. This README currently documents setup only.
> Architecture, API usage, evaluation results, limitations, and the live URL are
> **TBD** and will be completed in the final phase.

## Setup

Requires **Python 3.14**. Commands below use forward-slash paths and work on
Windows, macOS, and Linux; where a step differs by OS, both forms are shown.
Use `python3` instead of `python` if `python` is not Python 3 on your system.

1. **Clone and enter the project:**
   ```
   git clone <repo-url>
   cd email-ticket-classifier
   ```

2. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   ```
   Activate it:
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
   - macOS / Linux: `source venv/bin/activate`

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Configure your API key:**
   - Copy the template to `.env`
     (macOS/Linux: `cp .env.example .env` — Windows: `copy .env.example .env`).
   - Open `.env` and set `GROQ_API_KEY` to a real key from
     https://console.groq.com/keys. `.env` is gitignored; never commit it.

5. **Verify connectivity:**
   ```
   python scripts/verify_setup.py
   ```
   A successful run prints the model response, token counts, and latency, and
   exits 0. See `scripts/README.md` for what non-zero exit codes mean.
