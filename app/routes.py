"""HTTP layer: the `/health` and `/classify` endpoints.

Knows about requests and responses, not about LLMs. Validates input shape and
size before any expensive work, and maps failures to JSON error bodies (never
HTML tracebacks). The classifier is built once at app-factory time and looked up
here per request from `app.extensions`.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request
from flask.wrappers import Response

from app import config

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


def _error(status: int, message: str) -> tuple[Response, int]:
    """Build a JSON error response."""
    return jsonify({"error": message, "status": status}), status


@api_bp.get("/")
def index() -> Response:
    """Interactive landing page and API demo."""
    return Response(
        """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Email Ticket Classifier API</title>
    <style>
        :root {
            color-scheme: light;
            font-family: system-ui, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            background: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 48px 24px 64px;
        }

        .hero {
            margin-bottom: 42px;
        }

        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: #e2e8f0;
            color: #475569;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        h1 {
            margin: 0 0 10px;
            font-size: clamp(32px, 6vw, 48px);
            line-height: 1.15;
            letter-spacing: -1px;
        }

        .hero p {
            max-width: 720px;
            margin: 0;
            color: #64748b;
            font-size: 18px;
        }

        section {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 28px;
            margin-bottom: 22px;
        }

        h2 {
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 22px;
        }

        h3 {
            margin-bottom: 6px;
        }

        .muted {
            color: #64748b;
        }

        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 18px;
            margin-top: 20px;
        }

        .feature {
            padding: 16px;
            border-radius: 10px;
            background: #f8fafc;
        }

        .feature h3 {
            font-size: 16px;
            margin-top: 0;
        }

        .feature p {
            margin-bottom: 0;
            color: #64748b;
            font-size: 14px;
        }

        textarea {
            width: 100%;
            min-height: 150px;
            resize: vertical;
            padding: 14px;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            font: inherit;
            color: #1e293b;
            background: #ffffff;
        }

        textarea:focus {
            outline: 2px solid #94a3b8;
            outline-offset: 1px;
        }

        button {
            margin-top: 12px;
            padding: 11px 18px;
            border: 0;
            border-radius: 9px;
            background: #1e293b;
            color: #ffffff;
            font: inherit;
            font-weight: 700;
            cursor: pointer;
        }

        button:hover {
            background: #334155;
        }

        button:disabled {
            opacity: 0.6;
            cursor: wait;
        }

        .result {
            display: none;
            margin-top: 22px;
            padding: 20px;
            border-radius: 10px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }

        .result.visible {
            display: block;
        }

        .result-label {
            font-size: 28px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
        }

        .result-item {
            padding: 12px;
            background: #ffffff;
            border-radius: 8px;
        }

        .result-item strong {
            display: block;
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .result-item span {
            display: block;
            margin-top: 3px;
            font-weight: 600;
        }

        .error {
            color: #b91c1c;
            font-weight: 600;
        }

        code {
            padding: 2px 6px;
            border-radius: 5px;
            background: #f1f5f9;
            font-size: 0.92em;
        }

        .categories {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }

        .category {
            padding: 5px 10px;
            border-radius: 999px;
            background: #f1f5f9;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 13px;
        }

        .stack {
            color: #475569;
        }

        .api-row {
            padding: 13px 0;
            border-bottom: 1px solid #e2e8f0;
        }

        .api-row:last-child {
            border-bottom: 0;
            padding-bottom: 0;
        }

        .method {
            display: inline-block;
            min-width: 52px;
            font-weight: 800;
            font-size: 13px;
        }

        footer {
            text-align: center;
            color: #94a3b8;
            font-size: 13px;
            margin-top: 28px;
        }

        @media (max-width: 600px) {
            .container {
                padding: 30px 16px 48px;
            }

            section {
                padding: 20px;
            }
        }
    </style>
</head>

<body>
    <main class="container">
        <header class="hero">
            <div class="badge">AI / REST API / Production Demo</div>
            <h1>Email Ticket Classifier API</h1>
            <p>
                An AI-powered REST API that automatically classifies customer
                support emails into predefined categories using a hosted LLM.
            </p>
        </header>

        <section>
            <h2>What this project does</h2>
            <p class="muted">
                The system accepts raw customer email text, validates and
                preprocesses the input, builds a structured classification
                prompt, sends it to the LLM, and returns a normalized result
                with the predicted category, confidence, reasoning, model
                information, latency, and fallback status.
            </p>

            <div class="features">
                <div class="feature">
                    <h3>Automated classification</h3>
                    <p>
                        Routes incoming support emails into six predefined
                        business categories.
                    </p>
                </div>

                <div class="feature">
                    <h3>Prompt engineering</h3>
                    <p>
                        Uses few-shot examples and an explicit urgency
                        precedence rule to reduce common category confusions.
                    </p>
                </div>

                <div class="feature">
                    <h3>Input protection</h3>
                    <p>
                        Validates request structure and input size before
                        making an expensive LLM call.
                    </p>
                </div>

                <div class="feature">
                    <h3>Production API</h3>
                    <p>
                        Dockerized Flask service with structured JSON errors
                        and provider-failure handling.
                    </p>
                </div>
            </div>
        </section>

        <section>
            <h2>Try the classifier</h2>
            <p class="muted">
                Enter a customer support email below. This demo calls the same
                production <code>/classify</code> endpoint used by the API.
            </p>

            <textarea id="emailText" placeholder="Example: I was charged twice for my invoice and need a refund."></textarea>

            <button id="classifyButton" onclick="classifyEmail()">
                Classify Email
            </button>

            <div id="result" class="result">
                <div class="result-label" id="resultLabel"></div>

                <div class="result-grid">
                    <div class="result-item">
                        <strong>Confidence</strong>
                        <span id="resultConfidence"></span>
                    </div>

                    <div class="result-item">
                        <strong>Model</strong>
                        <span id="resultModel"></span>
                    </div>

                    <div class="result-item">
                        <strong>Latency</strong>
                        <span id="resultLatency"></span>
                    </div>

                    <div class="result-item">
                        <strong>Fallback</strong>
                        <span id="resultFallback"></span>
                    </div>
                </div>

                <p>
                    <strong>Reasoning</strong><br>
                    <span id="resultReasoning"></span>
                </p>
            </div>

            <p id="errorMessage" class="error"></p>
        </section>

        <section>
            <h2>Supported categories</h2>
            <p class="muted">
                Each email is assigned one primary support category.
            </p>

            <div class="categories">
                <span class="category">billing</span>
                <span class="category">technical</span>
                <span class="category">complaint</span>
                <span class="category">urgent</span>
                <span class="category">feedback</span>
                <span class="category">general</span>
            </div>
        </section>

        <section>
            <h2>Evaluation</h2>
            <p class="muted">
                The final prompt achieved <strong>100.0% accuracy</strong> and
                <strong>1.000 macro-F1</strong> on an 81-example synthetic
                evaluation set. The result is indicative of this test set and
                is not a claim of real-world accuracy.
            </p>
        </section>

        <section>
            <h2>Technology</h2>
            <p class="stack">
                <code>Python</code>
                <code>Flask</code>
                <code>Docker</code>
                <code>Groq</code>
                <code>openai/gpt-oss-120b</code>
            </p>
        </section>

        <section>
            <h2>API</h2>

            <div class="api-row">
                <span class="method">GET</span>
                <code>/health</code>
                <span class="muted">
                    — service health, model, and application version
                </span>
            </div>

            <div class="api-row">
                <span class="method">POST</span>
                <code>/classify</code>
                <span class="muted">
                    — classify a customer email and return structured results
                </span>
            </div>
        </section>

        <footer>
            Email Ticket Classifier · Flask REST API · Hosted on Render
        </footer>
    </main>

    <script>
        async function classifyEmail() {
            const emailText = document.getElementById("emailText").value.trim();
            const button = document.getElementById("classifyButton");
            const result = document.getElementById("result");
            const errorMessage = document.getElementById("errorMessage");

            result.classList.remove("visible");
            errorMessage.textContent = "";

            if (!emailText) {
                errorMessage.textContent = "Please enter an email to classify.";
                return;
            }

            button.disabled = true;
            button.textContent = "Classifying...";

            try {
                const response = await fetch("/classify", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        email_text: emailText
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(
                        data.error || "Classification request failed."
                    );
                }

                document.getElementById("resultLabel").textContent =
                    data.label || "unknown";

                document.getElementById("resultConfidence").textContent =
                    data.confidence !== undefined
                        ? Number(data.confidence).toFixed(2)
                        : "N/A";

                document.getElementById("resultModel").textContent =
                    data.model_id || "N/A";

                document.getElementById("resultLatency").textContent =
                    data.latency_ms !== undefined
                        ? Math.round(data.latency_ms) + " ms"
                        : "N/A";

                document.getElementById("resultFallback").textContent =
                    data.fallback_used ? "Yes" : "No";

                document.getElementById("resultReasoning").textContent =
                    data.reasoning || "No reasoning returned.";

                result.classList.add("visible");
            } catch (error) {
                errorMessage.textContent =
                    error.message || "Unable to classify the email.";
            } finally {
                button.disabled = false;
                button.textContent = "Classify Email";
            }
        }
    </script>
</body>
</html>""",
        mimetype="text/html",
    )

@api_bp.get("/health")
def health() -> Response:
    """Liveness/readiness check. Does not call the LLM."""
    return jsonify(
        {
            "status": "ok",
            "model_id": config.MODEL_ID,
            "version": config.APP_VERSION,
        }
    )


@api_bp.post("/classify")
def classify():
    """Classify one email supplied as ``{"email_text": "..."}``."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error(400, "Request body must be a JSON object.")

    if "email_text" not in body:
        return _error(400, "Missing required field: 'email_text'.")

    email_text = body["email_text"]
    if not isinstance(email_text, str) or not email_text.strip():
        return _error(400, "'email_text' must be a non-empty string.")

    if len(email_text) > config.MAX_INPUT_CHARS:
        return _error(
            400,
            f"'email_text' exceeds the maximum length of {config.MAX_INPUT_CHARS} characters.",
        )

    classifier = current_app.extensions["classifier"]
    try:
        result = classifier.classify(email_text)
    except Exception:
        # Any downstream failure (provider outage, rate-limit exhaustion, etc.).
        # Log server-side (the API key is never logged) and return a clean 502.
        logger.exception("classification failed")
        return _error(502, "Classification provider error.")

    return jsonify(result.to_dict())
