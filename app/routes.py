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
    """Simple landing page describing the API."""
    return Response(
        """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Email Ticket Classifier API</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 760px;
            margin: 60px auto;
            padding: 0 24px;
            line-height: 1.6;
            color: #1f2937;
        }
        h1 { margin-bottom: 8px; }
        .subtitle { color: #6b7280; }
        code {
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .endpoint {
            border-left: 3px solid #d1d5db;
            padding-left: 16px;
            margin: 20px 0;
        }
        .method {
            font-weight: 700;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <h1>Email Ticket Classifier API</h1>
    <p class="subtitle">
        AI-powered customer email classification using
        <code>openai/gpt-oss-120b</code>.
    </p>

    <h2>Available endpoints</h2>

    <div class="endpoint">
        <span class="method">GET</span>
        <code>/health</code>
        <p>Check API health, model status, and application version.</p>
    </div>

    <div class="endpoint">
        <span class="method">POST</span>
        <code>/classify</code>
        <p>
            Classify an email into
            <code>billing</code>, <code>technical</code>,
            <code>complaint</code>, <code>urgent</code>,
            <code>feedback</code>, or <code>general</code>.
        </p>
    </div>

    <h2>Example request</h2>
    <pre><code>{
  "email_text": "I was charged twice for my invoice."
}</code></pre>
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
