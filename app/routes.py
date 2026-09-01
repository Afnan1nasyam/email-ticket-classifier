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
