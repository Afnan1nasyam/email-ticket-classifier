"""Flask application factory.

Builds the classification stack (preprocessor, provider, prompt builder,
classifier) once and stores the classifier on the app so it is reused across
requests rather than rebuilt per request. A classifier may be injected (used by
the tests) to avoid constructing a real provider.
"""

from __future__ import annotations

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from app import config
from app.classifier import TicketClassifier


def _json_error(status: int, message: str):
    """Build a JSON error response with the given status code."""
    response = jsonify({"error": message, "status": status})
    response.status_code = status
    return response


def _register_error_handlers(app: Flask) -> None:
    """Register handlers so every error is JSON, never an HTML traceback."""

    @app.errorhandler(400)
    def _bad_request(exc):  # noqa: ANN001
        return _json_error(400, getattr(exc, "description", "Bad Request"))

    @app.errorhandler(404)
    def _not_found(exc):  # noqa: ANN001
        return _json_error(404, "Not Found.")

    @app.errorhandler(405)
    def _method_not_allowed(exc):  # noqa: ANN001
        return _json_error(405, "Method Not Allowed.")

    @app.errorhandler(413)
    def _payload_too_large(exc):  # noqa: ANN001
        # Spec maps an oversized payload to 400 rather than 413.
        return _json_error(400, "Request payload is too large.")

    @app.errorhandler(Exception)
    def _unhandled(exc):  # noqa: ANN001
        # Preserve intentional HTTP errors; convert everything else to a clean
        # JSON 500 so a stack trace is never returned to the client.
        if isinstance(exc, HTTPException):
            return _json_error(exc.code or 500, exc.description or exc.name)
        app.logger.exception("unhandled error")
        return _json_error(500, "Internal Server Error.")


def create_app(classifier: TicketClassifier | None = None) -> Flask:
    """Create and configure the Flask app.

    Args:
        classifier: an optional pre-built classifier (injected by tests). When
            omitted, a `GroqProvider`-backed classifier is built once here.

    Returns:
        The configured Flask application.
    """
    app = Flask(__name__)
    # Hard ceiling on request body size as defense in depth; the route also
    # validates email_text length and returns 400.
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_INPUT_CHARS * 2 + 1024

    if classifier is None:
        # Imported lazily so importing the app package does not require the groq
        # SDK path unless a real provider is actually being constructed.
        from app.llm_provider import GroqProvider
        from app.preprocessor import EmailPreprocessor
        from app.prompt_builder import PromptBuilder

        classifier = TicketClassifier(
            preprocessor=EmailPreprocessor(),
            provider=GroqProvider(),
            prompt_builder=PromptBuilder(),
        )
    app.extensions["classifier"] = classifier

    from app.routes import api_bp

    app.register_blueprint(api_bp)
    _register_error_handlers(app)
    return app
