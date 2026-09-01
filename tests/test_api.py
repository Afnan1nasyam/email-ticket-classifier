"""Offline API-contract tests for the Flask app.

The app is built via `create_app(classifier=...)` with an injected
FakeProvider-backed classifier, so no real provider is constructed and no
network call is made. Every error must be JSON, never an HTML traceback.
"""

from __future__ import annotations

import json

import pytest

from app import config
from app.classifier import TicketClassifier
from app.preprocessor import EmailPreprocessor
from app.prompt_builder import PromptBuilder

from tests.conftest import RaisingProvider


def _is_json(response) -> bool:
    return response.content_type is not None and "application/json" in response.content_type


def _no_html_or_traceback(response) -> None:
    body = response.get_data(as_text=True)
    assert "<html" not in body.lower()
    assert "Traceback" not in body


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #

def test_health_returns_200_with_expected_keys(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert _is_json(resp)
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["model_id"] == config.MODEL_ID
    assert data["version"] == config.APP_VERSION


def test_health_makes_no_provider_call(app):
    provider = app.extensions["classifier"].provider
    client = app.test_client()
    client.get("/health")
    assert provider.calls == []


# --------------------------------------------------------------------------- #
# /classify — happy path
# --------------------------------------------------------------------------- #

def test_classify_valid_input_returns_200_and_result_json(client):
    resp = client.post("/classify", json={"email_text": "My invoice is wrong."})
    assert resp.status_code == 200
    assert _is_json(resp)
    data = resp.get_json()
    assert data["label"] in config.CATEGORIES
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["fallback_used"], bool)
    assert "prompt_tokens" in data
    assert "completion_tokens" in data
    assert "total_tokens" in data


def test_classify_happy_path_returns_injected_label(client):
    # The injected fixture provider returns a valid high-confidence billing payload.
    resp = client.post("/classify", json={"email_text": "You charged me twice."})
    assert resp.get_json()["label"] == "billing"


# --------------------------------------------------------------------------- #
# /classify — 400 input validation
# --------------------------------------------------------------------------- #

def test_classify_returns_400_when_email_text_missing(client):
    resp = client.post("/classify", json={"something_else": "x"})
    assert resp.status_code == 400
    assert _is_json(resp)
    _no_html_or_traceback(resp)


def test_classify_returns_400_when_body_not_object(client):
    resp = client.post("/classify", json=["not", "a", "dict"])
    assert resp.status_code == 400
    assert _is_json(resp)


@pytest.mark.parametrize("value", ["", "   ", "\n\t "])
def test_classify_returns_400_when_email_text_empty_or_whitespace(client, value):
    resp = client.post("/classify", json={"email_text": value})
    assert resp.status_code == 400
    assert _is_json(resp)
    _no_html_or_traceback(resp)


def test_classify_returns_400_when_email_text_not_string(client):
    resp = client.post("/classify", json={"email_text": 12345})
    assert resp.status_code == 400
    assert _is_json(resp)


def test_classify_returns_400_when_email_text_oversized(client):
    oversized = "a" * (config.MAX_INPUT_CHARS + 1)
    resp = client.post("/classify", json={"email_text": oversized})
    assert resp.status_code == 400
    assert _is_json(resp)
    _no_html_or_traceback(resp)


def test_classify_returns_400_for_malformed_json_body(client):
    resp = client.post(
        "/classify", data="{bad", content_type="application/json"
    )
    assert resp.status_code == 400
    assert _is_json(resp)
    _no_html_or_traceback(resp)


# --------------------------------------------------------------------------- #
# /classify — 502 provider failure
# --------------------------------------------------------------------------- #

def test_classify_returns_502_when_provider_raises(make_app):
    classifier = TicketClassifier(
        EmailPreprocessor(), RaisingProvider(), PromptBuilder()
    )
    app = make_app(classifier)
    client = app.test_client()
    resp = client.post("/classify", json={"email_text": "trigger failure"})
    assert resp.status_code == 502
    assert _is_json(resp)
    _no_html_or_traceback(resp)


def test_classify_502_body_contains_no_traceback(make_app):
    classifier = TicketClassifier(
        EmailPreprocessor(), RaisingProvider(), PromptBuilder()
    )
    app = make_app(classifier)
    resp = app.test_client().post("/classify", json={"email_text": "boom"})
    body = resp.get_data(as_text=True)
    assert "RuntimeError" not in body
    assert "Traceback" not in body


# --------------------------------------------------------------------------- #
# Errors are always JSON, never HTML
# --------------------------------------------------------------------------- #

def test_unknown_route_returns_json_404(client):
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    assert _is_json(resp)
    _no_html_or_traceback(resp)


def test_wrong_method_returns_json_405(client):
    resp = client.get("/classify")
    assert resp.status_code == 405
    assert _is_json(resp)
    _no_html_or_traceback(resp)
