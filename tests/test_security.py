"""Security tests: the Groq API key prefix `gsk_` must never leak.

Covers structured logging during classification, the /health and happy-path
/classify response bodies, and the 502 provider-failure path.
"""

from __future__ import annotations

import logging

from app.classifier import TicketClassifier
from app.preprocessor import EmailPreprocessor
from app.prompt_builder import PromptBuilder

from tests.conftest import FakeProvider, RaisingProvider

KEY_PREFIX = "gsk_"


def test_no_key_prefix_in_logs_during_classify(classifier, caplog):
    with caplog.at_level(logging.DEBUG):
        classifier.classify("My invoice is wrong.")
    for record in caplog.records:
        assert KEY_PREFIX not in record.getMessage()


def test_health_response_body_has_no_key_prefix(client):
    resp = client.get("/health")
    assert KEY_PREFIX not in resp.get_data(as_text=True)


def test_classify_response_body_has_no_key_prefix(client):
    resp = client.post("/classify", json={"email_text": "You charged me twice."})
    assert KEY_PREFIX not in resp.get_data(as_text=True)


def test_502_response_body_has_no_key_prefix(make_app, caplog):
    # RaisingProvider embeds a fake key-shaped string in its exception message;
    # the error path must not surface it to the client.
    classifier = TicketClassifier(
        EmailPreprocessor(), RaisingProvider(), PromptBuilder()
    )
    app = make_app(classifier)
    with caplog.at_level(logging.DEBUG):
        resp = app.test_client().post("/classify", json={"email_text": "boom"})
    assert resp.status_code == 502
    assert KEY_PREFIX not in resp.get_data(as_text=True)


def test_classify_with_key_shaped_email_does_not_echo_it(client):
    # Even if a user pastes a key-shaped string, the JSON result must not echo it.
    resp = client.post(
        "/classify", json={"email_text": "here is my key gsk_SHOULDNOTAPPEAR please help"}
    )
    assert resp.status_code == 200
    assert KEY_PREFIX not in resp.get_data(as_text=True)
