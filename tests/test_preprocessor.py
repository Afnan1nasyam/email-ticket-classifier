"""Offline tests for `EmailPreprocessor` — clean / tokenize / truncate / batch.

No network, no API key. Assertions target deterministic cleaned output,
tokenization behaviour, and truncation invariants.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.preprocessor import EmailPreprocessor


@pytest.fixture
def pre() -> EmailPreprocessor:
    return EmailPreprocessor()


# --------------------------------------------------------------------------- #
# clean() — empty / whitespace
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw", ["", "   ", "\n\n", "\t \r\n  "])
def test_clean_returns_empty_for_empty_or_whitespace(pre, raw):
    assert pre.clean(raw) == ""


def test_clean_normalizes_internal_whitespace_to_single_spaces(pre):
    assert pre.clean("Hello    world\n\n\tthere") == "Hello world there"


def test_clean_collapses_repeated_punctuation(pre):
    assert pre.clean("URGENT!!! help???") == "URGENT! help?"


# --------------------------------------------------------------------------- #
# clean() — HTML tags, entities, script/style
# --------------------------------------------------------------------------- #

def test_clean_strips_html_tags_and_decodes_entities(pre):
    out = pre.clean("<p>Hello</p><b>World</b>&amp;more")
    assert "<" not in out and ">" not in out
    assert "Hello" in out
    assert "World" in out
    assert "&" in out  # &amp; decoded
    assert "amp;" not in out


def test_clean_removes_script_block_and_its_contents(pre):
    out = pre.clean("<p>Hi</p><script>alert('x'); var k=1;</script>Bye")
    assert "alert" not in out
    assert "var k" not in out
    assert "Hi" in out and "Bye" in out


def test_clean_removes_style_block_and_its_contents(pre):
    out = pre.clean("<style>.a{color:red}</style>Visible text")
    assert "color" not in out
    assert "Visible text" in out


# --------------------------------------------------------------------------- #
# clean() — quoted reply chains and forward headers
# --------------------------------------------------------------------------- #

def test_clean_drops_quoted_reply_lines(pre):
    raw = "My actual message\n> old quoted line\n> another quoted line"
    assert pre.clean(raw) == "My actual message"


def test_clean_drops_on_wrote_reply_header_and_following_quote(pre):
    raw = "Please advise.\nOn Mon, Aug 25, 2025, John Doe wrote:\n> earlier text"
    assert pre.clean(raw) == "Please advise."


def test_clean_drops_original_message_header(pre):
    raw = "Latest note.\n----- Original Message -----\n> quoted body"
    assert pre.clean(raw) == "Latest note."


def test_clean_drops_forwarded_header_fields(pre):
    raw = "See below.\nFrom: a@b.com\nSent: yesterday\nSubject: Re: hi\nBody line"
    out = pre.clean(raw)
    assert "See below." in out
    assert "a@b.com" not in out
    assert "Subject" not in out


# --------------------------------------------------------------------------- #
# clean() — signature blocks and footers
# --------------------------------------------------------------------------- #

def test_clean_cuts_at_signature_delimiter(pre):
    raw = "The real content.\n-- \nJohn Doe\nAcme Corp\nCEO"
    assert pre.clean(raw) == "The real content."


def test_clean_cuts_at_sent_from_my_footer(pre):
    raw = "Thanks for the help.\nSent from my iPhone"
    assert pre.clean(raw) == "Thanks for the help."


def test_clean_cuts_at_confidentiality_notice_footer(pre):
    raw = "Approved.\nConfidentiality notice: this message is private."
    assert pre.clean(raw) == "Approved."


# --------------------------------------------------------------------------- #
# clean() — unicode / emoji / non-Latin / no-alpha
# --------------------------------------------------------------------------- #

def test_clean_preserves_unicode_and_emoji(pre):
    out = pre.clean("Café ☕ is great 😀")
    assert "Café" in out
    assert "☕" in out
    assert "😀" in out


def test_clean_preserves_non_latin_script(pre):
    # Arabic and CJK should survive cleaning untouched.
    assert pre.clean("مرحبا بك") == "مرحبا بك"
    assert pre.clean("你好 世界") == "你好 世界"


def test_clean_handles_text_with_no_alphabetic_characters(pre):
    assert pre.clean("12345 !!! ???") == "12345 ! ?"


# --------------------------------------------------------------------------- #
# tokenize()
# --------------------------------------------------------------------------- #

def test_tokenize_lowercases_and_drops_punctuation(pre):
    assert pre.tokenize("Hello, WORLD! Foo-bar.") == ["hello", "world", "foo", "bar"]


def test_tokenize_empty_string_returns_empty_list(pre):
    assert pre.tokenize("") == []


def test_tokenize_keeps_digits_when_no_alpha(pre):
    assert pre.tokenize("12345 !!! ???") == ["12345"]


def test_tokenize_keeps_apostrophe_contractions(pre):
    assert pre.tokenize("I don't know") == ["i", "don't", "know"]


# --------------------------------------------------------------------------- #
# truncate()
# --------------------------------------------------------------------------- #

def test_truncate_returns_text_unchanged_when_within_budget(pre):
    text = "short message"
    assert pre.truncate(text, max_tokens=512) == text


def test_truncate_shortens_oversized_input_within_char_budget(pre):
    text = "word " * 1000  # 5000 chars, far over budget
    max_tokens = 10
    budget = int(max_tokens * pre.CHARS_PER_TOKEN * pre.SAFETY_FACTOR)
    out = pre.truncate(text, max_tokens=max_tokens)
    assert len(out) <= budget


def test_truncate_does_not_cut_mid_word(pre):
    text = "word " * 1000
    out = pre.truncate(text, max_tokens=10)
    # Every surviving token is a complete "word"; no partial fragment.
    assert out.split() == ["word"] * len(out.split())
    assert not out.endswith(" ")


def test_truncate_empty_text_returns_empty(pre):
    assert pre.truncate("", max_tokens=10) == ""


# --------------------------------------------------------------------------- #
# preprocess_one()
# --------------------------------------------------------------------------- #

def test_preprocess_one_empty_input_returns_empty(pre):
    assert pre.preprocess_one("") == ""


def test_preprocess_one_cleans_then_truncates(pre):
    raw = "<p>Hello</p>   world\n> quoted"
    assert pre.preprocess_one(raw) == "Hello world"


# --------------------------------------------------------------------------- #
# preprocess_batch()
# --------------------------------------------------------------------------- #

def test_preprocess_batch_adds_clean_text_and_n_tokens_columns(pre):
    df = pd.DataFrame(
        {"email_text": ["<b>Hi</b> there", "Line one\n> quoted away", ""]}
    )
    out = pre.preprocess_batch(df)
    assert "clean_text" in out.columns
    assert "n_tokens" in out.columns
    assert out.loc[0, "clean_text"] == "Hi there"
    assert out.loc[0, "n_tokens"] == 2
    assert out.loc[1, "clean_text"] == "Line one"
    assert out.loc[2, "clean_text"] == ""
    assert out.loc[2, "n_tokens"] == 0


def test_preprocess_batch_raises_keyerror_for_missing_column(pre):
    df = pd.DataFrame({"other": ["x"]})
    with pytest.raises(KeyError):
        pre.preprocess_batch(df)


def test_preprocess_batch_does_not_mutate_input(pre):
    df = pd.DataFrame({"email_text": ["hello"]})
    pre.preprocess_batch(df)
    assert list(df.columns) == ["email_text"]
