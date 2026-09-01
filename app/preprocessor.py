"""Email preprocessing: cleaning, tokenization, and truncation.

`EmailPreprocessor` turns raw, messy customer email text into a clean string
suitable for classification, and provides a lightweight regex tokenizer used for
length control and dataset statistics.

The tokenizer is a regex word-boundary tokenizer — NOT the model's tokenizer and
NOT a feature extractor (the LLM does the classification). Truncation therefore
approximates the model's token budget by characters rather than tokenizing
exactly.
"""

from __future__ import annotations

import html
import logging
import re

import pandas as pd

from app import config

logger = logging.getLogger(__name__)


class EmailPreprocessor:
    """Clean, tokenize, and truncate customer email text.

    Exposes a single-item path (`preprocess_one`) used by the API and a pandas
    batch path (`preprocess_batch`) used by the eval runner. Both produce the
    same cleaned text per email, so the eval measures what the API would send.
    """

    # Block-level HTML tags whose removal should leave a line break behind.
    _BLOCK_TAGS_RE = re.compile(
        r"</?\s*(?:br|p|div|li|tr|h[1-6]|ul|ol|table)\b[^>]*>", re.IGNORECASE
    )
    # <script>/<style> ... </script>/</style> blocks (content included).
    _SCRIPT_STYLE_RE = re.compile(
        r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
    )
    # Any remaining HTML tag.
    _TAG_RE = re.compile(r"<[^>]+>")

    # A quoted reply line, e.g. "> previous message".
    _QUOTED_LINE_RE = re.compile(r"^\s*>+.*$")
    # Reply/forward header lines that introduce a quoted chain.
    _REPLY_HEADER_RE = re.compile(
        r"^\s*(?:"
        r"On\s.+\bwrote:\s*$"                     # "On Mon, Aug 25, X wrote:"
        r"|-{2,}\s*Original Message\s*-{2,}\s*$"   # "----- Original Message -----"
        r"|(?:From|Sent|To|Subject|Cc|Date)\s*:\s.*$"  # forwarded header fields
        r")",
        re.IGNORECASE,
    )
    # Signature delimiter (RFC-3676 "-- ") — cut it and everything after.
    _SIG_DELIM_RE = re.compile(r"^--\s*$")
    # Common footer/disclaimer markers — cut from the first one onward.
    _FOOTER_MARKERS = (
        "sent from my ",
        "get outlook for",
        "confidentiality notice",
        "this email and any attachments",
        "disclaimer:",
    )

    # Runs of 2+ identical punctuation marks, collapsed to one.
    _REPEAT_PUNCT_RE = re.compile(r"([!?.,;:])\1+")
    # Whitespace normalization.
    _WS_RE = re.compile(r"\s+")

    # Word tokens for the regex tokenizer.
    _TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

    #: Average characters per token used to approximate the model's budget.
    CHARS_PER_TOKEN: int = 4
    #: Fraction of the estimated budget actually used, so truncation undershoots
    #: rather than overshoots the real token limit.
    SAFETY_FACTOR: float = 0.9

    def __init__(self, max_email_tokens: int = config.MAX_EMAIL_TOKENS) -> None:
        """Create a preprocessor.

        Args:
            max_email_tokens: token budget used by `preprocess_one` /
                `preprocess_batch` when truncating email bodies.
        """
        self.max_email_tokens = max_email_tokens

    # ------------------------------------------------------------------ #
    # Cleaning
    # ------------------------------------------------------------------ #

    def clean(self, text: str) -> str:
        """Strip HTML, quoted replies, and signatures, then normalize whitespace.

        Case is preserved (the LLM benefits from it); only tokenization
        lowercases. Returns a single-line, whitespace-normalized string.

        Args:
            text: raw email body.

        Returns:
            Cleaned email text. Empty string for empty/whitespace-only input.
        """
        if not text:
            return ""

        # 1. Normalize newlines and strip HTML.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._SCRIPT_STYLE_RE.sub(" ", text)
        text = self._BLOCK_TAGS_RE.sub("\n", text)
        text = self._TAG_RE.sub(" ", text)
        text = html.unescape(text)

        # 2. Line-based removal of quoted replies, headers, and signatures.
        kept: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if self._SIG_DELIM_RE.match(stripped):
                break  # signature delimiter: drop the rest
            if stripped.lower().startswith(self._FOOTER_MARKERS):
                break  # footer/disclaimer: drop the rest
            if self._QUOTED_LINE_RE.match(line):
                continue
            if self._REPLY_HEADER_RE.match(line):
                continue
            kept.append(line)
        text = "\n".join(kept)

        # 3. Collapse repeated punctuation ("URGENT!!!" -> "URGENT!").
        text = self._REPEAT_PUNCT_RE.sub(r"\1", text)

        # 4. Normalize all whitespace to single spaces.
        text = self._WS_RE.sub(" ", text).strip()
        return text

    # ------------------------------------------------------------------ #
    # Tokenization and truncation
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str) -> list[str]:
        """Split text into lowercased word tokens (regex word-boundary).

        Not the model's tokenizer; used for length control and dataset
        statistics only.

        Args:
            text: input text.

        Returns:
            List of lowercased tokens (possibly empty).
        """
        if not text:
            return []
        return self._TOKEN_RE.findall(text.lower())

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately `max_tokens`, on a word boundary.

        Approximates the token budget at ~`CHARS_PER_TOKEN` characters per token
        with a safety margin, so the result tends to undershoot the real limit.

        Args:
            text: input text.
            max_tokens: approximate token budget.

        Returns:
            `text` unchanged if within budget, otherwise a truncated prefix.
        """
        if not text or max_tokens <= 0:
            return "" if max_tokens <= 0 else text

        char_budget = int(max_tokens * self.CHARS_PER_TOKEN * self.SAFETY_FACTOR)
        if len(text) <= char_budget:
            return text

        clipped = text[:char_budget]
        # Prefer to cut at the last whitespace so we don't split a word.
        last_space = clipped.rfind(" ")
        if last_space > 0:
            clipped = clipped[:last_space]
        truncated = clipped.rstrip()
        logger.debug(
            "Truncated email from %d to %d chars (budget ~%d tokens)",
            len(text),
            len(truncated),
            max_tokens,
        )
        return truncated

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    def preprocess_one(self, text: str) -> str:
        """Clean and truncate a single email — the API classification path.

        Args:
            text: raw email body.

        Returns:
            Cleaned, budget-truncated email text.
        """
        return self.truncate(self.clean(text), self.max_email_tokens)

    def preprocess_batch(
        self, df: pd.DataFrame, text_column: str = "email_text"
    ) -> pd.DataFrame:
        """Clean/truncate a column of emails — the eval-runner path.

        Args:
            df: DataFrame containing raw email text.
            text_column: name of the column holding raw email bodies.

        Returns:
            A copy of `df` with two added columns: `clean_text` (the processed
            body the classifier would receive) and `n_tokens` (regex token count
            of `clean_text`).

        Raises:
            KeyError: if `text_column` is not present in `df`.
        """
        if text_column not in df.columns:
            raise KeyError(f"column {text_column!r} not found in DataFrame")

        out = df.copy()
        out["clean_text"] = out[text_column].fillna("").astype(str).map(self.preprocess_one)
        out["n_tokens"] = out["clean_text"].map(lambda t: len(self.tokenize(t)))
        logger.info(
            "Preprocessed %d emails (mean tokens=%.1f)",
            len(out),
            out["n_tokens"].mean() if len(out) else 0.0,
        )
        return out
