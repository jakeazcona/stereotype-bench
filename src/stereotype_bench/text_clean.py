"""Light output-cleaning utilities.

LLM outputs often include Markdown formatting (`# Headers`, `**bold**`,
code fences) that isn't part of the semantic content we want to score.
`clean_model_output` strips the most common of these while preserving
substantive text. Conservative by design: if a transformation might lose
meaning, we skip it.
"""
from __future__ import annotations

import re

# Leading Markdown ATX header on its own line, e.g. "# Name\n\n" or "## Foo\n"
_LEADING_HEADER_RE = re.compile(r"^\s*#+\s+[^\n]*\n+")

# Any ATX header line that remains later in the text
_INLINE_HEADER_RE = re.compile(r"(?m)^\s*#+\s+[^\n]*\n?")

# Markdown bold/italic/strike markers - keep the inner text.
_EMPHASIS_STAR_RE = re.compile(r"\*{1,3}([^*\n]+?)\*{1,3}")
_EMPHASIS_UND_RE = re.compile(r"(?<![A-Za-z0-9])_{1,3}([^_\n]+?)_{1,3}(?![A-Za-z0-9])")
_STRIKE_RE = re.compile(r"~~([^~\n]+?)~~")

# Fenced code blocks and inline code backticks.
_FENCED_CODE_RE = re.compile(r"```[a-zA-Z]*\n?")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")

# Common leading preambles the model might add before the real content.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:'s| is) (?:a |the )?(?:\d+[- ]word )?description(?:\s*of[^\:\n]*)?\s*:?\s*\n+)",
    re.IGNORECASE,
)

# Collapse whitespace (newlines included) to single spaces.
_WHITESPACE_RE = re.compile(r"\s+")


def clean_model_output(text: str) -> str:
    """Strip Markdown formatting + common preamble; collapse whitespace.

    Safe for text that has none of these (no-op).
    """
    if not text:
        return ""

    s = text.strip()

    # Order matters: strip headers before emphasis in case headers contain bold.
    s = _LEADING_HEADER_RE.sub("", s, count=1)
    s = _PREAMBLE_RE.sub("", s, count=1)
    s = _INLINE_HEADER_RE.sub("", s)
    s = _FENCED_CODE_RE.sub("", s)
    s = _INLINE_CODE_RE.sub(r"\1", s)
    s = _EMPHASIS_STAR_RE.sub(r"\1", s)
    s = _EMPHASIS_UND_RE.sub(r"\1", s)
    s = _STRIKE_RE.sub(r"\1", s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()
