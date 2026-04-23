from stereotype_bench.text_clean import clean_model_output


def test_empty_and_none_input():
    assert clean_model_output("") == ""


def test_plain_text_is_unchanged():
    s = "Michael is a 30-year-old man with a thoughtful demeanor."
    assert clean_model_output(s) == s


def test_strips_leading_markdown_header():
    raw = "# Michael\n\nMichael is a 30-year-old man with a thoughtful demeanor."
    assert clean_model_output(raw) == "Michael is a 30-year-old man with a thoughtful demeanor."


def test_strips_leading_h2_h3_headers():
    for prefix in ("## Name", "### Profile"):
        raw = f"{prefix}\n\nBody content here."
        assert clean_model_output(raw) == "Body content here."


def test_strips_mid_text_headers():
    raw = "Opening sentence.\n\n## Section\n\nMore content."
    assert clean_model_output(raw) == "Opening sentence. More content."


def test_strips_bold_and_italic_markers():
    raw = "She is **warm** and *caring*, a _gentle_ soul."
    assert clean_model_output(raw) == "She is warm and caring, a gentle soul."


def test_keeps_underscores_inside_words():
    """Snake_case tokens shouldn't be mangled by the italic regex."""
    raw = "He uses snake_case_vars in code."
    assert clean_model_output(raw) == "He uses snake_case_vars in code."


def test_strips_inline_and_fenced_code():
    raw = "Run `pip install x`\n```\ncode block\n```\nand continue."
    assert clean_model_output(raw) == "Run pip install x code block and continue."


def test_strips_strikethrough():
    raw = "This is ~~not~~ a good idea."
    assert clean_model_output(raw) == "This is not a good idea."


def test_collapses_whitespace():
    raw = "Line one.\n\n\nLine   two.\nLine three."
    assert clean_model_output(raw) == "Line one. Line two. Line three."


def test_strips_common_preamble():
    raw = "Here's a 30-word description of Sarah:\n\nSarah is warm and caring."
    assert clean_model_output(raw) == "Sarah is warm and caring."


def test_strips_preamble_without_word_count():
    raw = "Here is a description:\n\nBody."
    assert clean_model_output(raw) == "Body."


def test_combined_real_world_example():
    """A typical Claude-Haiku output with markdown header + paragraph."""
    raw = "# Sarah\n\nSarah is a **30-year-old** woman with a thoughtful, composed demeanor. She demonstrates *wisdom* beyond her years, balancing ambition with compassion."
    expected = "Sarah is a 30-year-old woman with a thoughtful, composed demeanor. She demonstrates wisdom beyond her years, balancing ambition with compassion."
    assert clean_model_output(raw) == expected
