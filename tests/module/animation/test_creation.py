from __future__ import annotations

import numpy as np
import pytest

from manim import AddTextLetterByLetter, Text


def test_non_empty_text_creation():
    """Check if AddTextLetterByLetter works for non-empty text."""
    s = Text("Hello")
    anim = AddTextLetterByLetter(s)
    assert anim.mobject.text == "Hello"


def test_empty_text_creation():
    """Ensure ValueError is raised for empty text."""
    with pytest.raises(ValueError, match="does not seem to contain any characters"):
        AddTextLetterByLetter(Text(""))


def test_whitespace_text_creation():
    """Ensure ValueError is raised for whitespace-only text, assuming the whitespace characters have no points."""
    with pytest.raises(ValueError, match="does not seem to contain any characters"):
        AddTextLetterByLetter(Text("    "))


def test_run_time_for_non_empty_text(config):
    """Ensure the run_time is calculated correctly for non-empty text."""
    s = Text("Hello")
    run_time_per_char = 0.1
    expected_run_time = np.max((1 / config.frame_rate, run_time_per_char)) * len(s.text)
    anim = AddTextLetterByLetter(s, time_per_char=run_time_per_char)
    assert anim.run_time == expected_run_time
