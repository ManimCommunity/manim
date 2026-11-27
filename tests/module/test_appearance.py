from __future__ import annotations

import manim
import manim.appearance
from manim.appearance import material


def test_appearance_exports_the_names_of_its_modules():
    assert set(manim.appearance.__all__) == set(material.__all__)
    for name in manim.appearance.__all__:
        assert getattr(manim.appearance, name) is getattr(material, name)
        assert getattr(manim, name) is getattr(material, name)
