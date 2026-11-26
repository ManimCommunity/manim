from __future__ import annotations

import manim
import manim.appearance
from manim.appearance import light, material, paint


def test_appearance_exports_the_names_of_its_modules():
    modules = (light, material, paint)
    assert set(manim.appearance.__all__) == {
        name for module in modules for name in module.__all__
    }
    for module in modules:
        for name in module.__all__:
            assert getattr(manim.appearance, name) is getattr(module, name)
            assert getattr(manim, name) is getattr(module, name)


def test_the_light_names_are_listed_where_they_are_loaded():
    from manim.appearance import _LIGHT_NAMES

    assert set(light.__all__) == _LIGHT_NAMES
