from __future__ import annotations

import numpy as np
import pytest

from manim import (
    RIGHT,
    Label,
    MathTex,
    NumberLine,
    Tex,
    Typst,
    TypstMath,
    Vector,
    VectorScene,
)


def test_Typst(config):
    """Basic Typst creation produces an SVG file."""
    m = Typst(r"$ x^2 $")
    assert m.height > 0
    assert m.width > 0
    assert len(m.submobjects) > 0


def test_TypstMath(config):
    """TypstMath wraps the expression in math delimiters."""
    m = TypstMath(r"alpha + beta")
    assert m.typst_code == "$ alpha + beta $"
    assert m.height > 0


def test_typst_default_font_size(config):
    """Default font_size is 48 (DEFAULT_FONT_SIZE)."""
    m = Typst(r"$ a + b $")
    assert np.isclose(m.font_size, 48)


def test_typst_custom_font_size(config):
    """Passing a custom font_size scales the mobject accordingly."""
    m = Typst(r"$ a + b $", font_size=72)
    assert np.isclose(m.font_size, 72)


def test_typst_font_size_property_setter(config):
    """Setting font_size after creation rescales correctly."""
    m = Typst(r"$ a + b $")
    original_height = m.height
    m.font_size = 96
    assert np.isclose(m.font_size, 96)
    assert m.height > original_height


def test_typst_font_size_error(config):
    """Setting font_size to a non-positive value raises ValueError."""
    m = Typst(r"$ a + b $")
    with pytest.raises(ValueError, match="font_size must be greater than 0"):
        m.font_size = -1


def test_typst_caching(config):
    """Compiling the same source twice uses the cached SVG."""
    m1 = Typst(r"$ e^{i pi} + 1 = 0 $")
    m2 = Typst(r"$ e^{i pi} + 1 = 0 $")
    assert np.isclose(m1.height, m2.height)
    assert np.isclose(m1.width, m2.width)


def test_typst_preamble(config):
    """A custom preamble is accepted without error."""
    m = Typst(
        r"$ x^2 $",
        typst_preamble='#set text(font: "New Computer Modern")',
    )
    assert m.height > 0


def test_typst_repr(config):
    """__repr__ includes the Typst source."""
    m = Typst("hello")
    assert repr(m) == "Typst('hello')"

    m2 = TypstMath("x")
    assert repr(m2) == "TypstMath('$ x $')"


def test_typst_text_rendering(config):
    """Non-math Typst markup renders correctly."""
    m = Typst(r"*Bold* and _italic_")
    assert m.height > 0
    assert len(m.submobjects) > 0


def test_typst_preserves_svg_stroke_widths_by_default(config):
    """Default stroke_width=None preserves Typst-authored SVG strokes."""
    m = Typst("#underline[abc]", use_svg_cache=False)
    assert any(submobject.stroke_width > 0 for submobject in m.submobjects)


def test_typst_text_font_size_matches_tex_closely(config):
    """Typst text is calibrated close to Tex for the same font_size."""
    tex = Tex("Hello", font_size=48)
    typst = Typst("Hello", font_size=48, use_svg_cache=False)
    assert np.isclose(typst.height, tex.height, rtol=0.02)
    assert np.isclose(typst.width, tex.width, rtol=0.02)


def test_typstmath_font_size_matches_mathtex_closely(config):
    """Typst math is calibrated close to MathTex for the same font_size."""
    mathtex = MathTex(r"\frac{a}{b}", font_size=48)
    typstmath = TypstMath("frac(a,b)", font_size=48, use_svg_cache=False)
    assert np.isclose(typstmath.height, mathtex.height, rtol=0.02)
    assert np.isclose(typstmath.width, mathtex.width, rtol=0.02)


# -- data-typst-label → id mapping tests ------------------------------------

MANIMGRP_PREAMBLE = '#let manimgrp(lbl, body) = [#box(body) #label(lbl)]'


def test_typst_labels_mapped_to_vgroups(config):
    """data-typst-label attributes are promoted to id and appear in id_to_vgroup_dict."""
    m = Typst(
        '$ #manimgrp("numer", $a + b$) / #manimgrp("denom", $c - d$) $',
        typst_preamble=MANIMGRP_PREAMBLE,
        use_svg_cache=False,
    )
    assert "numer" in m.id_to_vgroup_dict
    assert "denom" in m.id_to_vgroup_dict
    # a, +, b → 3 submobjects; c, -, d → 3 submobjects
    assert len(m.id_to_vgroup_dict["numer"]) == 3
    assert len(m.id_to_vgroup_dict["denom"]) == 3


def test_typst_nested_labels(config):
    """Nested labeled boxes produce nested VGroups without cross-contamination."""
    m = Typst(
        '$ #manimgrp("outer", $#manimgrp("inner", $a$) + b$) $',
        typst_preamble=MANIMGRP_PREAMBLE,
        use_svg_cache=False,
    )
    assert "outer" in m.id_to_vgroup_dict
    assert "inner" in m.id_to_vgroup_dict
    # "inner" contains only "a" (1 submobject)
    assert len(m.id_to_vgroup_dict["inner"]) == 1
    # "outer" contains everything: a, +, b (3 submobjects)
    assert len(m.id_to_vgroup_dict["outer"]) == 3
    # The inner submobject is a subset of the outer one
    inner_mob = m.id_to_vgroup_dict["inner"][0]
    assert inner_mob in m.id_to_vgroup_dict["outer"]


def test_typst_no_labels_no_extra_keys(config):
    """Without labeled boxes, no extra label keys appear."""
    m = Typst(r"$ a + b $", use_svg_cache=False)
    label_keys = [
        k
        for k in m.id_to_vgroup_dict
        if not k.startswith(("numbered_group", "root", "g"))
    ]
    assert label_keys == []


def test_typst_select(config):
    """select() returns the correct VGroup for a given label."""
    m = Typst(
        '$ #manimgrp("lhs", $a + b$) = #manimgrp("rhs", $c$) $',
        typst_preamble=MANIMGRP_PREAMBLE,
        use_svg_cache=False,
    )
    lhs = m.select("lhs")
    rhs = m.select("rhs")
    assert len(lhs) == 3  # a, +, b
    assert len(rhs) == 1  # c


def test_typst_select_keyerror(config):
    """select() raises KeyError for a nonexistent label."""
    m = Typst(r"$ a + b $", use_svg_cache=False)
    with pytest.raises(KeyError, match="No group with label 'missing'"):
        m.select("missing")


# -- {{ }} double-brace preprocessor tests ----------------------------------


def test_typstmath_double_brace_auto_numbered(config):
    """{{ }} groups are auto-numbered and selectable by index."""
    eq = TypstMath("{{ a + b }} / {{ c - d }} = {{ x }}", use_svg_cache=False)
    assert eq._group_labels == ["_grp-0", "_grp-1", "_grp-2"]
    assert len(eq.select(0)) == 3  # a, +, b
    assert len(eq.select(1)) == 3  # c, -, d
    assert len(eq.select(2)) == 1  # x


def test_typstmath_double_brace_named(config):
    """{{ content : label }} assigns an explicit label."""
    eq = TypstMath(
        "{{ a + b : numer }} / {{ c - d : denom }}", use_svg_cache=False
    )
    assert "numer" in eq._group_labels
    assert "denom" in eq._group_labels
    assert len(eq.select("numer")) == 3
    assert len(eq.select("denom")) == 3


def test_typstmath_double_brace_mixed_named_auto(config):
    """Named and auto-numbered groups can coexist."""
    eq = TypstMath(
        "{{ a : lhs }} = {{ b }}", use_svg_cache=False
    )
    assert eq._group_labels == ["lhs", "_grp-0"]
    assert len(eq.select("lhs")) == 1
    assert len(eq.select(0)) == 1


def test_typstmath_no_braces_no_preamble(config):
    """Without {{ }}, the manimgrp preamble is not injected."""
    eq = TypstMath("a + b", use_svg_cache=False)
    assert eq._group_labels == []
    assert "manimgrp" not in eq.typst_preamble


def test_typstmath_select_index_error(config):
    """select(int) raises IndexError for out-of-range index."""
    eq = TypstMath("{{ a }}", use_svg_cache=False)
    with pytest.raises(IndexError, match="out of range"):
        eq.select(1)


def test_typstmath_preprocessor_skips_strings():
    """{{ }} inside string literals are not processed."""
    processed, labels = TypstMath._preprocess_groups(
        'x =_("{{ not a group }}") z'
    )
    assert labels == []
    assert "manimgrp" not in processed


def test_typstmath_preprocessor_skips_content_blocks():
    """{{ }} inside [...] content blocks are not processed."""
    processed, labels = TypstMath._preprocess_groups(
        "[text {{ here }}] {{ real }}"
    )
    assert labels == ["_grp-0"]
    assert processed.count("manimgrp") == 1


# -- integration tests for existing APIs ------------------------------------


def test_label_accepts_typst(config):
    """Label accepts a prebuilt Typst mobject."""
    rendered = Typst("hello", use_svg_cache=False)
    label = Label(rendered)
    assert label.rendered_label is rendered


def test_numberline_add_labels_with_typstmath_constructor_uses_typst(config):
    """String labels use Typst text mode when label_constructor is TypstMath."""
    number_line = NumberLine(x_range=[-1, 1])
    number_line.add_labels({0: "origin"}, label_constructor=TypstMath)
    assert len(number_line.labels) == 1
    assert isinstance(number_line.labels[0], Typst)
    assert not isinstance(number_line.labels[0], TypstMath)


def test_vector_scene_get_vector_label_accepts_typst(config):
    """VectorScene accepts a prebuilt Typst label mobject."""
    scene = VectorScene()
    vector = Vector(RIGHT)
    label = Typst("v", use_svg_cache=False)
    returned = scene.get_vector_label(vector, label)
    assert returned is label
