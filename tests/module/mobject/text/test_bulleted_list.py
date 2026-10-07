from __future__ import annotations

import pytest

from manim import BLACK, BLUE, RED, WHITE, BulletedList


@pytest.fixture
def compiled_tex_svg(tmp_path, monkeypatch):
    """Isolate list styling from the external LaTeX compiler."""

    def compile_svg(expression, **kwargs):
        bullet = "\\cdot" in expression
        svg = tmp_path / ("bullet.svg" if bullet else "list.svg")
        fill = ' fill="#fc6255"' if "textcolor" in expression else ""
        paths = f'<g id="unique000"><path{fill} d="M0 0 L1 0 L1 1 L0 1 Z"/></g>'
        if not bullet:
            paths += '<g id="unique001"><path d="M0 2 L1 2 L1 3 L0 3 Z"/></g>'
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="20">'
            f'<g id="root">{paths}</g></svg>',
        )
        return svg

    monkeypatch.setattr(
        "manim.mobject.text.tex_mobject.tex_to_svg_file",
        compile_svg,
    )


@pytest.mark.parametrize("color", [BLACK, BLUE, WHITE])
def test_bullets_inherit_list_color(compiled_tex_svg, color):
    items = BulletedList("First", "Second", color=color)

    assert len(items) == 2
    for item in items:
        bullet = item[0]
        assert item[1].get_color() == color
        assert bullet.family_members_with_points()
        assert all(
            glyph.get_fill_color() == color
            for glyph in bullet.family_members_with_points()
        )
        assert all(
            glyph.get_stroke_color() == color
            for glyph in bullet.family_members_with_points()
        )


def test_bullet_uses_list_color_without_recoloring_tex(compiled_tex_svg):
    items = BulletedList(r"\textcolor{red}{First}", "Second", color=BLUE)

    assert items[0][1].get_fill_color() == RED
    for item in items:
        assert len(item[0].family_members_with_points()) == 1
        assert item[0].family_members_with_points()[0].get_fill_color() == BLUE


def test_none_color_uses_default_bullet_color(compiled_tex_svg):
    items = BulletedList("First", "Second", color=None)

    for item in items:
        assert item[0].family_members_with_points()[0].get_fill_color() == WHITE
