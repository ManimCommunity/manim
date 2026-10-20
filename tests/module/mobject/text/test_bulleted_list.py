from __future__ import annotations

from manim import BLUE, RED, WHITE, BulletedList, TexTemplate


def test_bullets_inherit_list_color():
    items = BulletedList("First", "Second", color=BLUE)

    assert len(items) == 2
    for item in items:
        bullet = item[0]
        assert item[1].get_color() == BLUE
        assert bullet.family_members_with_points()
        assert all(
            glyph.get_fill_color() == BLUE
            for glyph in bullet.family_members_with_points()
        )
        assert all(
            glyph.get_stroke_color() == BLUE
            for glyph in bullet.family_members_with_points()
        )


def test_bullet_uses_list_color_without_recoloring_tex():
    template = TexTemplate()
    template.add_to_preamble(r"\usepackage{xcolor}")
    items = BulletedList(
        r"\textcolor[HTML]{FC6255}{First}", "Second", color=BLUE, tex_template=template
    )

    text_glyphs = items[0][1].family_members_with_points()
    assert text_glyphs
    assert all(glyph.get_fill_color() == RED for glyph in text_glyphs)
    for item in items:
        glyphs = item[0].family_members_with_points()
        assert glyphs
        assert all(glyph.get_fill_color() == BLUE for glyph in glyphs)


def test_none_color_uses_default_bullet_color():
    items = BulletedList("First", "Second", color=None)

    for item in items:
        glyphs = item[0].family_members_with_points()
        assert glyphs
        assert all(glyph.get_fill_color() == WHITE for glyph in glyphs)
