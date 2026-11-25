from __future__ import annotations

from pathlib import Path

from manim import RIGHT, BraceLabel, Circle, Dot, Mobject, Square, VGroup


def test_mobject_copy():
    """Test that a copy is a deepcopy."""
    orig = Mobject()
    orig.add(*(Mobject() for _ in range(10)))
    copy = orig.copy()

    assert orig is orig
    assert orig is not copy
    assert orig.submobjects is not copy.submobjects
    for i in range(10):
        assert orig.submobjects[i] is not copy.submobjects[i]


def test_bracelabel_copy(tmp_path, config):
    """Test that a copy is a deepcopy."""
    # For this test to work, we need to tweak some folders temporarily
    original_text_dir = config["text_dir"]
    original_tex_dir = config["tex_dir"]
    mediadir = Path(tmp_path) / "deepcopy"
    config["text_dir"] = str(mediadir.joinpath("Text"))
    config["tex_dir"] = str(mediadir.joinpath("Tex"))
    for el in ["text_dir", "tex_dir"]:
        Path(config[el]).mkdir(parents=True, exist_ok=True)

    # Before the refactoring of Mobject.copy(), the class BraceLabel was the
    # only one to have a non-trivial definition of copy.  Here we test that it
    # still works after the refactoring.
    orig = BraceLabel(Mobject(), "label")
    copy = orig.copy()

    assert orig is orig
    assert orig is not copy
    assert orig.brace is not copy.brace
    assert orig.label is not copy.label
    assert orig.submobjects is not copy.submobjects
    assert orig.submobjects[0] is orig.brace
    assert copy.submobjects[0] is copy.brace
    assert orig.submobjects[0] is not copy.brace
    assert copy.submobjects[0] is not orig.brace

    # Restore the original folders
    config["text_dir"] = original_text_dir
    config["tex_dir"] = original_tex_dir


def test_a_member_of_a_group_with_a_fixed_orientation_is_copied_alone():
    # fix_orientation stores the group's get_center in every member; copying a
    # member must not copy the group along with it (review of #5173).
    group = VGroup(Square(), Circle().shift(3 * RIGHT))
    group.fix_orientation()

    member = group[0].copy()

    assert member.fixed_orientation.__self__ is group


def test_a_copied_group_with_a_fixed_orientation_follows_its_copy():
    group = VGroup(Square(), Circle().shift(3 * RIGHT))
    group.fix_orientation()

    copied = group.copy()

    for mob in copied.get_family():
        assert mob.fixed_orientation.__self__ is copied


def test_a_fixed_orientation_that_follows_another_mobject_still_does_when_copied():
    anchor = Dot()
    label = Square().fix_orientation(anchor.get_center)

    assert label.copy().fixed_orientation.__self__ is anchor


def test_changing_a_color_in_place_does_not_change_a_copy():
    # HSV colors have setters: copies do not share colors (review of #5173).
    from manim.utils.color import HSV

    mob = Mobject()
    mob.color = HSV((0.1, 1, 1))
    copied = mob.copy()
    assert copied.color is not mob.color
    mob.color.hue = 0.6
    assert copied.color.to_hex() == HSV((0.1, 1, 1)).to_hex()
