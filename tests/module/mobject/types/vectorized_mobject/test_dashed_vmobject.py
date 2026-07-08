import pytest

from manim import ORIGIN, RIGHT, UR, Arrow, DashedLine, DashedVMobject, VGroup
from manim.mobject.geometry.tips import ArrowTip, StealthTip


def _collect_tips(mobject):
    return [mob for mob in mobject.get_family() if isinstance(mob, ArrowTip)]


def test_dashed_arrow_has_single_tip():
    dashed = DashedVMobject(Arrow(ORIGIN, 2 * UR))
    tips = _collect_tips(dashed)

    assert len(tips) == 1


def test_dashed_arrow_tip_not_duplicated_in_group_opacity():
    base_arrow = Arrow(ORIGIN, 2 * UR)
    faded_arrow = base_arrow.copy().set_fill(opacity=0.4).set_stroke(opacity=0.4)

    dashed_group = (
        VGroup(DashedVMobject(faded_arrow))
        .set_fill(opacity=0.4, family=True)
        .set_stroke(opacity=0.4, family=True)
    )

    tips = _collect_tips(dashed_group)

    assert len(tips) == 1


def test_dashed_arrow_custom_tip_shape_has_single_tip():
    dashed = DashedVMobject(Arrow(ORIGIN, 2 * UR, tip_shape=StealthTip))
    tips = _collect_tips(dashed)

    assert len(tips) == 1
    assert isinstance(tips[0], StealthTip)


def test_dashed_arrow_with_start_tip_has_two_tips():
    dashed = DashedVMobject(Arrow(ORIGIN, 2 * UR).add_tip(at_start=True))
    tips = _collect_tips(dashed)

    assert len(tips) == 2


def test_zero_length_dashed_line_submobjects_have_2d_points():
    """Submobjects of a zero-length DashedLine must have 2-D point arrays."""
    line = DashedLine(ORIGIN, ORIGIN)
    for sub in line.submobjects:
        assert sub.points.ndim == 2, (
            f"Expected 2-D points array, got shape {sub.points.shape}"
        )


def test_become_nonzero_to_zero_dashed_line_does_not_crash():
    """become() from a normal DashedLine to a zero-length one should not crash."""
    normal = DashedLine(ORIGIN, 2 * UR)
    zero = DashedLine(ORIGIN, ORIGIN)
    normal.become(zero)


def test_dashed_line_recomputes_dashes_on_put_start_and_end_on():
    """Regression test for https://github.com/ManimCommunity/manim/issues/3989."""
    line = DashedLine(ORIGIN, RIGHT)
    initial_dash_length = line.submobjects[0].get_length()
    initial_num_dashes = len(line.submobjects)
    line.put_start_and_end_on(ORIGIN, 5 * RIGHT)
    assert len(line.submobjects) > initial_num_dashes
    assert line.submobjects[0].get_length() == pytest.approx(
        initial_dash_length, rel=0.05
    )
