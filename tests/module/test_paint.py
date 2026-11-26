"""Paints are immutable values that vectorized mobjects are coloured with."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from manim import (
    BLUE,
    DOWN,
    GREEN,
    PI,
    RED,
    RIGHT,
    UL,
    ColorField,
    LinearGradient,
    SolidColor,
    Square,
    Transform,
    interpolate_paint,
)
from manim.appearance.paint import Paint
from manim.utils.space_ops import rotate_vector
from manim.utils.versioned_fields import field_version


def test_paints_cannot_be_changed():
    solid = SolidColor(RED, 0.5)
    gradient = LinearGradient([RED, BLUE])
    for paint in (solid, gradient):
        with pytest.raises(AttributeError):
            paint._rgbas = np.zeros((1, 4))
        with pytest.raises(ValueError, match="read-only"):
            paint.rgbas[0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        gradient.direction[0] = 1


def test_a_paint_copies_what_it_is_given():
    opacities = np.array([0.2, 0.8])
    direction = np.array([1.0, 0.0, 0.0])
    gradient = LinearGradient([RED, BLUE], opacities, direction)
    opacities[0], direction[0] = 1, 0
    np.testing.assert_array_equal(gradient.opacities, [0.2, 0.8])
    np.testing.assert_array_equal(gradient.direction, RIGHT)


def test_gradients_default_to_up_left():
    np.testing.assert_array_equal(LinearGradient([RED, BLUE]).direction, UL)


def test_interpolation_keeps_the_paints_themselves():
    start, end = SolidColor(RED), LinearGradient([BLUE, GREEN], direction=RIGHT)
    assert interpolate_paint(start, end, 0) is start
    assert interpolate_paint(start, end, 1) is end
    assert interpolate_paint(start, start, 0.3) is start
    middle = interpolate_paint(start, end, 0.5)
    assert isinstance(middle, LinearGradient)
    np.testing.assert_allclose(
        middle.rgbas, (np.repeat(start.rgbas, 2, axis=0) + end.rgbas) / 2
    )
    # A solid takes the other paint's direction.
    np.testing.assert_array_equal(middle.direction, RIGHT)


def test_paints_of_unequal_length_are_stretched():
    three = LinearGradient([RED, GREEN, BLUE], opacities=[0, 0.5, 1])
    two = LinearGradient([BLUE, RED])
    middle = interpolate_paint(two, three, 0.5)
    assert len(middle.rgbas) == 3


def test_a_geometry_only_transform_changes_no_paint_version():
    square = Square().set_fill(RED, opacity=1)
    target = square.copy().shift(RIGHT).scale(0.5)
    versions = {name: field_version(square, name) for name in ("fill", "stroke")}
    animation = Transform(square, target)
    animation.begin()
    for alpha in (0.25, 0.5, 1):
        animation.interpolate(alpha)
    assert {name: field_version(square, name) for name in versions} == versions


def test_rotate_turns_gradients_and_leaves_solids():
    square = Square().set_fill([RED, BLUE], opacity=1)
    stroke = square.stroke
    square.rotate(PI / 2)
    np.testing.assert_allclose(square.fill.direction, rotate_vector(UL, PI / 2))
    assert square.stroke is stroke


def test_setters_keep_what_they_are_not_given():
    square = Square().set_fill([RED, BLUE], opacity=1).rotate(PI / 4)
    direction = square.fill.direction
    square.set_fill(opacity=0.5)
    np.testing.assert_array_equal(square.fill.direction, direction)
    assert square.get_fill_colors() == [RED, BLUE]
    square.set_fill(GREEN)
    np.testing.assert_allclose(square.get_fill_opacities(), [0.5, 0.5])


def test_setters_take_paints():
    gradient = LinearGradient([RED, BLUE], direction=DOWN)
    square = Square().set_fill(gradient)
    assert square.fill is gradient
    assert square.set_stroke(SolidColor(GREEN)).stroke.color == GREEN


def test_match_style_carries_the_paint():
    source = Square().set_fill(LinearGradient([RED, BLUE], direction=DOWN))
    target = Square().match_style(source)
    assert target.fill is source.fill


def test_copies_share_their_paints():
    square = Square().set_fill(RED, opacity=1)
    assert square.copy().fill is square.fill


_HASH = """
from manim import RED, Square, LinearGradient
from manim.utils.hashing import _get_json, _Memoizer
square = Square().set_fill(LinearGradient([RED, "#00FF00"]), opacity=1)
print(_get_json(square.fill, _Memoizer()))
"""


def test_paints_hash_by_content_in_any_process():
    from manim.utils.hashing import _get_json, _Memoizer

    def json(paint: Paint) -> str:
        return _get_json(paint, _Memoizer())

    assert json(SolidColor(RED)) == json(SolidColor(RED))
    assert json(SolidColor(RED)) != json(SolidColor(BLUE))
    assert json(LinearGradient([RED, BLUE])) != json(
        LinearGradient([RED, BLUE], direction=RIGHT)
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", _HASH], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    assert len(runs) == 1


def _field(opacity=1.0, corners=None):
    return ColorField(
        [[[1, 0, 0, 1], [0, 0, 1, 1]]],
        corners or [[-2, 1, 0], [2, 1, 0], [-2, -1, 0]],
        opacity,
    )


def test_fields_on_one_grid_fade_and_other_pairs_switch_at_the_end():
    field = _field()
    faded = field.with_opacity(0)
    middle = interpolate_paint(field, faded, 0.25)
    assert middle.same_grid(field)
    assert middle.opacity == pytest.approx(0.75)
    assert interpolate_paint(field, faded, 1) is faded
    other = _field(corners=[[-2, 2, 0], [2, 2, 0], [-2, -2, 0]])
    assert interpolate_paint(field, other, 0.99) is field
    assert interpolate_paint(field, other, 1) is other
    solid = SolidColor(GREEN)
    assert interpolate_paint(solid, field, 0.5) is solid
    assert interpolate_paint(field, solid, 1) is solid


def test_a_field_comes_back_with_there_and_back():
    from manim import there_and_back

    square = Square().set_stroke(_field())
    target = square.copy().set_stroke(SolidColor(GREEN))
    animation = Transform(square, target, rate_func=there_and_back)
    animation.begin()
    animation.interpolate(1)
    assert isinstance(square.stroke, ColorField)


def test_opacity_changes_keep_the_grid():
    field = _field()
    square = Square().set_stroke(field)
    square.set_stroke(opacity=0.5)
    assert square.stroke.same_grid(field)
    assert square.stroke.opacity == 0.5
    square.fade(0.5)
    assert square.stroke.opacity == pytest.approx(0.25)
    assert square.stroke.same_grid(field)


def test_match_style_carries_a_field():
    source = Square().set_stroke(_field())
    assert Square().match_style(source).stroke is source.stroke


def test_field_hashes_depend_on_the_content():
    from manim.utils.hashing import _get_json, _Memoizer

    def json(paint):
        return _get_json(paint, _Memoizer())

    assert json(_field()) == json(_field())
    assert json(_field()) != json(_field(0.5))
    assert json(_field()) != json(_field(corners=[[-2, 2, 0], [2, 2, 0], [-2, -2, 0]]))


def test_from_function_samples_texel_centres():
    field = ColorField.from_function(
        lambda point: (point[0] > 0, 0, 1), (-2, 2), (-1, 1), (4, 2)
    )
    assert field.grid.shape == (2, 4, 4)
    np.testing.assert_array_equal(field.grid[0, :, 0], [0, 0, 1, 1])
    vectorized = ColorField.from_function(
        lambda points: np.column_stack(
            [points[:, 0] > 0, np.zeros(len(points)), np.ones(len(points))]
        ),
        (-2, 2),
        (-1, 1),
        (4, 2),
        vectorized=True,
    )
    np.testing.assert_array_equal(vectorized.grid, field.grid)


class _Stripes(Paint):
    """A paint of one's own: two colours, drawn as the built-in gradient of them."""

    __slots__ = ("_turns",)

    def __init__(self, turns: float) -> None:
        object.__setattr__(self, "_rgbas", np.array([[1.0, 0, 0, 1], [0, 0, 1, 1]]))
        object.__setattr__(self, "_turns", float(turns))

    def resolve(self, points):
        from manim.renderer.snapshot import RLinearGradient

        return RLinearGradient(np.zeros(3), np.array([self._turns, 0, 0]), self._rgbas)

    def interpolate(self, end, alpha):
        if not isinstance(end, _Stripes):
            return super().interpolate(end, alpha)
        return _Stripes((1 - alpha) * self._turns + alpha * end._turns)


class _Opaque(Paint):
    """A paint of one's own that resolves to nothing a renderer knows."""

    __slots__ = ()

    def __init__(self) -> None:
        object.__setattr__(self, "_rgbas", np.ones((1, 4)))

    def resolve(self, points):
        return ("my", "paint")


def test_paints_resolve_themselves_into_snapshot_paints():
    from manim.renderer.extract import snapshot
    from manim.renderer.snapshot import RColorField, RLinearGradient, RSolidColor

    square = Square()
    for paint, kind in (
        (SolidColor(RED), RSolidColor),
        (LinearGradient([RED, BLUE], direction=RIGHT), RLinearGradient),
        (ColorField(np.ones((2, 2, 4)), np.eye(3)), RColorField),
        (_Stripes(2), RLinearGradient),
    ):
        square.fill = paint
        assert isinstance(snapshot(square).payload.fill, kind)
    # The gradient's ends: the box of the anchors along its direction.
    square.fill = LinearGradient([RED, BLUE], direction=RIGHT)
    fill = snapshot(square).payload.fill
    np.testing.assert_allclose(fill.start, [-1, 0, 0])
    np.testing.assert_allclose(fill.end, [1, 0, 0])


def test_paints_of_ones_own_interpolate_and_hash_by_their_values():
    from manim.utils.hashing import get_json

    two, three = _Stripes(2), _Stripes(3)
    # interpolate_paint asks the paint itself; at the ends, the paints themselves.
    assert interpolate_paint(two, three, 0.5)._turns == 2.5
    assert interpolate_paint(two, three, 1) is three
    # Without an interpolate of its own, a paint switches at the end.
    assert interpolate_paint(_Opaque(), two, 0.5)._rgbas.shape == (1, 4)
    assert interpolate_paint(_Opaque(), two, 1) is two
    assert get_json(two) != get_json(three)
    assert get_json(two) == get_json(_Stripes(2))
    with pytest.raises(NotImplementedError, match="resolve"):
        Paint.resolve(two, np.zeros((0, 3)))


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
def test_renderers_raise_for_snapshot_paints_they_do_not_know(renderer):
    from manim import Scene, tempconfig
    from manim.renderer.protocol import CapabilityError

    with tempconfig(
        {"renderer": renderer, "dry_run": True, "pixel_width": 64, "pixel_height": 36}
    ):
        for painted in (
            Square(fill_opacity=1).set_fill(_Opaque()),
            Square().set_stroke(_Opaque(), width=4),
        ):
            scene = Scene()
            scene.add(painted)
            try:
                with pytest.raises(CapabilityError, match="_Opaque|tuple"):
                    scene.get_image()
            finally:
                scene.renderer.close()
        # A paint of one's own that resolves to a built-in snapshot paint is drawn.
        scene = Scene()
        scene.add(Square(fill_opacity=1).set_fill(_Stripes(2)))
        try:
            assert np.asarray(scene.get_image())[..., :3].any()
        finally:
            scene.renderer.close()
