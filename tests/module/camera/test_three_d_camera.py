from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from manim import (
    DEGREES,
    GREEN,
    RED,
    Camera,
    Circle,
    DecimalNumber,
    Square,
    Text,
    ThreeDScene,
    VGroup,
    tempconfig,
)


@pytest.fixture
def scene(dry_run, monkeypatch):
    # Exercise real DecimalNumber replacement using Pango instead of requiring TeX.
    monkeypatch.setattr("manim.mobject.text.numbers.string_to_mob_map", {})
    with tempconfig({"pixel_width": 320, "pixel_height": 180, "frame_height": 8}):
        result = ThreeDScene()
        result.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
        yield result


def _number(value=0):
    return DecimalNumber(
        value, num_decimal_places=0, mob_class=Text, color=RED
    ).move_to((2, 1, 0))


def _capture(scene):
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    return np.asarray(scene.camera.get_image()).astype(int)


def _assert_fixed(camera, mobject):
    for member in mobject.family_members_with_points():
        np.testing.assert_allclose(
            camera.transform_points_pre_display(member, member.points), member.points
        )


def _red_mask(image):
    return (
        (image[:, :, 0] > 150)
        & (image[:, :, 0] > 1.5 * image[:, :, 1])
        & (image[:, :, 0] > 1.5 * image[:, :, 2])
    )


@pytest.mark.parametrize("value", [1, 12, 100, -5])
def test_updated_decimal_number_stays_fixed_in_frame(scene, value):
    number = _number()
    circle = Circle(radius=0.3, color=GREEN).move_to((-2, 1, 0))
    scene.add_fixed_in_frame_mobjects(number, circle)
    number.set_value(value)

    actual = _capture(scene)
    reference = Camera()
    reference.capture_mobjects(scene.mobjects)
    expected = np.asarray(reference.get_image()).astype(int)

    assert np.count_nonzero(_red_mask(actual)) > 0
    np.testing.assert_array_equal(_red_mask(actual), _red_mask(expected))
    _assert_fixed(scene.camera, number)
    _assert_fixed(scene.camera, circle)


def test_removing_updated_decimal_number_restores_projection(scene):
    number = _number()
    circle = Circle().shift((-2, 1, 0))
    scene.add_fixed_in_frame_mobjects(number, circle)
    number.set_value(100)
    _capture(scene)

    scene.remove_fixed_in_frame_mobjects(number)
    _capture(scene)

    assert scene.camera.fixed_in_frame_mobjects == set(circle.get_family())
    for member in number.family_members_with_points():
        np.testing.assert_allclose(
            scene.camera.transform_points_pre_display(member, member.points),
            scene.camera.project_points(member.points),
        )
    _assert_fixed(scene.camera, circle)


def test_replaced_decimal_children_are_released(scene):
    number = _number()
    scene.add_fixed_in_frame_mobjects(number)
    old_digits = []

    for value in [1, 12, 100, 4, 55]:
        old_digits.append(weakref.ref(number[0]))
        number.set_value(value)
        _capture(scene)
        assert scene.camera.fixed_in_frame_mobjects == set(number.get_family())

    gc.collect()
    assert all(digit() is None for digit in old_digits)


def test_repeated_registration_does_not_accumulate_replaced_children(scene):
    number = _number()
    scene.add_fixed_in_frame_mobjects(number)

    for value in [1, 12, 100, 4, 55]:
        number.set_value(value)
        scene.add_fixed_in_frame_mobjects(number)
        scene.add_fixed_in_frame_mobjects(number)
        _capture(scene)
        assert scene.camera.fixed_in_frame_mobjects == set(number.get_family())
        _assert_fixed(scene.camera, number)


def test_nested_unfix_and_refix_keep_registration_order(scene):
    square = Square().shift((2, 1, 0))
    child = VGroup(square)
    circle = Circle().shift((-2, 1, 0))
    group = VGroup(child, circle)
    scene.add_fixed_in_frame_mobjects(group)
    scene.remove_fixed_in_frame_mobjects(child)
    _capture(scene)
    assert square not in scene.camera.fixed_in_frame_mobjects
    _assert_fixed(scene.camera, circle)

    scene.add_fixed_in_frame_mobjects(square)
    _capture(scene)
    _assert_fixed(scene.camera, square)
    scene.remove_fixed_in_frame_mobjects(child)
    scene.add_fixed_in_frame_mobjects(group)
    _capture(scene)
    _assert_fixed(scene.camera, group)

    scene.remove_fixed_in_frame_mobjects(group)
    _capture(scene)
    assert not scene.camera.fixed_in_frame_mobjects


def test_shared_child_can_be_unfixed_and_refixed(scene):
    shared = Square().shift((2, 1, 0))
    first = VGroup(shared, Circle().shift((-2, 1, 0)))
    second = VGroup(shared)
    scene.add_fixed_in_frame_mobjects(first, second)
    scene.remove_fixed_in_frame_mobjects(shared)
    _capture(scene)
    assert shared not in scene.camera.fixed_in_frame_mobjects

    scene.add_fixed_in_frame_mobjects(first)
    _capture(scene)
    _assert_fixed(scene.camera, shared)


def test_detached_unfixed_children_are_released(scene):
    child = Square().shift((2, 1, 0))
    group = VGroup(child, Circle())
    scene.add_fixed_in_frame_mobjects(group)
    scene.remove_fixed_in_frame_mobjects(child)
    child_ref = weakref.ref(child)
    group.remove(child)
    del child

    _capture(scene)
    gc.collect()

    assert child_ref() is None


def test_fixed_orientation_and_ordinary_projection_are_preserved(scene):
    square = Square().shift((2, 1, 1))
    scene.add(square)
    _capture(scene)
    np.testing.assert_allclose(
        scene.camera.transform_points_pre_display(square, square.points),
        scene.camera.project_points(square.points),
    )
    scene.add_fixed_orientation_mobjects(square)
    _capture(scene)
    oriented = scene.camera.transform_points_pre_display(square, square.points)
    np.testing.assert_allclose(
        oriented[1:] - oriented[0], square.points[1:] - square.points[0], atol=1e-8
    )

    scene.add_fixed_in_frame_mobjects(square)
    _capture(scene)
    _assert_fixed(scene.camera, square)
    scene.remove_fixed_in_frame_mobjects(square)
    _capture(scene)
    np.testing.assert_allclose(
        scene.camera.transform_points_pre_display(square, square.points), oriented
    )
