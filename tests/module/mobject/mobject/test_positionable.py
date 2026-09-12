from __future__ import annotations

import numpy as np

from manim import DOWN, LEFT, ORIGIN, RIGHT, TAU, UP
from manim.mobject.abstract.positionable import Positionable


def test_get_all_points() -> None:
    p = Positionable().set_points([(0, 0, 0), (1, 1, 0)])
    np.testing.assert_allclose(p.get_all_points(), [(0, 0, 0), (1, 1, 0)])


def test_set_points() -> None:
    p = Positionable().set_points([(1, 2, 3)])
    np.testing.assert_allclose(p.points, [(1, 2, 3)])


def test_match_points() -> None:
    p = Positionable().set_points([(5, 5, 5), (6, 6, 6)])
    other = Positionable().set_points([(0, 0, 0), (1, 1, 1)])
    p.match_points(other)
    np.testing.assert_allclose(p.points, [(0, 0, 0), (1, 1, 1)])


def test_reset_points() -> None:
    p = Positionable().set_points([(1, 2, 3)])
    p.reset_points()
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


def test_repeat() -> None:
    p = Positionable().set_points([(1, 2, 3), (4, 5, 6)])
    p.repeat(3)

    expected = [(1, 2, 3), (4, 5, 6), (1, 2, 3), (4, 5, 6), (1, 2, 3), (4, 5, 6)]
    np.testing.assert_allclose(p.points, expected)


def test_num_points() -> None:
    p = Positionable()
    assert p.get_num_points() == 0

    p = Positionable().set_points([(0, 0, 0), (1, 1, 1)])
    assert p.get_num_points() == 2


def test_has_no_points() -> None:
    assert Positionable().has_no_points()
    assert not Positionable().set_points([(0, 0, 0)]).has_no_points()


def test_has_points() -> None:
    assert not Positionable().set_points([]).has_points()
    assert Positionable().set_points([(0, 0, 0)]).has_points()


def test_apply_to_family() -> None:
    p = Positionable().set_points([(0, 0, 0), (1, 1, 1), (1, 2, 3)])
    p.apply_to_family(lambda mob: mob.points.__imul__(2))
    np.testing.assert_allclose(p.points, [(0, 0, 0), (2, 2, 2), (2, 4, 6)])


def test_apply_to_family_should_skip() -> None:
    p = Positionable()
    p.apply_to_family(
        lambda mob: mob.set_points([(1, 2, 3)]),
        should_skip=lambda _: True,
    )
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))

    p = Positionable()
    p.apply_to_family(
        lambda mob: mob.set_points([(1, 2, 3)]),
        should_skip=lambda _: False,
    )
    np.testing.assert_allclose(p.points, [(1, 2, 3)])


def test_apply_function() -> None:
    p = Positionable().set_points([(1, 2, 3)])
    p.apply_function(lambda point: point + 1)
    np.testing.assert_allclose(p.points, [(2, 3, 4)])


def test_apply_function_about_point() -> None:
    p = Positionable().set_points([(2, 0, 0)])
    p.apply_function(lambda point: 2 * point, about_point=(1, 1, 1))
    np.testing.assert_allclose(p.points, [(3, -1, -1)])


def test_apply_complex_function() -> None:
    p = Positionable().set_points([(1, 0, 0)])
    p.apply_complex_function(lambda z: z * 1j)
    np.testing.assert_allclose(p.points, [(0, 1, 0)], atol=1e-7)


def test_translate() -> None:
    p = Positionable().set_points([(0, 0, 0)])
    p.translate(RIGHT)
    np.testing.assert_allclose(p.points, [(1, 0, 0)])


def test_scale() -> None:
    p = Positionable().set_points([(-1, 0, 0), (1, 0, 0)])
    p.scale(2)
    np.testing.assert_allclose(p.points, [(-2, 0, 0), (2, 0, 0)])


def test_scale_about_point() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 0, 0)])
    p.scale(2, about_point=(2, 0, 0))
    np.testing.assert_allclose(p.points, [(-2, 0, 0), (2, 0, 0)])


def test_scale_about_edge() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 0, 0)])
    p.scale(2, about_edge=RIGHT)
    np.testing.assert_allclose(p.points, [(-2, 0, 0), (2, 0, 0)])


def test_stretch() -> None:
    p = Positionable().set_points([(-1, -1, 0), (1, 1, 0)])
    p.stretch(2, dim=0)
    np.testing.assert_allclose(p.points, [(-2, -1, 0), (2, 1, 0)])


def test_rotate() -> None:
    p = Positionable().set_points([(1, 0, 0)])
    p.rotate(TAU / 4, about_point=ORIGIN)
    np.testing.assert_allclose(p.points, [(0, 1, 0)], atol=1e-7)


def test_rotate_axis() -> None:
    p = Positionable().set_points([(0, 0, 1)])
    p.rotate(TAU / 4, axis=RIGHT, about_point=ORIGIN)
    np.testing.assert_allclose(p.points, [(0, -1, 0)], atol=1e-7)


def test_apply_matrix() -> None:
    p = Positionable().set_points([(1, 0, 0)])
    p.apply_matrix([(0, -1), (1, 0)])
    np.testing.assert_allclose(p.points, [(0, 1, 0)], atol=1e-7)


def test_apply_matrix_about_point() -> None:
    p = Positionable().set_points([(2, 0, 0)])
    p.apply_matrix([(0, -1), (1, 0)], about_point=(2, 0, 0))
    np.testing.assert_allclose(p.points, [(2, 0, 0)], atol=1e-7)


def test_get_anchor() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 4, 0)])
    np.testing.assert_allclose(p.get_anchor(RIGHT), [2, 2, 0])


def test_get_anchor_directions() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 4, 6)])
    np.testing.assert_allclose(p.get_anchor(LEFT), [0, 2, 3])
    np.testing.assert_allclose(p.get_anchor(DOWN), [1, 0, 3])


def test_set_anchor() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    p.set_anchor((5, 5, 0))
    np.testing.assert_allclose(p.get_center(), [5, 5, 0])


def test_set_anchor_direction() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    p.set_anchor((5, 5, 0), direction=UP)
    np.testing.assert_allclose(p.get_top(), [5, 5, 0])


def test_match_anchor() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    other = Positionable().set_points([(10, 10, 0)])
    p.match_anchor(other)
    np.testing.assert_allclose(p.get_center(), [10, 10, 0])


def test_get_coordinate() -> None:
    p = Positionable().set_points([(0, 0, 0), (4, 2, 0)])
    assert p.get_coordinate(dim=0, direction=RIGHT) == 4


def test_get_coordinate_directions() -> None:
    p = Positionable().set_points([(0, 0, 0), (4, 2, 0)])
    assert p.get_coordinate(dim=0, direction=ORIGIN) == 2
    assert p.get_coordinate(dim=1, direction=UP) == 2


def test_set_coordinate() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 0, 0)])
    p.set_coordinate(10, dim=0, direction=RIGHT)
    np.testing.assert_allclose(p.points, [(8, 0, 0), (10, 0, 0)])


def test_match_coordinate() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 0, 0)])
    other = Positionable().set_points([(9, 0, 0)])
    p.match_coordinate(other, dim=0, direction=RIGHT)
    np.testing.assert_allclose(p.points, [(7, 0, 0), (9, 0, 0)])


def test_align_on_border() -> None:
    from manim._config import config

    p = Positionable().set_points([(0, 0, 0)])
    p.align_on_border(UP, buff=0)
    np.testing.assert_allclose(p.get_top()[1], config.frame_y_radius)


def test_align_on_border_buff() -> None:
    from manim._config import config

    p = Positionable().set_points([(0, 0, 0)])
    p.align_on_border(UP, buff=1)
    np.testing.assert_allclose(p.get_top()[1], config.frame_y_radius - 1)


def test_align_to() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    p.align_to((5, 5, 0), direction=UP)
    np.testing.assert_allclose(p.get_top(), [1, 5, 0])


def test_align_to_other() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    other = Positionable().set_points([(5, 5, 0)])
    p.align_to(other, direction=UP)
    np.testing.assert_allclose(p.get_top(), [1, 5, 0])


def test_next_to() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    p.next_to((0, 0, 0), direction=RIGHT, buff=0)
    np.testing.assert_allclose(p.get_left(), [0, 0, 0])


def test_next_to_aligned_edge() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    p.next_to((0, 0, 0), direction=RIGHT, aligned_edge=UP, buff=0)
    np.testing.assert_allclose(p.get_top(), [1, 0, 0])


def test_next_to_buff() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    p.next_to((0, 0, 0), direction=RIGHT, buff=1)
    np.testing.assert_allclose(p.get_left(), [1, 0, 0])


def test_apply_function_to_position() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 2, 0)])
    p.apply_function_to_position(lambda point: point + RIGHT)
    np.testing.assert_allclose(p.get_center(), [2, 1, 0])


def test_is_off_screen() -> None:
    p = Positionable().set_points([(0, 0, 0)])
    assert not p.is_off_screen()

    p = Positionable().set_points([(1000, 0, 0)])
    assert p.is_off_screen()


def test_get_center_of_mass() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 0, 0), (4, 0, 0)])
    np.testing.assert_allclose(p.get_center_of_mass(), [2, 0, 0])


def test_get_boundary_point() -> None:
    p = Positionable().set_points([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    np.testing.assert_allclose(p.get_boundary_point(RIGHT), [1, 0, 0])


def test_get_dim_size() -> None:
    p = Positionable().set_points([(0, 0, 0), (3, 1, 0)])
    assert p.get_dim_size(dim=0) == 3


def test_set_dim_size() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 0, 0)])
    p.set_dim_size(4, dim=0)
    assert p.get_dim_size(dim=0) == 4


def test_set_dim_size_stretch() -> None:
    p = Positionable().set_points([(-1, -1, 0), (1, 1, 0)])
    p.set_dim_size(4, dim=0, stretch=True)
    assert p.get_width() == 4
    assert p.get_height() == 2


def test_scale_to_fit_dim() -> None:
    p = Positionable().set_points([(0, 0, 0), (2, 0, 0)])
    p.scale_to_fit_dim(6, dim=0)
    assert p.get_width() == 6
    assert p.get_height() == 0


def test_stretch_to_fit_dim() -> None:
    p = Positionable().set_points([(-1, -1, 0), (1, 1, 0)])
    p.stretch_to_fit_dim(4, dim=0)
    assert p.get_width() == 4
    assert p.get_height() == 2
