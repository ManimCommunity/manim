import numpy as np

from manim.constants import DEGREES, DOWN, IN, LEFT, ORIGIN, OUT, RIGHT, UP
from manim.mobject.abstract.positionable import Positionable


def test_translate() -> None:
    p = Positionable().set_points([(1, 2, 3)])
    p.translate((-40, -50, -60))
    np.testing.assert_allclose(p.points, [(1 - 40, 2 - 50, 3 - 60)])

    p = Positionable().set_points(
        [
            (-1, -1, -1),
            (-1, -1, +1),
            (-1, +1, -1),
            (-1, +1, +1),
            (+1, -1, -1),
            (+1, -1, +1),
            (+1, +1, -1),
            (+1, +1, +1),
        ]
    )
    p.translate((2, -3, 4))
    np.testing.assert_allclose(
        p.points,
        [
            (-1 + 2, -1 - 3, -1 + 4),
            (-1 + 2, -1 - 3, +1 + 4),
            (-1 + 2, +1 - 3, -1 + 4),
            (-1 + 2, +1 - 3, +1 + 4),
            (+1 + 2, -1 - 3, -1 + 4),
            (+1 + 2, -1 - 3, +1 + 4),
            (+1 + 2, +1 - 3, -1 + 4),
            (+1 + 2, +1 - 3, +1 + 4),
        ],
    )


def test_rotate() -> None:
    p = Positionable().set_points([(1, 1, 1)])
    p.rotate(90 * DEGREES, axis=(1, 0, 0), about_point=ORIGIN)
    np.testing.assert_allclose(p.points, [(1, -1, 1)])

    p = Positionable().set_points([(1, 1, 1)])
    p.rotate(90 * DEGREES, axis=(0, 1, 0), about_point=ORIGIN)
    np.testing.assert_allclose(p.points, [(1, 1, -1)])

    p = Positionable().set_points([(1, 1, 1)])
    p.rotate(90 * DEGREES, axis=(0, 0, 1), about_point=ORIGIN)
    np.testing.assert_allclose(p.points, [(-1, 1, 1)])

    p = Positionable().set_points(
        [
            (0, 0, 0),
            (0, 0, 2),
            (0, 2, 0),
            (0, 2, 2),
            (2, 0, 0),
            (2, 0, 2),
            (2, 2, 0),
            (2, 2, 2),
        ]
    )
    p.rotate(90 * DEGREES, axis=(1, 0, 0))
    np.testing.assert_allclose(
        p.points,
        [
            (0, 2, 0),
            (0, 0, 0),
            (0, 2, 2),
            (0, 0, 2),
            (2, 2, 0),
            (2, 0, 0),
            (2, 2, 2),
            (2, 0, 2),
        ],
        atol=1e-8,
    )


def test_scale() -> None:
    p = Positionable().set_points(
        [
            (0, 0, 0),
            (0, 0, 2),
            (0, 2, 0),
            (0, 2, 2),
            (2, 0, 0),
            (2, 0, 2),
            (2, 2, 0),
            (2, 2, 2),
        ]
    )
    p.scale(2)
    np.testing.assert_allclose(
        p.points,
        [
            (-1, -1, -1),
            (-1, -1, +3),
            (-1, +3, -1),
            (-1, +3, +3),
            (+3, -1, -1),
            (+3, -1, +3),
            (+3, +3, -1),
            (+3, +3, +3),
        ],
    )


def test_stretch() -> None:
    p = Positionable().set_points(
        [
            (0, 0, 0),
            (0, 0, 2),
            (0, 2, 0),
            (0, 2, 2),
            (2, 0, 0),
            (2, 0, 2),
            (2, 2, 0),
            (2, 2, 2),
        ]
    )
    p.stretch(2, dim=1)
    np.testing.assert_allclose(
        p.points,
        [
            (0, -1, 0),
            (0, -1, 2),
            (0, +3, 0),
            (0, +3, 2),
            (2, -1, 0),
            (2, -1, 2),
            (2, +3, 0),
            (2, +3, 2),
        ],
    )


def test_get_position() -> None:
    p = Positionable()
    np.testing.assert_allclose(p.get_position(), (0, 0, 0))

    p = Positionable().set_points([(1, 2, 3)])
    np.testing.assert_allclose(p.get_position(), (1, 2, 3))

    p = Positionable().set_points(
        [
            (0, 0, 0),
            (0, 0, 2),
            (0, 2, 0),
            (0, 2, 2),
            (2, 0, 0),
            (2, 0, 2),
            (2, 2, 0),
            (2, 2, 2),
        ]
    )
    np.testing.assert_allclose(p.get_position((-1, -1, -1)), (0, 0, 0))
    np.testing.assert_allclose(p.get_position((-1, -1, +0)), (0, 0, 1))
    np.testing.assert_allclose(p.get_position((-1, -1, +1)), (0, 0, 2))
    np.testing.assert_allclose(p.get_position((-1, +0, -1)), (0, 1, 0))
    np.testing.assert_allclose(p.get_position((-1, +0, +0)), (0, 1, 1))
    np.testing.assert_allclose(p.get_position((-1, +0, +1)), (0, 1, 2))
    np.testing.assert_allclose(p.get_position((-1, +1, -1)), (0, 2, 0))
    np.testing.assert_allclose(p.get_position((-1, +1, +0)), (0, 2, 1))
    np.testing.assert_allclose(p.get_position((-1, +1, +1)), (0, 2, 2))
    np.testing.assert_allclose(p.get_position((+0, -1, -1)), (1, 0, 0))
    np.testing.assert_allclose(p.get_position((+0, -1, +0)), (1, 0, 1))
    np.testing.assert_allclose(p.get_position((+0, -1, +1)), (1, 0, 2))
    np.testing.assert_allclose(p.get_position((+0, +0, -1)), (1, 1, 0))
    np.testing.assert_allclose(p.get_position((+0, +0, +0)), (1, 1, 1))
    np.testing.assert_allclose(p.get_position((+0, +0, +1)), (1, 1, 2))
    np.testing.assert_allclose(p.get_position((+0, +1, -1)), (1, 2, 0))
    np.testing.assert_allclose(p.get_position((+0, +1, +0)), (1, 2, 1))
    np.testing.assert_allclose(p.get_position((+0, +1, +1)), (1, 2, 2))
    np.testing.assert_allclose(p.get_position((+1, -1, -1)), (2, 0, 0))
    np.testing.assert_allclose(p.get_position((+1, -1, +0)), (2, 0, 1))
    np.testing.assert_allclose(p.get_position((+1, -1, +1)), (2, 0, 2))
    np.testing.assert_allclose(p.get_position((+1, +0, -1)), (2, 1, 0))
    np.testing.assert_allclose(p.get_position((+1, +0, +0)), (2, 1, 1))
    np.testing.assert_allclose(p.get_position((+1, +0, +1)), (2, 1, 2))
    np.testing.assert_allclose(p.get_position((+1, +1, -1)), (2, 2, 0))
    np.testing.assert_allclose(p.get_position((+1, +1, +0)), (2, 2, 1))
    np.testing.assert_allclose(p.get_position((+1, +1, +1)), (2, 2, 2))


def test_set_position() -> None:
    p = Positionable().set_points([(0, 0, 0)])
    p.set_position((1, 2, 3))
    np.testing.assert_allclose(p.points, [(1, 2, 3)])

    p = Positionable().set_points(
        [
            (-1, -1, -1),
            (-1, -1, +1),
            (-1, +1, -1),
            (-1, +1, +1),
            (+1, -1, -1),
            (+1, -1, +1),
            (+1, +1, -1),
            (+1, +1, +1),
        ]
    )
    p.set_position((3, 2, 1))
    np.testing.assert_allclose(
        p.points,
        [
            (-1 + 3, -1 + 2, -1 + 1),
            (-1 + 3, -1 + 2, +1 + 1),
            (-1 + 3, +1 + 2, -1 + 1),
            (-1 + 3, +1 + 2, +1 + 1),
            (+1 + 3, -1 + 2, -1 + 1),
            (+1 + 3, -1 + 2, +1 + 1),
            (+1 + 3, +1 + 2, -1 + 1),
            (+1 + 3, +1 + 2, +1 + 1),
        ],
    )


def test_get_coordinate() -> None:
    p = Positionable()
    np.testing.assert_allclose(p.get_coordinate(0), 0)
    np.testing.assert_allclose(p.get_coordinate(1), 0)
    np.testing.assert_allclose(p.get_coordinate(2), 0)

    p = Positionable().set_points([(1, 2, 3)])
    np.testing.assert_allclose(p.get_coordinate(0), 1)
    np.testing.assert_allclose(p.get_coordinate(1), 2)
    np.testing.assert_allclose(p.get_coordinate(2), 3)

    p = Positionable().set_points(
        [
            (-1, -2, -3),
            (-1, -2, +3),
            (-1, +2, -3),
            (-1, +2, +3),
            (+1, -2, -3),
            (+1, -2, +3),
            (+1, +2, -3),
            (+1, +2, +3),
        ]
    )
    np.testing.assert_allclose(p.get_coordinate(0, LEFT), -1)
    np.testing.assert_allclose(p.get_coordinate(0, RIGHT), 1)
    np.testing.assert_allclose(p.get_coordinate(1, DOWN), -2)
    np.testing.assert_allclose(p.get_coordinate(1, UP), 2)
    np.testing.assert_allclose(p.get_coordinate(2, IN), -3)
    np.testing.assert_allclose(p.get_coordinate(2, OUT), 3)


def test_set_coordinate_x() -> None:
    p = Positionable().set_points(
        [
            (-1, -2, -3),
            (-1, -2, +3),
            (-1, +2, -3),
            (-1, +2, +3),
            (+1, -2, -3),
            (+1, -2, +3),
            (+1, +2, -3),
            (+1, +2, +3),
        ]
    )
    p.set_coordinate(coordinate=4, dim=0)
    np.testing.assert_allclose(
        p.points,
        [
            (-1 + 4, -2, -3),
            (-1 + 4, -2, +3),
            (-1 + 4, +2, -3),
            (-1 + 4, +2, +3),
            (+1 + 4, -2, -3),
            (+1 + 4, -2, +3),
            (+1 + 4, +2, -3),
            (+1 + 4, +2, +3),
        ],
    )


def test_set_coordinate_y() -> None:
    p = Positionable().set_points(
        [
            (-1, -2, -3),
            (-1, -2, +3),
            (-1, +2, -3),
            (-1, +2, +3),
            (+1, -2, -3),
            (+1, -2, +3),
            (+1, +2, -3),
            (+1, +2, +3),
        ]
    )
    p.set_coordinate(coordinate=5, dim=1)
    np.testing.assert_allclose(
        p.points,
        [
            (-1, -2 + 5, -3),
            (-1, -2 + 5, +3),
            (-1, +2 + 5, -3),
            (-1, +2 + 5, +3),
            (+1, -2 + 5, -3),
            (+1, -2 + 5, +3),
            (+1, +2 + 5, -3),
            (+1, +2 + 5, +3),
        ],
    )


def test_set_coordinate_z() -> None:
    p = Positionable().set_points(
        [
            (-1, -2, -3),
            (-1, -2, +3),
            (-1, +2, -3),
            (-1, +2, +3),
            (+1, -2, -3),
            (+1, -2, +3),
            (+1, +2, -3),
            (+1, +2, +3),
        ]
    )
    p.set_coordinate(coordinate=6, dim=2)
    np.testing.assert_allclose(
        p.points,
        [
            (-1, -2, -3 + 6),
            (-1, -2, +3 + 6),
            (-1, +2, -3 + 6),
            (-1, +2, +3 + 6),
            (+1, -2, -3 + 6),
            (+1, -2, +3 + 6),
            (+1, +2, -3 + 6),
            (+1, +2, +3 + 6),
        ],
    )
