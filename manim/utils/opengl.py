from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as linalg

from manim._config import config
from manim.typing import ManimFloat

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import TypeAlias

    from manim.typing import MatrixMN, Point3D


if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from manim.typing import MatrixMN


depth = 20

__all__ = [
    "matrix_to_shader_input",
    "orthographic_projection_matrix",
    "perspective_projection_matrix",
    "translation_matrix",
    "x_rotation_matrix",
    "y_rotation_matrix",
    "z_rotation_matrix",
    "rotate_in_place_matrix",
    "rotation_matrix",
    "scale_matrix",
    "view_matrix",
]

FlattenedMatrix4x4: TypeAlias = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


def matrix_to_shader_input(matrix: MatrixMN) -> FlattenedMatrix4x4:
    return tuple(matrix.T.ravel())


def orthographic_projection_matrix(
    width: float | None = None,
    height: float | None = None,
    near: float = 1,
    far: float = depth + 1,
    format_: bool = True,
) -> MatrixMN | FlattenedMatrix4x4:
    if width is None:
        width = config["frame_width"]
    if height is None:
        height = config["frame_height"]
    projection_matrix = np.array(
        [
            [2 / width, 0, 0, 0],
            [0, 2 / height, 0, 0],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1],
        ],
    )
    if format_:
        return matrix_to_shader_input(projection_matrix)
    else:
        return projection_matrix


def perspective_projection_matrix(
    width: float | None = None,
    height: float | None = None,
    near: float = 2,
    far: float = 50,
    format_: bool = True,
) -> MatrixMN | FlattenedMatrix4x4:
    if width is None:
        width = config["frame_width"] / 6
    if height is None:
        height = config["frame_height"] / 6
    projection_matrix = np.array(
        [
            [2 * near / width, 0, 0, 0],
            [0, 2 * near / height, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ],
    )
    if format_:
        return matrix_to_shader_input(projection_matrix)
    else:
        return projection_matrix


def translation_matrix(x: float = 0, y: float = 0, z: float = 0) -> MatrixMN:
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=ManimFloat,
    )


def x_rotation_matrix(x: float = 0) -> MatrixMN:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x), -np.sin(x), 0],
            [0, np.sin(x), np.cos(x), 0],
            [0, 0, 0, 1],
        ],
    )


def y_rotation_matrix(y: float = 0) -> MatrixMN:
    return np.array(
        [
            [np.cos(y), 0, np.sin(y), 0],
            [0, 1, 0, 0],
            [-np.sin(y), 0, np.cos(y), 0],
            [0, 0, 0, 1],
        ],
    )


def z_rotation_matrix(z: float = 0) -> MatrixMN:
    return np.array(
        [
            [np.cos(z), -np.sin(z), 0, 0],
            [np.sin(z), np.cos(z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )


# TODO: When rotating around the x axis, rotation eventually stops.
def rotate_in_place_matrix(
    initial_position: Point3D, x: float = 0, y: float = 0, z: float = 0
) -> MatrixMN:
    return np.matmul(
        translation_matrix(*-initial_position),
        np.matmul(
            rotation_matrix(x, y, z),
            translation_matrix(*initial_position),
        ),
    )


def rotation_matrix(x: float = 0, y: float = 0, z: float = 0) -> MatrixMN:
    return np.matmul(
        np.matmul(x_rotation_matrix(x), y_rotation_matrix(y)),
        z_rotation_matrix(z),
    )


def scale_matrix(scale_factor: float = 1) -> npt.NDArray:
    return np.array(
        [
            [scale_factor, 0, 0, 0],
            [0, scale_factor, 0, 0],
            [0, 0, scale_factor, 0],
            [0, 0, 0, 1],
        ],
        dtype=ManimFloat,
    )


def view_matrix(
    translation: Point3D | None = None,
    x_rotation: float = 0,
    y_rotation: float = 0,
    z_rotation: float = 0,
) -> MatrixMN:
    if translation is None:
        translation = np.array([0, 0, depth / 2 + 1])
    model_matrix = np.matmul(
        np.matmul(
            translation_matrix(*translation),
            rotation_matrix(x=x_rotation, y=y_rotation, z=z_rotation),
        ),
        scale_matrix(),
    )
    return tuple(linalg.inv(model_matrix).T.ravel())
