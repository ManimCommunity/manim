from __future__ import annotations

import numpy as np
import numpy.linalg as linalg

from .. import config

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


def matrix_to_shader_input(matrix):
    return tuple(matrix.T.ravel())


def orthographic_projection_matrix(
    width=None,
    height=None,
    near=1,
    far=depth + 1,
    format=True,
):
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
    if format:
        return matrix_to_shader_input(projection_matrix)
    else:
        return projection_matrix


def perspective_projection_matrix(width=None, height=None, near=2, far=50, format=True):
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
    if format:
        return matrix_to_shader_input(projection_matrix)
    else:
        return projection_matrix


def translation_matrix(x=0, y=0, z=0):
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
    )


def x_rotation_matrix(x=0):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x), -np.sin(x), 0],
            [0, np.sin(x), np.cos(x), 0],
            [0, 0, 0, 1],
        ],
    )


def y_rotation_matrix(y=0):
    return np.array(
        [
            [np.cos(y), 0, np.sin(y), 0],
            [0, 1, 0, 0],
            [-np.sin(y), 0, np.cos(y), 0],
            [0, 0, 0, 1],
        ],
    )


def z_rotation_matrix(z=0):
    return np.array(
        [
            [np.cos(z), -np.sin(z), 0, 0],
            [np.sin(z), np.cos(z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )


# TODO: When rotating around the x axis, rotation eventually stops.
def rotate_in_place_matrix(initial_position, x=0, y=0, z=0):
    return np.matmul(
        translation_matrix(*-initial_position),
        np.matmul(
            rotation_matrix(x, y, z),
            translation_matrix(*initial_position),
        ),
    )


def rotation_matrix(x=0, y=0, z=0):
    return np.matmul(
        np.matmul(x_rotation_matrix(x), y_rotation_matrix(y)),
        z_rotation_matrix(z),
    )


def scale_matrix(scale_factor=1):
    return np.array(
        [
            [scale_factor, 0, 0, 0],
            [0, scale_factor, 0, 0],
            [0, 0, scale_factor, 0],
            [0, 0, 0, 1],
        ],
    )


def view_matrix(
    translation=None,
    x_rotation=0,
    y_rotation=0,
    z_rotation=0,
):
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
