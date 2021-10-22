"""Utility functions for two- and three-dimensional vectors."""

__all__ = [
    "quaternion_mult",
    "quaternion_from_angle_axis",
    "angle_axis_from_quaternion",
    "quaternion_conjugate",
    "rotate_vector",
    "thick_diagonal",
    "rotation_matrix",
    "rotation_about_z",
    "z_to_vector",
    "angle_of_vector",
    "angle_between_vectors",
    "project_along_vector",
    "normalize",
    "get_unit_normal",
    "compass_directions",
    "regular_vertices",
    "complex_to_R3",
    "R3_to_complex",
    "complex_func_to_R3_func",
    "center_of_mass",
    "midpoint",
    "find_intersection",
    "line_intersection",
    "get_winding_number",
    "cross2d",
    "earclip_triangulation",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "perpendicular_bisector",
]


import itertools as it
import math
from functools import reduce
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from mapbox_earcut import triangulate_float32 as earcut

from .. import config
from ..constants import DOWN, OUT, PI, RIGHT, TAU
from ..utils.iterables import adjacent_pairs


def norm_squared(v: float) -> float:
    return np.dot(v, v)


# Quaternions
# TODO, implement quaternion type


def quaternion_mult(
    *quats: Sequence[float],
) -> Union[np.ndarray, List[Union[float, np.ndarray]]]:
    """Gets the Hamilton product of the quaternions provided.
    For more information, check `this Wikipedia page
    <https://en.wikipedia.org/wiki/Quaternion>`__.

    Returns
    -------
    Union[np.ndarray, List[Union[float, np.ndarray]]]
        Returns a list of product of two quaternions.
    """
    if config.renderer == "opengl":
        if len(quats) == 0:
            return [1, 0, 0, 0]
        result = quats[0]
        for next_quat in quats[1:]:
            w1, x1, y1, z1 = result
            w2, x2, y2, z2 = next_quat
            result = [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            ]
        return result
    else:
        q1 = quats[0]
        q2 = quats[1]

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            ],
        )


def quaternion_from_angle_axis(
    angle: float,
    axis: np.ndarray,
    axis_normalized: bool = False,
) -> List[float]:
    """Gets a quaternion from an angle and an axis.
    For more information, check `this Wikipedia page
    <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles>`__.

    Parameters
    ----------
    angle
        The angle for the quaternion.
    axis
        The axis for the quaternion
    axis_normalized : bool, optional
        Checks whether the axis is normalized, by default False

    Returns
    -------
    List[float]
        Gives back a quaternion from the angle and axis
    """
    if config.renderer == "opengl":
        if not axis_normalized:
            axis = normalize(axis)
        return [math.cos(angle / 2), *(math.sin(angle / 2) * axis)]
    else:
        return np.append(np.cos(angle / 2), np.sin(angle / 2) * normalize(axis))


def angle_axis_from_quaternion(quaternion: Sequence[float]) -> Sequence[float]:
    """Gets angle and axis from a quaternion.

    Parameters
    ----------
    quaternion
        The quaternion from which we get the angle and axis.

    Returns
    -------
    Sequence[float]
        Gives the angle and axis
    """
    axis = normalize(quaternion[1:], fall_back=np.array([1, 0, 0]))
    angle = 2 * np.arccos(quaternion[0])
    if angle > TAU / 2:
        angle = TAU - angle
    return angle, axis


def quaternion_conjugate(quaternion: Sequence[float]) -> np.ndarray:
    """Used for finding the conjugate of the quaternion

    Parameters
    ----------
    quaternion
        The quaternion for which you want to find the conjugate for.

    Returns
    -------
    np.ndarray
        The conjugate of the quaternion.
    """
    result = np.array(quaternion)
    result[1:] *= -1
    return result


def rotate_vector(vector: np.ndarray, angle: int, axis: np.ndarray = OUT) -> np.ndarray:
    """Function for rotating a vector.

    Parameters
    ----------
    vector
        The vector to be rotated.
    angle
        The angle to be rotated by.
    axis
        The axis to be rotated, by default OUT

    Returns
    -------
    np.ndarray
        The rotated vector with provided angle and axis.

    Raises
    ------
    ValueError
        If vector is not of dimension 2 or 3.
    """

    if len(vector) == 2:
        # Use complex numbers...because why not
        z = complex(*vector) * np.exp(complex(0, angle))
        return np.array([z.real, z.imag])
    elif len(vector) == 3:
        # Use quaternions...because why not
        quat = quaternion_from_angle_axis(angle, axis)
        quat_inv = quaternion_conjugate(quat)
        product = reduce(quaternion_mult, [quat, np.append(0, vector), quat_inv])
        return product[1:]
    else:
        raise ValueError("vector must be of dimension 2 or 3")


def thick_diagonal(dim: int, thickness=2) -> np.ndarray:
    row_indices = np.arange(dim).repeat(dim).reshape((dim, dim))
    col_indices = np.transpose(row_indices)
    return (np.abs(row_indices - col_indices) < thickness).astype("uint8")


def rotation_matrix_transpose_from_quaternion(quat: np.ndarray) -> List[np.ndarray]:
    """Converts the quaternion, quat, to an equivalent rotation matrix representation.
    For more information, check `this page
    <https://in.mathworks.com/help/driving/ref/quaternion.rotmat.html>`_.

    Parameters
    ----------
    quat
        The quaternion which is to be converted.

    Returns
    -------
    List[np.ndarray]
        Gives back the Rotation matrix representation, returned as a 3-by-3
        matrix or 3-by-3-by-N multidimensional array.
    """
    quat_inv = quaternion_conjugate(quat)
    return [
        quaternion_mult(quat, [0, *basis], quat_inv)[1:]
        for basis in [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ]


def rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    return np.transpose(rotation_matrix_transpose_from_quaternion(quat))


def rotation_matrix_transpose(angle: float, axis: np.ndarray) -> np.ndarray:
    if axis[0] == 0 and axis[1] == 0:
        # axis = [0, 0, z] case is common enough it's worth
        # having a shortcut
        sgn = 1 if axis[2] > 0 else -1
        cos_a = math.cos(angle)
        sin_a = math.sin(angle) * sgn
        return [
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0],
            [0, 0, 1],
        ]
    quat = quaternion_from_angle_axis(angle, axis)
    return rotation_matrix_transpose_from_quaternion(quat)


def rotation_matrix(
    angle: float,
    axis: np.ndarray,
    homogeneous: bool = False,
) -> np.ndarray:
    """
    Rotation in R^3 about a specified axis of rotation.
    """
    about_z = rotation_about_z(angle)
    z_to_axis = z_to_vector(axis)
    axis_to_z = np.linalg.inv(z_to_axis)
    inhomogeneous_rotation_matrix = reduce(np.dot, [z_to_axis, about_z, axis_to_z])
    if not homogeneous:
        return inhomogeneous_rotation_matrix
    else:
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = inhomogeneous_rotation_matrix
        return rotation_matrix


def rotation_about_z(angle: float) -> List[List[float]]:
    """Returns a rotation matrix for a given angle.

    Parameters
    ----------
    angle : float
        Angle for the rotation matrix.

    Returns
    -------
    List[float]
        Gives back the rotated matrix.
    """
    return [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]


def z_to_vector(vector: np.ndarray) -> np.ndarray:
    """
    Returns some matrix in SO(3) which takes the z-axis to the
    (normalized) vector provided as an argument
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.identity(3)
    v = np.array(vector) / norm
    phi = np.arccos(v[2])
    if any(v[:2]):
        # projection of vector to unit circle
        axis_proj = v[:2] / np.linalg.norm(v[:2])
        theta = np.arccos(axis_proj[0])
        if axis_proj[1] < 0:
            theta = -theta
    else:
        theta = 0
    phi_down = np.array(
        [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]],
    )
    return np.dot(rotation_about_z(theta), phi_down)


def angle_of_vector(vector: Sequence[float]) -> float:
    """Returns polar coordinate theta when vector is projected on xy plane.

    Parameters
    ----------
    vector
        The vector to find the angle for.

    Returns
    -------
    float
        The angle of the vector projected.
    """
    if config.renderer == "opengl":
        return np.angle(complex(*vector[:2]))
    else:
        z = complex(*vector[:2])
        if z == 0:
            return 0
        return np.angle(complex(*vector[:2]))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns the angle between two vectors.
    This angle will always be between 0 and pi

    Parameters
    ----------
    v1
        The first vector.
    v2
        The second vector.

    Returns
    -------
    np.ndarray
        The angle between the vectors.
    """

    return 2 * np.arctan2(
        np.linalg.norm(normalize(v1) - normalize(v2)),
        np.linalg.norm(normalize(v1) + normalize(v2)),
    )


def project_along_vector(point: float, vector: np.ndarray) -> np.ndarray:
    """Projects a vector along a point.

    Parameters
    ----------
    point
        The point to be project from.
    vector
        The vector which has to projected.

    Returns
    -------
    np.ndarray
        A dot product of the point and vector.
    """
    matrix = np.identity(3) - np.outer(vector, vector)
    return np.dot(point, matrix.T)


def normalize(vect: Union[np.ndarray, Tuple[float]], fall_back=None) -> np.ndarray:
    norm = np.linalg.norm(vect)
    if norm > 0:
        return np.array(vect) / norm
    else:
        if fall_back is not None:
            return fall_back
        else:
            return np.zeros(len(vect))


def normalize_along_axis(array: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Normalizes an array with the provided axis.

    Parameters
    ----------
    array
        The array which has to be normalized.
    axis
        The axis to be normalized to.

    Returns
    -------
    np.ndarray
        Array which has been normalized according to the axis.
    """
    norms = np.sqrt((array * array).sum(axis))
    norms[norms == 0] = 1
    buffed_norms = np.repeat(norms, array.shape[axis]).reshape(array.shape)
    array /= buffed_norms
    return array


def get_unit_normal(v1: np.ndarray, v2: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Gets the unit normal of the vectors.

    Parameters
    ----------
    v1
        The first vector.
    v2
        The second vector
    tol
        [description], by default 1e-6

    Returns
    -------
    np.ndarray
        The normal of the two vectors.
    """
    if config.renderer == "opengl":
        v1 = normalize(v1)
        v2 = normalize(v2)
        cp = np.cross(v1, v2)
        cp_norm = np.linalg.norm(cp)
        if cp_norm < tol:
            # Vectors align, so find a normal to them in the plane shared with the z-axis
            new_cp = np.cross(np.cross(v1, OUT), v1)
            new_cp_norm = np.linalg.norm(new_cp)
            if new_cp_norm < tol:
                return DOWN
            return new_cp / new_cp_norm
        return cp / cp_norm
    else:
        return normalize(np.cross(v1, v2))


###


def compass_directions(n: int = 4, start_vect: np.ndarray = RIGHT) -> np.ndarray:
    """Finds the cardinal directions using tau.

    Parameters
    ----------
    n
        The amount to be rotated, by default 4
    start_vect
        The direction for the angle to start with, by default RIGHT

    Returns
    -------
    np.ndarray
        The angle which has been rotated.
    """
    angle = TAU / n
    return np.array([rotate_vector(start_vect, k * angle) for k in range(n)])


def regular_vertices(
    n: int, *, radius: float = 1, start_angle: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """Generates regularly spaced vertices around a circle centered at the origin.

    Parameters
    ----------
    n
        The number of vertices
    radius
        The radius of the circle that the vertices are placed on.
    start_angle
        The angle the vertices start at.

        If unspecified, for even ``n`` values, ``0`` will be used.
        For odd ``n`` values, 90 degrees is used.

    Returns
    -------
    vertices : :class:`numpy.ndarray`
        The regularly spaced vertices.
    start_angle : :class:`float`
        The angle the vertices start at.
    """

    if start_angle is None:
        if n % 2 == 0:
            start_angle = 0
        else:
            start_angle = TAU / 4

    start_vector = rotate_vector(RIGHT * radius, start_angle)
    vertices = compass_directions(n, start_vector)

    return vertices, start_angle


def complex_to_R3(complex_num: complex) -> np.ndarray:
    return np.array((complex_num.real, complex_num.imag, 0))


def R3_to_complex(point: Sequence[float]) -> np.ndarray:
    return complex(*point[:2])


def complex_func_to_R3_func(complex_func):
    return lambda p: complex_to_R3(complex_func(R3_to_complex(p)))


def center_of_mass(points: Sequence[float]) -> np.ndarray:
    """Gets the center of mass of the points in space.

    Parameters
    ----------
    points
        The points to find the center of mass from.

    Returns
    -------
    np.ndarray
        The center of mass of the points.
    """
    points = [np.array(point).astype("float") for point in points]
    return sum(points) / len(points)


def midpoint(
    point1: Sequence[float],
    point2: Sequence[float],
) -> Union[float, np.ndarray]:
    """Gets the midpoint of two points.

    Parameters
    ----------
    point1
        The first point.
    point2
        The second point.

    Returns
    -------
    Union[float, np.ndarray]
        The midpoint of the points
    """
    return center_of_mass([point1, point2])


def line_intersection(line1: Sequence[float], line2: Sequence[float]) -> np.ndarray:
    """Returns intersection point of two lines, each defined with
    a pair of vectors determining the end points.

    Parameters
    ----------
    line1
        The first line.
    line2
        The second line.

    Returns
    -------
    np.ndarray
        The intersection points of the two lines which are intersecting.

    Raises
    ------
    ValueError
        Error is produced if the two lines don't intersect with each other
    """
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        raise ValueError("Lines do not intersect")
    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return np.array([x, y, 0])


def find_intersection(p0, v0, p1, v1, threshold=1e-5) -> np.ndarray:
    """
    Return the intersection of a line passing through p0 in direction v0
    with one passing through p1 in direction v1.  (Or array of intersections
    from arrays of such points/directions).
    For 3d values, it returns the point on the ray p0 + v0 * t closest to the
    ray p1 + v1 * t
    """
    p0 = np.array(p0, ndmin=2)
    v0 = np.array(v0, ndmin=2)
    p1 = np.array(p1, ndmin=2)
    v1 = np.array(v1, ndmin=2)
    m, n = np.shape(p0)
    assert n in [2, 3]

    numerator = np.cross(v1, p1 - p0)
    denominator = np.cross(v1, v0)
    if n == 3:
        d = len(np.shape(numerator))
        new_numerator = np.multiply(numerator, numerator).sum(d - 1)
        new_denominator = np.multiply(denominator, numerator).sum(d - 1)
        numerator, denominator = new_numerator, new_denominator

    denominator[abs(denominator) < threshold] = np.inf  # So that ratio goes to 0 there
    ratio = numerator / denominator
    ratio = np.repeat(ratio, n).reshape((m, n))
    return p0 + ratio * v0


def get_winding_number(points: Sequence[float]) -> float:
    total_angle = 0
    for p1, p2 in adjacent_pairs(points):
        d_angle = angle_of_vector(p2) - angle_of_vector(p1)
        d_angle = ((d_angle + PI) % TAU) - PI
        total_angle += d_angle
    return total_angle / TAU


def shoelace(x_y: np.ndarray) -> float:
    """2D implementation of the shoelace formula.

    Returns
    -------
    :class:`float`
        Returns signed area.
    """
    x = x_y[:, 0]
    y = x_y[:, 1]
    area = 0.5 * np.array(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def shoelace_direction(x_y: np.ndarray) -> str:
    """
    Uses the area determined by the shoelace method to determine whether
    the input set of points is directed clockwise or counterclockwise.

    Returns
    -------
    :class:`str`
        Either ``"CW"`` or ``"CCW"``.
    """
    area = shoelace(x_y)
    return "CW" if area > 0 else "CCW"


def cross2d(a, b):
    if len(a.shape) == 2:
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    else:
        return a[0] * b[1] - b[0] * a[1]


def earclip_triangulation(verts: np.ndarray, ring_ends: list) -> list:
    """Returns a list of indices giving a triangulation
    of a polygon, potentially with holes.

    Parameters
    ----------
    verts
        verts is a numpy array of points.
    ring_ends
        ring_ends is a list of indices indicating where
    the ends of new paths are.

    Returns
    -------
    list
        A list of indices giving a triangulation of a polygon.
    """
    # First, connect all the rings so that the polygon
    # with holes is instead treated as a (very convex)
    # polygon with one edge.  Do this by drawing connections
    # between rings close to each other
    rings = [list(range(e0, e1)) for e0, e1 in zip([0, *ring_ends], ring_ends)]
    attached_rings = rings[:1]
    detached_rings = rings[1:]
    loop_connections = {}

    while detached_rings:
        i_range, j_range = (
            list(
                filter(
                    # Ignore indices that are already being
                    # used to draw some connection
                    lambda i: i not in loop_connections,
                    it.chain(*ring_group),
                ),
            )
            for ring_group in (attached_rings, detached_rings)
        )

        # Closest point on the attached rings to an estimated midpoint
        # of the detached rings
        tmp_j_vert = midpoint(verts[j_range[0]], verts[j_range[len(j_range) // 2]])
        i = min(i_range, key=lambda i: norm_squared(verts[i] - tmp_j_vert))
        # Closest point of the detached rings to the aforementioned
        # point of the attached rings
        j = min(j_range, key=lambda j: norm_squared(verts[i] - verts[j]))
        # Recalculate i based on new j
        i = min(i_range, key=lambda i: norm_squared(verts[i] - verts[j]))

        # Remember to connect the polygon at these points
        loop_connections[i] = j
        loop_connections[j] = i

        # Move the ring which j belongs to from the
        # attached list to the detached list
        new_ring = next(filter(lambda ring: ring[0] <= j < ring[-1], detached_rings))
        detached_rings.remove(new_ring)
        attached_rings.append(new_ring)

    # Setup linked list
    after = []
    end0 = 0
    for end1 in ring_ends:
        after.extend(range(end0 + 1, end1))
        after.append(end0)
        end0 = end1

    # Find an ordering of indices walking around the polygon
    indices = []
    i = 0
    for _ in range(len(verts) + len(ring_ends) - 1):
        # starting = False
        if i in loop_connections:
            j = loop_connections[i]
            indices.extend([i, j])
            i = after[j]
        else:
            indices.append(i)
            i = after[i]
        if i == 0:
            break

    meta_indices = earcut(verts[indices, :2], [len(indices)])
    return [indices[mi] for mi in meta_indices]


def cartesian_to_spherical(vec: Sequence[float]) -> np.ndarray:
    """Returns an array of numbers corresponding to each
    polar coordinate value (distance, phi, theta).

    Parameters
    ----------
    vec
        A numpy array ``[x, y, z]``.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return 0, 0, 0
    r = norm
    phi = np.arccos(vec[2] / r)
    theta = np.arctan2(vec[1], vec[0])
    return np.array([r, phi, theta])


def spherical_to_cartesian(spherical: Sequence[float]) -> np.ndarray:
    """Returns a numpy array ``[x, y, z]`` based on the spherical
    coordinates given.

    Parameters
    ----------
    spherical
        A list of three floats that correspond to the following:

        r - The distance between the point and the origin.

        theta - The azimuthal angle of the point to the positive x-axis.

        phi - The vertical angle of the point to the positive z-axis.
    """
    r, theta, phi = spherical
    return np.array(
        [
            r * np.cos(theta) * np.sin(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(phi),
        ],
    )


def perpendicular_bisector(
    line: Sequence[np.ndarray],
    norm_vector=OUT,
) -> Sequence[np.ndarray]:
    """Returns a list of two points that correspond
    to the ends of the perpendicular bisector of the
    two points given.

    Parameters
    ----------
    line
        a list of two numpy array points (corresponding
        to the ends of a line).
    norm_vector
        the vector perpendicular to both the line given
        and the perpendicular bisector.

    Returns
    -------
    list
        A list of two numpy array points that correspond
        to the ends of the perpendicular bisector
    """
    p1 = line[0]
    p2 = line[1]
    direction = np.cross(p1 - p2, norm_vector)
    m = midpoint(p1, p2)
    return [m + direction, m - direction]
