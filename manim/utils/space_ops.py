"""Utility functions for two- and three-dimensional vectors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from mapbox_earcut import triangulate_float32 as earcut
from scipy.spatial.transform import Rotation

from manim.constants import DOWN, OUT, PI, RIGHT, TAU, UP
from manim.utils.iterables import adjacent_pairs

if TYPE_CHECKING:
    import numpy.typing as npt

    from manim.typing import (
        ManimFloat,
        MatrixMN,
        Point2D_Array,
        Point3D,
        Point3DLike,
        Point3DLike_Array,
        PointND,
        PointNDLike_Array,
        Vector2D,
        Vector2D_Array,
        Vector3D,
        Vector3DLike,
        Vector3DLike_Array,
    )

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
    "shoelace",
    "shoelace_direction",
    "cross2d",
    "earclip_triangulation",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "perpendicular_bisector",
]


def norm_squared(v: float) -> float:
    val: float = np.dot(v, v)
    return val


def cross(v1: Vector3DLike, v2: Vector3DLike) -> Vector3D:
    return np.array(
        [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]
    )


# Quaternions
# TODO, implement quaternion type


def quaternion_mult(
    *quats: Sequence[float],
) -> np.ndarray | list[float | np.ndarray]:
    """Gets the Hamilton product of the quaternions provided.
    For more information, check `this Wikipedia page
    <https://en.wikipedia.org/wiki/Quaternion>`__.

    Returns
    -------
    Union[np.ndarray, List[Union[float, np.ndarray]]]
        Returns a list of product of two quaternions.
    """
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


def quaternion_from_angle_axis(
    angle: float,
    axis: np.ndarray,
    axis_normalized: bool = False,
) -> list[float]:
    """Gets a quaternion from an angle and an axis.
    For more information, check `this Wikipedia page
    <https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles>`__.

    Parameters
    ----------
    angle
        The angle for the quaternion.
    axis
        The axis for the quaternion
    axis_normalized
        Checks whether the axis is normalized, by default False

    Returns
    -------
    list[float]
        Gives back a quaternion from the angle and axis
    """
    if not axis_normalized:
        axis = normalize(axis)
    return [np.cos(angle / 2), *(np.sin(angle / 2) * axis)]


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


def rotate_vector(
    vector: Vector3DLike, angle: float, axis: Vector3DLike = OUT
) -> Vector3D:
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
    if len(vector) > 3:
        raise ValueError("Vector must have the correct dimensions.")
    if len(vector) == 2:
        vector = np.append(vector, 0)
    return rotation_matrix(angle, axis) @ vector


def thick_diagonal(dim: int, thickness: int = 2) -> MatrixMN:
    row_indices = np.arange(dim).repeat(dim).reshape((dim, dim))
    col_indices = np.transpose(row_indices)
    return (np.abs(row_indices - col_indices) < thickness).astype("uint8")


def rotation_matrix_transpose_from_quaternion(quat: np.ndarray) -> list[np.ndarray]:
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


def rotation_matrix_transpose(angle: float, axis: Vector3DLike) -> np.ndarray:
    if all(np.array(axis)[:2] == np.zeros(2)):
        return rotation_about_z(angle * np.sign(axis[2])).T
    return rotation_matrix(angle, axis).T


def rotation_matrix(
    angle: float,
    axis: Vector3DLike,
    homogeneous: bool = False,
) -> np.ndarray:
    """Rotation in R^3 about a specified axis of rotation."""
    inhomogeneous_rotation_matrix = Rotation.from_rotvec(
        angle * normalize(axis)
    ).as_matrix()
    if not homogeneous:
        return inhomogeneous_rotation_matrix
    else:
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = inhomogeneous_rotation_matrix
        return rotation_matrix


def rotation_about_z(angle: float) -> np.ndarray:
    """Returns a rotation matrix for a given angle.

    Parameters
    ----------
    angle
        Angle for the rotation matrix.

    Returns
    -------
    np.ndarray
        Gives back the rotated matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def get_norm(vector: np.ndarray) -> float:
    """Returns the norm of the vector.

    Parameters
    ----------
    vector
        The vector for which you want to find the norm.

    Returns
    -------
    float
        The norm of the vector.
    """
    return np.linalg.norm(vector)


def z_to_vector(vector: np.ndarray) -> np.ndarray:
    """
    Returns some matrix in SO(3) which takes the z-axis to the
    (normalized) vector provided as an argument
    """
    axis_z = normalize(vector)
    axis_y = normalize(cross(axis_z, RIGHT))
    axis_x = cross(axis_y, axis_z)
    if np.linalg.norm(axis_y) == 0:
        # the vector passed just so happened to be in the x direction.
        axis_x = normalize(cross(UP, axis_z))
        axis_y = -cross(axis_x, axis_z)

    return np.array([axis_x, axis_y, axis_z]).T


def angle_of_vector(vector: Sequence[float] | np.ndarray) -> float:
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
    if isinstance(vector, np.ndarray) and len(vector.shape) > 1:
        if vector.shape[0] < 2:
            raise ValueError("Vector must have the correct dimensions. (2, n)")
        c_vec = np.empty(vector.shape[1], dtype=np.complex128)
        c_vec.real = vector[0]
        c_vec.imag = vector[1]
        val1: float = np.angle(c_vec)
        return val1
    val: float = np.angle(complex(*vector[:2]))
    return val


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
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
    float
        The angle between the vectors.
    """
    val: float = 2 * np.arctan2(
        np.linalg.norm(normalize(v1) - normalize(v2)),
        np.linalg.norm(normalize(v1) + normalize(v2)),
    )

    return val


def normalize(
    vect: npt.NDArray[float], fall_back: npt.NDArray[float] | None = None
) -> npt.NDArray[float]:
    norm = get_norm(vect)
    if norm > 0:
        return np.array(vect) / norm
    elif fall_back is not None:
        return np.array(fall_back)
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


def get_unit_normal(v1: Vector3DLike, v2: Vector3DLike, tol: float = 1e-6) -> Vector3D:
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
    np_v1 = np.asarray(v1)
    np_v2 = np.asarray(v2)

    # Instead of normalizing v1 and v2, just divide by the greatest
    # of all their absolute components, which is just enough
    div1, div2 = max(np.abs(np_v1)), max(np.abs(np_v2))
    if div1 == 0.0:
        if div2 == 0.0:
            return DOWN
        u = np_v2 / div2
    elif div2 == 0.0:
        u = np_v1 / div1
    else:
        # Normal scenario: v1 and v2 are both non-null
        u1, u2 = np_v1 / div1, np_v2 / div2
        cp = cross(u1, u2)
        cp_norm = np.sqrt(norm_squared(cp))
        if cp_norm > tol:
            return cp / cp_norm
        # Otherwise, v1 and v2 were aligned
        u = u1

    # If you are here, you have an "unique", non-zero, unit-ish vector u
    # If it's also too aligned to the Z axis, just return DOWN
    if abs(u[0]) < tol and abs(u[1]) < tol:
        return DOWN
    # Otherwise rotate u in the plane it shares with the Z axis,
    # 90° TOWARDS the Z axis. This is done via (u x [0, 0, 1]) x u,
    # which gives [-xz, -yz, x²+y²] (slightly scaled as well)
    cp = np.array([-u[0] * u[2], -u[1] * u[2], u[0] * u[0] + u[1] * u[1]])
    cp_norm = np.sqrt(norm_squared(cp))
    # Because the norm(u) == 0 case was filtered in the beginning,
    # there is no need to check if the norm of cp is 0
    return cp / cp_norm


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
    n: int, *, radius: float = 1, start_angle: float | None = None
) -> tuple[np.ndarray, float]:
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
        start_angle = 0 if n % 2 == 0 else TAU / 4

    start_vector = rotate_vector(RIGHT * radius, start_angle)
    vertices = compass_directions(n, start_vector)

    return vertices, start_angle


def complex_to_R3(complex_num: complex) -> np.ndarray:
    return np.array((complex_num.real, complex_num.imag, 0))


def R3_to_complex(point: Sequence[float]) -> np.ndarray:
    return complex(*point[:2])


def complex_func_to_R3_func(
    complex_func: Callable[[complex], complex],
) -> Callable[[Point3DLike], Point3D]:
    return lambda p: complex_to_R3(complex_func(R3_to_complex(p)))


def center_of_mass(points: PointNDLike_Array) -> PointND:
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
    return np.average(points, 0, np.ones(len(points)))


def midpoint(
    point1: Sequence[float],
    point2: Sequence[float],
) -> float | np.ndarray:
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


def line_intersection(
    line1: Sequence[np.ndarray], line2: Sequence[np.ndarray]
) -> np.ndarray:
    """Returns the intersection point of two lines, each defined by
    a pair of distinct points lying on the line.

    Parameters
    ----------
    line1
        A list of two points that determine the first line.
    line2
        A list of two points that determine the second line.

    Returns
    -------
    np.ndarray
        The intersection points of the two lines which are intersecting.

    Raises
    ------
    ValueError
        Error is produced if the two lines don't intersect with each other
        or if the coordinates don't lie on the xy-plane.
    """
    if any(np.array([line1, line2])[:, :, 2].reshape(-1)):
        # checks for z coordinates != 0
        raise ValueError("Coords must be in the xy-plane.")

    # algorithm from https://stackoverflow.com/a/42727584
    padded = (
        np.pad(np.array(i)[:, :2], ((0, 0), (0, 1)), constant_values=1)
        for i in (line1, line2)
    )
    line1, line2 = (cross(*i) for i in padded)
    x, y, z = cross(line1, line2)

    if z == 0:
        raise ValueError(
            "The lines are parallel, there is no unique intersection point."
        )

    return np.array([x / z, y / z, 0])


def find_intersection(
    p0s: Point3DLike_Array,
    v0s: Vector3DLike_Array,
    p1s: Point3DLike_Array,
    v1s: Vector3DLike_Array,
    threshold: float = 1e-5,
) -> list[Point3D]:
    """Return the intersection of a line passing through p0 in direction v0
    with one passing through p1 in direction v1 (or array of intersections
    from arrays of such points/directions).
    For 3d values, it returns the point on the ray p0 + v0 * t closest to the
    ray p1 + v1 * t
    """
    # algorithm from https://en.wikipedia.org/wiki/Skew_lines#Nearest_points
    result = []

    for p0, v0, p1, v1 in zip(p0s, v0s, p1s, v1s, strict=False):
        normal = cross(v1, cross(v0, v1))
        denom = max(np.dot(v0, normal), threshold)
        result += [p0 + np.dot(p1 - p0, normal) / denom * v0]
    return result


def get_winding_number(points: Sequence[np.ndarray]) -> float:
    """Determine the number of times a polygon winds around the origin.

    The orientation is measured mathematically positively, i.e.,
    counterclockwise.

    Parameters
    ----------
    points
        The vertices of the polygon being queried.

    Examples
    --------

    >>> from manim import Square, UP, get_winding_number
    >>> polygon = Square()
    >>> get_winding_number(polygon.get_vertices())
    np.float64(1.0)
    >>> polygon.shift(2 * UP)
    Square
    >>> get_winding_number(polygon.get_vertices())
    np.float64(0.0)
    """
    total_angle: float = 0
    for p1, p2 in adjacent_pairs(points):
        d_angle = angle_of_vector(p2) - angle_of_vector(p1)
        d_angle = ((d_angle + PI) % TAU) - PI
        total_angle += d_angle
    val: float = total_angle / TAU
    return val


def shoelace(x_y: Point2D_Array) -> float:
    """2D implementation of the shoelace formula.

    Returns
    -------
    :class:`float`
        Returns signed area.
    """
    x = x_y[:, 0]
    y = x_y[:, 1]
    val: float = np.trapezoid(y, x)
    return val


def shoelace_direction(x_y: Point2D_Array) -> str:
    """Uses the area determined by the shoelace method to determine whether
    the input set of points is directed clockwise or counterclockwise.

    Returns
    -------
    :class:`str`
        Either ``"CW"`` or ``"CCW"``.
    """
    area = shoelace(x_y)
    return "CW" if area > 0 else "CCW"


def cross2d(
    a: Vector2D | Vector2D_Array,
    b: Vector2D | Vector2D_Array,
) -> ManimFloat | npt.NDArray[ManimFloat]:
    """Compute the determinant(s) of the passed
    vector (sequences).

    Parameters
    ----------
    a
        A vector or a sequence of vectors.
    b
        A vector or a sequence of vectors.

    Returns
    -------
    Sequence[float] | float
        The determinant or sequence of determinants
        of the first two components of the specified
        vectors.

    Examples
    --------
    .. code-block:: pycon

        >>> cross2d(np.array([1, 2]), np.array([3, 4]))
        np.int64(-2)
        >>> cross2d(
        ...     np.array([[1, 2, 0], [1, 0, 0]]),
        ...     np.array([[3, 4, 0], [0, 1, 0]]),
        ... )
        array([-2,  1])
    """
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
    rings = [list(range(e0, e1)) for e0, e1 in zip([0, *ring_ends], ring_ends)]

    def is_in(point, ring_id):
        return (
            abs(abs(get_winding_number([i - point for i in verts[rings[ring_id]]])) - 1)
            < 1e-5
        )

    def ring_area(ring_id):
        ring = rings[ring_id]
        s = 0
        for i, j in zip(ring[1:], ring):
            s += cross2d(verts[i], verts[j])
        return abs(s) / 2

    # Points at the same position may cause problems
    for i in rings:
        verts[i[0]] += (verts[i[1]] - verts[i[0]]) * 1e-6
        verts[i[-1]] += (verts[i[-2]] - verts[i[-1]]) * 1e-6

    # First, we should know which rings are directly contained in it for each ring

    right = [max(verts[rings[i], 0]) for i in range(len(rings))]
    left = [min(verts[rings[i], 0]) for i in range(len(rings))]
    top = [max(verts[rings[i], 1]) for i in range(len(rings))]
    bottom = [min(verts[rings[i], 1]) for i in range(len(rings))]
    area = [ring_area(i) for i in range(len(rings))]

    # The larger ring must be outside
    rings_sorted = list(range(len(rings)))
    rings_sorted.sort(key=lambda x: area[x], reverse=True)

    def is_in_fast(ring_a, ring_b):
        # Whether a is in b
        return (
            left[ring_b] <= left[ring_a] <= right[ring_a] <= right[ring_b]
            and bottom[ring_b] <= bottom[ring_a] <= top[ring_a] <= top[ring_b]
            and is_in(verts[rings[ring_a][0]], ring_b)
        )

    children = [[]] * len(rings)
    for idx, i in enumerate(rings_sorted):
        for j in rings_sorted[:idx][::-1]:
            if is_in_fast(i, j):
                children[j].append(i)
                break

    res = []

    # Then, we can use earcut for each part
    used = [False] * len(rings)
    for i in rings_sorted:
        if used[i]:
            continue
        v = rings[i]
        ring_ends = [len(v)]
        for j in children[i]:
            used[j] = True
            v += rings[j]
            ring_ends.append(len(v))
        np_ring_ends = np.array(ring_ends, dtype=np.uint32)
        res += [v[i] for i in earcut(verts[v, :2], np_ring_ends)]

    return res


def cartesian_to_spherical(vec: Vector3DLike) -> np.ndarray:
    """Returns an array of numbers corresponding to each
    polar coordinate value (distance, phi, theta).

    Parameters
    ----------
    vec
        A numpy array or a sequence of floats ``[x, y, z]``.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros(3)
    r = norm
    phi = np.arccos(vec[2] / r)
    theta = np.arctan2(vec[1], vec[0])
    return np.array([r, theta, phi])


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
    norm_vector: Vector3D = OUT,
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
    direction = cross(p1 - p2, norm_vector)
    m = midpoint(p1, p2)
    return [m + direction, m - direction]
