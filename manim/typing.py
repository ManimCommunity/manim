"""Custom type definitions used in Manim.

.. admonition:: Note for developers
    :class: important

    Around the source code there are multiple strings which look like this:

    .. code-block::

        '''
        [CATEGORY]
        <category_name>
        '''

    All type aliases defined under those strings will be automatically
    classified under that category.

    If you need to define a new category, respect the format described above.

.. autotypingmodule:: manim.typing

"""
from __future__ import annotations

from os import PathLike
from typing import Annotated, Callable, Literal, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

__all__ = [
    "ManimFloat",
    "ManimInt",
    "ManimColorDType",
    "RGB_Array_Float",
    "RGB_Tuple_Float",
    "RGB_Array_Int",
    "RGB_Tuple_Int",
    "RGBA_Array_Float",
    "RGBA_Tuple_Float",
    "RGBA_Array_Int",
    "RGBA_Tuple_Int",
    "HSV_Array_Float",
    "HSV_Tuple_Float",
    "ManimColorInternal",
    "PointDType",
    "InternalPoint2D",
    "Point2D",
    "InternalPoint2D_Array",
    "Point2D_Array",
    "InternalPoint3D",
    "Point3D",
    "InternalPoint3D_Array",
    "Point3D_Array",
    "Vector2",
    "Vector3",
    "Vector",
    "RowVector",
    "ColVector",
    "MatrixMN",
    "Zeros",
    "QuadraticBezierPoints",
    "QuadraticBezierPoints_Array",
    "QuadraticBezierPath",
    "QuadraticSpline",
    "CubicBezierPoints",
    "CubicBezierPoints_Array",
    "CubicBezierPath",
    "CubicSpline",
    "BezierPoints",
    "BezierPoints_Array",
    "BezierPath",
    "Spline",
    "FlatBezierPoints",
    "FunctionOverride",
    "PathFuncType",
    "MappingFunction",
    "Image",
    "StrPath",
    "StrOrBytesPath",
]


"""
[CATEGORY]
Primitive data types
"""

ManimFloat: TypeAlias = np.float64
ManimInt: TypeAlias = np.int64


"""
[CATEGORY]
Color types
"""

ManimColorDType: TypeAlias = ManimFloat

RGB_Array_Float: TypeAlias = npt.NDArray[ManimFloat]
RGB_Tuple_Float: TypeAlias = tuple[float, float, float]

RGB_Array_Int: TypeAlias = npt.NDArray[ManimInt]
RGB_Tuple_Int: TypeAlias = tuple[int, int, int]

RGBA_Array_Float: TypeAlias = npt.NDArray[ManimFloat]
RGBA_Tuple_Float: TypeAlias = tuple[float, float, float, float]

RGBA_Array_Int: TypeAlias = npt.NDArray[ManimInt]
RGBA_Tuple_Int: TypeAlias = tuple[int, int, int, int]

HSV_Array_Float: TypeAlias = RGB_Array_Float
HSV_Tuple_Float: TypeAlias = RGB_Tuple_Float

ManimColorInternal: TypeAlias = npt.NDArray[ManimColorDType]


"""
[CATEGORY]
Point types
"""

PointDType: TypeAlias = ManimFloat

InternalPoint2D: TypeAlias = npt.NDArray[PointDType]
""" `shape: (2,)` A 2D point: `[float, float]`.
This type alias is mostly made available for internal use and only includes the NumPy type.
"""

Point2D: TypeAlias = Union[InternalPoint2D, tuple[float, float]]
"""`shape (2,)`
A 2D point: `[float, float]`
"""

InternalPoint2D_Array: TypeAlias = npt.NDArray[PointDType]
"""`shape (N, 3)`
An array of Point2D: `[[float, float], ...]`
This type alias is mostly made available for internal use and only includes the NumPy type.
"""

Point2D_Array: TypeAlias = Union[InternalPoint2D_Array, tuple[Point2D, ...]]
"""`shape (N, 2)`
An array of Point2D objects: `[[float, float], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

InternalPoint3D: TypeAlias = npt.NDArray[PointDType]
"""`shape (3,)`
A 3D point: `[float, float, float]`
This type alias is mostly made available for internal use and only includes the NumPy type.
"""

Point3D: TypeAlias = Union[InternalPoint3D, tuple[float, float, float]]
"""`shape (3,)`
A 3D point: `[float, float, float]`
"""

InternalPoint3D_Array: TypeAlias = npt.NDArray[PointDType]
"""`shape (N, 3)`
An array of Point3D objects: `[[float, float, float], ...]`
This type alias is mostly made available for internal use and only includes the NumPy type.
"""

Point3D_Array: TypeAlias = Union[InternalPoint3D_Array, tuple[Point3D, ...]]
"""`shape (N, 3)`
An array of Point3D objects: `[[float, float, float], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""


"""
[CATEGORY]
Vector types
"""

Vector2: TypeAlias = Union[npt.NDArray[PointDType], tuple[float, float]]
"""`shape (2,)`
A 2D vector: `[float, float]`
"""

Vector3: TypeAlias = Union[npt.NDArray[PointDType], tuple[float, float, float]]
"""`shape (3,)`
A 3D vector: `[float, float, float]`
"""

Vector: TypeAlias = Union[npt.NDArray[PointDType], tuple[float, ...]]
"""`shape (N,)`
An `N`-D vector: `[float, ...]`
"""

RowVector: TypeAlias = Union[npt.NDArray[PointDType], tuple[tuple[float, ...]]]
"""`shape (1, N)`
A row vector: `[[float, ...]]`
"""

ColVector: TypeAlias = Union[npt.NDArray[PointDType], tuple[tuple[float], ...]]
"""`shape (N, 1)`
A column vector: `[[float], [float], ...]`
"""

MatrixMN: TypeAlias = Union[npt.NDArray[PointDType], tuple[tuple[float, ...], ...]]
"""`shape (M, N)`
A matrix: `[[float, ...], [float, ...], ...]`
"""

Zeros: TypeAlias = Union[npt.NDArray[ManimFloat], tuple[tuple[Literal[0], ...], ...]]
"""A matrix of zeros, typically created with ``numpy.zeros((M, N))``"""


"""
[CATEGORY]
Bézier types
"""

QuadraticBezierPoints: TypeAlias = Union[
    npt.NDArray[PointDType], tuple[Point3D, Point3D, Point3D]
]
"""`shape (3, 3)`
A `Point3D_Array` of control points for a single quadratic Bézier curve: `[[float, float, float], [float, float, float], [float, float, float]]`
"""

QuadraticBezierPoints_Array: TypeAlias = Union[
    npt.NDArray[PointDType], tuple[QuadraticBezierPoints, ...]
]
"""`shape (N, 3, 3)`
An array of `N QuadraticBezierPoints` objects: `[[[float, float, float], [float, float, float], [float, float, float]], ...]`
"""

QuadraticBezierPath: TypeAlias = Point3D_Array
"""`shape (3*N, 3)`
An array of `3*N Point3D` objects, where each one of the `N` consecutive blocks of
3 points represents a quadratic Bézier curve: `[[float, float, float], ...], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

QuadraticSpline: TypeAlias = QuadraticBezierPath
"""`shape (3*N, 3)`
A special case of `QuadraticBezierPath` where all the `N` quadratic Bézier
curves are connected, forming a quadratic spline: `[[float, float, float], ...], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

CubicBezierPoints: TypeAlias = Union[
    npt.NDArray[PointDType], tuple[Point3D, Point3D, Point3D, Point3D]
]
"""`shape (4, 3)`
A `Point3D_Array` of control points for a single cubic Bézier curve: `[[float, float, float], [float, float, float], [float, float, float], [float, float, float]]`
"""

CubicBezierPoints_Array: TypeAlias = Union[
    npt.NDArray[PointDType], tuple[CubicBezierPoints, ...]
]
"""`shape (N, 4, 3)`
An array of `N CubicBezierPoints` objects: `[[[float, float, float], [float, float, float], [float, float, float], [float, float, float]], ...]`
"""

CubicBezierPath: TypeAlias = Point3D_Array
"""`shape (4*N, 3)`
An array of `4*N Point3D` objects, where each one of the `N` consecutive blocks of
4 points represents a cubic Bézier curve: `[[float, float, float], ...], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

CubicSpline: TypeAlias = CubicBezierPath
"""`shape (4*N, 3)`
A special case of `CubicBezierPath` where all the `N` cubic Bézier
curves are connected, forming a quadratic spline: `[[float, float, float], ...], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

BezierPoints: TypeAlias = Point3D_Array
"""`shape (PPC, 3)`
A `Point3D_Array` of `PPC = n + 1` ("Points Per Curve") control points for a single `n`-th
degree Bézier curve: `[[float, float, float], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

BezierPoints_Array: TypeAlias = Union[npt.NDArray[PointDType], tuple[BezierPoints, ...]]
"""`shape (N, PPC, 3)`
An array of `N BezierPoints` objects containing `PPC Point3D` objects each: `[[[float, float, float], ...], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

BezierPath: TypeAlias = Point3D_Array
"""`shape (PPC*N, 3)`
An array of `PPC * N Point3D` objects, where where each one of the `N` consecutive blocks of
`PPC = n + 1` ("Points Per Curve") points represents a Bézier curve of `n`-th degree:
`[[float, float, float], ...], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

Spline: TypeAlias = BezierPath
"""`shape (PPC*N, 3)`
A special case of BezierPath where all the `N` Bézier curves consisting of `PPC` ("Points Per Curve")
`Point3D` objects are connected, forming an `n`-th degree spline: `[[float, float, float], ...], ...]`
(Please refer to the documentation of the function you are using for further type information.)
"""

FlatBezierPoints: TypeAlias = Union[npt.NDArray[PointDType], tuple[float, ...]]
"""`shape (N)`
A flattened array of Bézier control points: `[float, ...]`
"""


"""
[CATEGORY]
Function types
"""

# Due to current limitations (see https://github.com/python/mypy/issues/14656 / 8263), we don't specify the first argument type (Mobject).
FunctionOverride: TypeAlias = Callable[..., None]
"""Function type returning an `Animation` for the specified `Mobject`."""

PathFuncType: TypeAlias = Callable[[Point3D, Point3D, float], Point3D]
"""Function mapping two `Point3D` objects and an alpha value to a new `Point3D`."""

MappingFunction: TypeAlias = Callable[[Point3D], Point3D]
"""A function mapping a `Point3D` to another `Point3D`."""


"""
[CATEGORY]
Image types
"""

Image: TypeAlias = np.ndarray
"""An image."""


"""
[CATEGORY]
Path types
"""

StrPath: TypeAlias = Union[str, PathLike[str]]
StrOrBytesPath: TypeAlias = Union[str, bytes, PathLike[str], PathLike[bytes]]
