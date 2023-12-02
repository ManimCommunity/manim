from __future__ import annotations

from os import PathLike
from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

manim_type_aliases = [
    # Primitive Data Types
    "ManimFloat",
    "ManimInt",
    # Color Types
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
    # Point Types
    "PointDType",
    "InternalPoint2D",
    "Point2D",
    "InternalPoint2D_Array",
    "Point2D_Array",
    "InternalPoint3D",
    "Point3D",
    "InternalPoint3D_Array",
    "Point3D_Array",
    # Vector Types
    "Vector2",
    "Vector3",
    "Vector",
    "RowVector",
    "ColVector",
    "MatrixMN",
    "Zeros",
    # Bezier Types
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
    # Function Types
    "FunctionOverride",
    "PathFuncType",
    "MappingFunction",
    # Image Types
    "Image",
    # Path Types
    "StrPath",
    "StrOrBytesPath",
]

# Primitive Data Types
ManimFloat: TypeAlias = np.float64
ManimInt: TypeAlias = np.int64

# Color Types
ManimColorDType: TypeAlias = ManimFloat

RGB_Array_Float: TypeAlias = npt.NDArray[ManimFloat]
RGB_Tuple_Float: TypeAlias = Tuple[float, float, float]

RGB_Array_Int: TypeAlias = npt.NDArray[ManimInt]
RGB_Tuple_Int: TypeAlias = Tuple[int, int, int]

RGBA_Array_Float: TypeAlias = npt.NDArray[ManimFloat]
RGBA_Tuple_Float: TypeAlias = Tuple[float, float, float, float]

RGBA_Array_Int: TypeAlias = npt.NDArray[ManimInt]
RGBA_Tuple_Int: TypeAlias = Tuple[int, int, int, int]

HSV_Array_Float: TypeAlias = RGB_Array_Float
HSV_Tuple_Float: TypeAlias = RGB_Tuple_Float

ManimColorInternal: TypeAlias = npt.NDArray[ManimColorDType]

# Point Types

PointDType: TypeAlias = ManimFloat
""" DType for all points. """

InternalPoint2D: TypeAlias = npt.NDArray[PointDType]
""" `shape: (2,)` A 2D point. `[float, float]`.
This type alias is mostly made available for internal use and only includes the NumPy type.
"""

Point2D: TypeAlias = Union[InternalPoint2D, Tuple[float, float]]
""" `shape: (2,)` A 2D point. `[float, float]`. """

InternalPoint2D_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3)` An array of Point2D: `[[float, float], ...]`.
This type alias is mostly made available for internal use and only includes the NumPy type.
"""

Point2D_Array: TypeAlias = Union[InternalPoint2D_Array, Tuple[Tuple[float, float], ...]]
""" `shape: (N,2)` An array of Point2D objects: `[[float, float], ...]`.
(Please refer to the documentation of the function you are using for further type information.)
"""

InternalPoint3D: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3,)` A 3D point. `[float, float, float]`.
This type alias is mostly made available for internal use and only includes the NumPy type.
"""

Point3D: TypeAlias = Union[InternalPoint3D, Tuple[float, float, float]]
""" `shape: (3,)` A 3D point. `[float, float, float]` """

InternalPoint3D_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3)` An array of Point3D objects: `[[float, float, float], ...]`.
This type alias is mostly made available for internal use and only includes the NumPy type.
"""

Point3D_Array: TypeAlias = Union[
    InternalPoint3D_Array, Tuple[Tuple[float, float, float], ...]
]
""" `shape: (N,3)` An array of Point3D objects: `[[float, float, float], ...]`.

(Please refer to the documentation of the function you are using for further type Information)
"""

# Vector Types

Vector2: TypeAlias = npt.NDArray[PointDType]
""" `shape: (2,)` A vector `[float, float]`. """

Vector3: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3,)` A vector `[float, float, float]`. """

Vector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,)` A vector `[float, ...]`. """

RowVector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (1,N)` A row vector `[[float, ...]]`. """

ColVector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,1)` A column vector `[[float], [float], ...]`. """

MatrixMN: TypeAlias = npt.NDArray[PointDType]
""" `shape: (M,N)` A matrix `[[float, ...], [float, ...], ...]`. """

Zeros: TypeAlias = npt.NDArray[ManimFloat]
"""A matrix of zeros. Typically created with `numpy.zeros((M,N))`."""

# Bezier Types

QuadraticBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3,3)` A Point3D_Array of control points for a single quadratic Bézier curve: `[[float, float, float], [float, float, float], [float, float, float]]`. """

QuadraticBezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3,3)` An array of N QuadraticBezierPoint objects: `[[[float, float, float], [float, float, float], [float, float, float]], ...]`. """

QuadraticBezierPath: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3*N, 3)` An array of `3*N` Point3D objects, where every consecutive block of
3 points represents a quadratic Bézier curve: `[[float, float, float], ...], ...]`.
`N` is the number of Bézier curves.
(Please refer to the documentation of the function you are using for further type information.)
"""

QuadraticSpline: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3*N, 3)` A special case of QuadraticBezierPath where all the N quadratic Bézier
curves are connected, forming a quadratic spline: `[[float, float, float], ...], ...]`.
`N` is the number of Bézier curves.
(Please refer to the documentation of the function you are using for further type information.)
"""

CubicBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (4,3)` A Point3D_Array of control points for a single cubic Bézier curve: `[[float, float, float], [float, float, float], [float, float, float], [float, float, float]]`. """

CubicBezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,4,3)` An array of N CubicBezierPoint objects: `[[[float, float, float], [float, float, float], [float, float, float], [float, float, float]], ...]`.
"""

CubicBezierPath: TypeAlias = npt.NDArray[PointDType]
""" `shape: (4*N, 3)` An array of `4*N` Point3D objects, where every consecutive block of
4 points represents a cubic Bézier curve: `[[float, float, float], ...], ...]`.
`N` is the number of Bézier curves.
(Please refer to the documentation of the function you are using for further type information.)
"""

CubicSpline: TypeAlias = npt.NDArray[PointDType]
""" `shape: (4*N, 3)` A special case of CubicBezierPath where all the N cubic Bézier
curves are connected, forming a quadratic spline: `[[float, float, float], ...], ...]`.
`N` is the number of Bézier curves.
(Please refer to the documentation of the function you are using for further type information.)
"""

BezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (PPC, 3)` A Point3D_Array of control points for a single n-th degree Bézier curve: `[[float, float, float], ...]`.
`PPC` ("Points Per Curve") is the number of points defining the Bézier curve, which is always 1 plus its degree n.
(Please refer to the documentation of the function you are using for further type information.)
"""

BezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N, PPC, 3)` An array of N BezierPoint objects: `[[[float, float, float], ...], ...]`.
`N` is the number of Bézier curves.
`PPC` ("Points Per Curve") is the number of points per Bézier curve.
(Please refer to the documentation of the function you are using for further type information.)
"""

BezierPath: TypeAlias = npt.NDArray[PointDType]
""" `shape: (PPC * N, 3)` An array of `PPC * N` Point3D objects, where every consecutive block of
`PPC` points represents a Bézier curve of n-th degree: `[[float, float, float], ...], ...]`.
`PPC` ("Points Per Curve") is the number of points per Bézier curve.
`N` is the number of Bézier curves.
(Please refer to the documentation of the function you are using for further type information.)
"""

Spline: TypeAlias = npt.NDArray[PointDType]
""" `shape: (PPC * N, 3)` A special case of BezierPath where all the N Bézier
curves are connected, forming an n-th degree spline: `[[float, float, float], ...], ...]`.
`PPC` ("Points Per Curve") is the number of points per Bézier curve.
`N` is the number of Bézier curves.
(Please refer to the documentation of the function you are using for further type information.)
"""

FlatBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N)` A flattened array of Bézier control points: `[float, ...]`."""

# Function Types

# Due to current limitations (see https://github.com/python/mypy/issues/14656 / 8263), we don't specify the first argument type (Mobject).
FunctionOverride: TypeAlias = Callable[..., None]
"""Function type returning an animation for the specified Mobject."""

PathFuncType: TypeAlias = Callable[[Point3D, Point3D, float], Point3D]
"""Function mapping two Point3D objects and an alpha value to a new Point3D."""

MappingFunction: TypeAlias = Callable[[Point3D], Point3D]
"""A function mapping a Point3D to another Point3D."""

# Image Types

Image: TypeAlias = np.ndarray
"""An image."""

# Path Types

StrPath: TypeAlias = "str | PathLike[str]"
StrOrBytesPath: TypeAlias = "str | bytes | PathLike[str] | PathLike[bytes]"
