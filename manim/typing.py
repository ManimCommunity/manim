from __future__ import annotations

from os import PathLike
from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

# Color Types

ManimFloat: TypeAlias = np.float64
ManimInt: TypeAlias = np.int64
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
This type alias is mostly made available for internal use and only includes the numpy type.
"""

Point2D: TypeAlias = Union[InternalPoint2D, Tuple[float, float]]
""" `shape: (2,)` A 2D point. `[float, float]`. """

InternalPoint3D: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3,)` A 3D point. `[float, float, float]`.
This type alias is mostly made available for internal use and only includes the numpy type.
"""

Point3D: TypeAlias = Union[InternalPoint3D, Tuple[float, float, float]]
""" `shape: (3,)` A 3D point. `[float, float, float]` """

# Bezier Types
QuadraticBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3,3)` An Array of Quadratic Bezier Handles `[[float, float, float], [float, float, float], [float, float, float]]`. """

QuadraticBezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3,3)` An Array of Quadratic Bezier Handles `[[[float, float, float], [float, float, float], [float, float, float]], ...]`. """

CubicBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (4,3)` An Array of Cubic Bezier Handles `[[float, float, float], [float, float, float], [float, float, float], [float, float, float]]`. """

BezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3)` An Array of Cubic Bezier Handles `[[float, float, float], ...]`.
`N` Is always multiples of the degree of the Bezier curve.
(Please refer to the documentation of the function you are using for further type Information)
"""

FlatBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N)` An Array of Bezier Handles but flattened `[float, ...]`."""

Point2D_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,2)` An Array of Points in 2D Space `[[float, float], ...]`.

(Please refer to the documentation of the function you are using for further type Information)
"""

InternalPoint3D_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3)` An Array of Points in 3D Space `[[float, float, float], ...]`.
This type alias is mostly made available for internal use and only includes the numpy type.
"""

Point3D_Array: TypeAlias = Union[
    InternalPoint3D_Array, Tuple[Tuple[float, float, float], ...]
]
""" `shape: (N,3)` An Array of Points in 3D Space `[[float, float, float], ...]`.

(Please refer to the documentation of the function you are using for further type Information)
"""

BezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,PPC,3)` An Array of Bezier Handles `[[[float, float, float], ...], ...]`.
`PPC` Is the number of points per bezier curve. `N` Is the number of bezier curves.
(Please refer to the documentation of the function you are using for further type Information)
"""

# Vector Types
Vector3: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3,)` A Vector `[float, float, float]`. """

Vector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,)` A Vector `[float, ...]`. """

RowVector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (1,N)` A Row Vector `[[float, ...]]`. """

ColVector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,1)` A Column Vector `[[float], [float], ...]`. """

MatrixMN: TypeAlias = npt.NDArray[PointDType]
""" `shape: (M,N)` A Matrix `[[float, ...], [float, ...], ...]`. """

Zeros: TypeAlias = npt.NDArray[ManimFloat]
"""A Matrix of Zeros. Typically created with `numpy.zeros((M,N))`"""

# Due to current limitations (see https://github.com/python/mypy/issues/14656 / 8263), we don't specify the first argument type (Mobject).
FunctionOverride: TypeAlias = Callable[..., None]
"""Function type returning an animation for the specified Mobject."""


# Misc
PathFuncType: TypeAlias = Callable[[Point3D, Point3D, float], Point3D]
"""Function mapping two points and an alpha value to a new point"""

MappingFunction: TypeAlias = Callable[[Point3D], Point3D]
"""A function mapping a Point3D to another Point3D"""

Image: TypeAlias = np.ndarray
"""An Image"""

StrPath: TypeAlias = "str | PathLike[str]"
StrOrBytesPath: TypeAlias = "str | bytes | PathLike[str] | PathLike[bytes]"
