from typing import Protocol, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import Annotated, Literal, TypeAlias, TypeVar

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

ManimColorInternal: TypeAlias = npt.NDArray[ManimColorDType]

# Point Types

PointDType: TypeAlias = ManimFloat
""" DType for all points. """
Point2D: TypeAlias = npt.NDArray[PointDType]
""" `shape: (2,) A 2D point. [float,float] """
Point3D: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3,) A 3D point. `[float,float,float]` """

# Bezier Types
QuadraticBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (3,3)` An Array of Quadratic Bezier Handles `[[float,float,float], [float,float,float], [float,float,float]]`. """

QuadraticBezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3,3)` An Array of Quadratic Bezier Handles `[[[float,float,float], [float,float,float], [float,float,float]], ...]`. """

CubicBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (4,3)` An Array of Cubic Bezier Handles `[[float,float,float], [float,float,float], [float,float,float], [float,float,float]]`. """

BezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3)` An Array of Cubic Bezier Handles `[[float,float,float],...]`.
`N` Is always multiples of the degree of the Bezier curve.
(Please refer to the documentation of the function you are using for further type Information)
"""

FlatBezierPoints: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N)` An Array of Bezier Handles but flattened `[float,...]`."""

Point3D_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,3)` An Array Points in 3D Space `[[float,float,float],...]`.

(Please refer to the documentation of the function you are using for further type Information)
"""

BezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,PPC,3)` An Array of Bezier Handles `[[[float,float,float],...],...]`.
`PPC` Is the number of points per bezier curve. `N` Is the number of bezier curves.
(Please refer to the documentation of the function you are using for further type Information)
"""

# Vector Types
Vector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,)` A Vector `[float,...]`. """

RowVector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (1,N)` A Row Vector `[[float,...]]`. """

ColVector: TypeAlias = npt.NDArray[PointDType]
""" `shape: (N,1)` A Column Vector `[[float],[float],...]`. """

MatrixMN: TypeAlias = npt.NDArray[PointDType]
""" `shape: (M,N)` A Matrix `[[float,...],[float,...],...]`. """
