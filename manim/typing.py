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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from os import PathLike
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "ManimFloat",
    "ManimInt",
    "ManimColorDType",
    "FloatRGB",
    "FloatRGBLike",
    "FloatRGB_Array",
    "FloatRGBLike_Array",
    "IntRGB",
    "IntRGBLike",
    "FloatRGBA",
    "FloatRGBALike",
    "FloatRGBA_Array",
    "FloatRGBALike_Array",
    "IntRGBA",
    "IntRGBALike",
    "FloatHSV",
    "FloatHSVLike",
    "FloatHSL",
    "FloatHSLLike",
    "FloatHSVA",
    "FloatHSVALike",
    "ManimColorInternal",
    "PointDType",
    "Point2D",
    "Point2DLike",
    "Point2D_Array",
    "Point2DLike_Array",
    "Point3D",
    "Point3DLike",
    "Point3D_Array",
    "Point3DLike_Array",
    "PointND",
    "PointNDLike",
    "PointND_Array",
    "PointNDLike_Array",
    "Vector2D",
    "Vector2DLike",
    "Vector2D_Array",
    "Vector2DLike_Array",
    "Vector3D",
    "Vector3DLike",
    "Vector3D_Array",
    "Vector3DLike_Array",
    "VectorND",
    "VectorNDLike",
    "VectorND_Array",
    "VectorNDLike_Array",
    "RowVector",
    "ColVector",
    "MatrixMN",
    "Zeros",
    "QuadraticBezierPoints",
    "QuadraticBezierPointsLike",
    "QuadraticBezierPoints_Array",
    "QuadraticBezierPointsLike_Array",
    "QuadraticBezierPath",
    "QuadraticBezierPathLike",
    "QuadraticSpline",
    "QuadraticSplineLike",
    "CubicBezierPoints",
    "CubicBezierPointsLike",
    "CubicBezierPoints_Array",
    "CubicBezierPointsLike_Array",
    "CubicBezierPath",
    "CubicBezierPathLike",
    "CubicSpline",
    "CubicSplineLike",
    "BezierPoints",
    "BezierPointsLike",
    "BezierPoints_Array",
    "BezierPointsLike_Array",
    "BezierPath",
    "BezierPathLike",
    "Spline",
    "SplineLike",
    "FlatBezierPoints",
    "FunctionOverride",
    "PathFuncType",
    "MappingFunction",
    "MultiMappingFunction",
    "PixelArray",
    "GrayscalePixelArray",
    "RGBPixelArray",
    "RGBAPixelArray",
    "StrPath",
    "StrOrBytesPath",
]


"""
[CATEGORY]
Primitive data types
"""

ManimFloat: TypeAlias = np.float64
"""A double-precision floating-point value (64 bits, or 8 bytes),
according to the IEEE 754 standard.
"""

ManimInt: TypeAlias = np.int64
r"""A long integer (64 bits, or 8 bytes).

It can take values between :math:`-2^{63}` and :math:`+2^{63} - 1`,
which expressed in base 10 is a range between around
:math:`-9.223 \cdot 10^{18}` and :math:`+9.223 \cdot 10^{18}`.
"""


"""
[CATEGORY]
Color types
"""

ManimColorDType: TypeAlias = ManimFloat
"""Data type used in :class:`~.ManimColorInternal`: a
double-precision float between 0 and 1.
"""

FloatRGB: TypeAlias = npt.NDArray[ManimColorDType]
"""``shape: (3,)``

A :class:`numpy.ndarray` of 3 floats between 0 and 1, representing a
color in RGB format.

Its components describe, in order, the intensity of Red, Green, and
Blue in the represented color.
"""

FloatRGBLike: TypeAlias = FloatRGB | tuple[float, float, float]
"""``shape: (3,)``

An array of 3 floats between 0 and 1, representing a color in RGB
format.

This represents anything which can be converted to a :class:`.FloatRGB` NumPy
array.
"""

FloatRGB_Array: TypeAlias = npt.NDArray[ManimColorDType]
"""``shape: (M, 3)``

A :class:`numpy.ndarray` of many rows of 3 floats representing RGB colors.
"""

FloatRGBLike_Array: TypeAlias = FloatRGB_Array | Sequence[FloatRGBLike]
"""``shape: (M, 3)``

An array of many rows of 3 floats representing RGB colors.

This represents anything which can be converted to a :class:`.FloatRGB_Array` NumPy
array.
"""

IntRGB: TypeAlias = npt.NDArray[ManimInt]
"""``shape: (3,)``

A :class:`numpy.ndarray` of 3 integers between 0 and 255,
representing a color in RGB format.

Its components describe, in order, the intensity of Red, Green, and
Blue in the represented color.
"""

IntRGBLike: TypeAlias = IntRGB | tuple[int, int, int]
"""``shape: (3,)``

An array of 3 integers between 0 and 255, representing a color in RGB
format.

This represents anything which can be converted to an :class:`.IntRGB` NumPy
array.
"""

FloatRGBA: TypeAlias = npt.NDArray[ManimColorDType]
"""``shape: (4,)``

A :class:`numpy.ndarray` of 4 floats between 0 and 1, representing a
color in RGBA format.

Its components describe, in order, the intensity of Red, Green, Blue
and Alpha (opacity) in the represented color.
"""

FloatRGBALike: TypeAlias = FloatRGBA | tuple[float, float, float, float]
"""``shape: (4,)``

An array of 4 floats between 0 and 1, representing a color in RGBA
format.

This represents anything which can be converted to a :class:`.FloatRGBA` NumPy
array.
"""

FloatRGBA_Array: TypeAlias = npt.NDArray[ManimColorDType]
"""``shape: (M, 4)``

A :class:`numpy.ndarray` of many rows of 4 floats representing RGBA colors.
"""

FloatRGBALike_Array: TypeAlias = FloatRGBA_Array | Sequence[FloatRGBALike]
"""``shape: (M, 4)``

An array of many rows of 4 floats representing RGBA colors.

This represents anything which can be converted to a :class:`.FloatRGBA_Array` NumPy
array.
"""

IntRGBA: TypeAlias = npt.NDArray[ManimInt]
"""``shape: (4,)``

A :class:`numpy.ndarray` of 4 integers between 0 and 255,
representing a color in RGBA format.

Its components describe, in order, the intensity of Red, Green, Blue
and Alpha (opacity) in the represented color.
"""

IntRGBALike: TypeAlias = IntRGBA | tuple[int, int, int, int]
"""``shape: (4,)``

An array of 4 integers between 0 and 255, representing a color in RGBA
format.

This represents anything which can be converted to an :class:`.IntRGBA` NumPy
array.
"""

FloatHSV: TypeAlias = FloatRGB
"""``shape: (3,)``

A :class:`numpy.ndarray` of 3 floats between 0 and 1, representing a
color in HSV (or HSB) format.

Its components describe, in order, the Hue, Saturation and Value (or
Brightness) in the represented color.
"""

FloatHSVLike: TypeAlias = FloatRGBLike
"""``shape: (3,)``

An array of 3 floats between 0 and 1, representing a color in HSV (or
HSB) format.

This represents anything which can be converted to a :class:`.FloatHSV` NumPy
array.
"""

FloatHSVA: TypeAlias = FloatRGBA
"""``shape: (4,)``

A :class:`numpy.ndarray` of 4 floats between 0 and 1, representing a
color in HSVA (or HSBA) format.

Its components describe, in order, the Hue, Saturation and Value (or
Brightness) in the represented color.
"""

FloatHSVALike: TypeAlias = FloatRGBALike
"""``shape: (4,)``

An array of 4 floats between 0 and 1, representing a color in HSVA (or
HSBA) format.

This represents anything which can be converted to a :class:`.FloatHSVA` NumPy
array.
"""

FloatHSL: TypeAlias = FloatRGB
"""``shape: (3,)``

A :class:`numpy.ndarray` of 3 floats between 0 and 1, representing a
color in HSL format.

Its components describe, in order, the Hue, Saturation and Lightness
in the represented color.
"""

FloatHSLLike: TypeAlias = FloatRGBLike
"""``shape: (3,)``

An array of 3 floats between 0 and 1, representing a color in HSL format.

This represents anything which can be converted to a :class:`.FloatHSL` NumPy
array.
"""

ManimColorInternal: TypeAlias = FloatRGBA
"""``shape: (4,)``

Internal color representation used by :class:`~.ManimColor`,
following the RGBA format.

It is a :class:`numpy.ndarray` consisting of 4 floats between 0 and
1, describing respectively the intensities of Red, Green, Blue and
Alpha (opacity) in the represented color.
"""


"""
[CATEGORY]
Point types
"""

PointDType: TypeAlias = ManimFloat
"""Default type for arrays representing points: a double-precision
floating point value.
"""

Point2D: TypeAlias = npt.NDArray[PointDType]
"""``shape: (2,)``

A NumPy array representing a 2-dimensional point: ``[float, float]``.
"""

Point2DLike: TypeAlias = Point2D | tuple[float, float]
"""``shape: (2,)``

A 2-dimensional point: ``[float, float]``.

This represents anything which can be converted to a :class:`.Point2D` NumPy
array.
"""

Point2D_Array: TypeAlias = npt.NDArray[PointDType]
"""``shape: (M, 2)``

A NumPy array representing a sequence of :class:`.Point2D` objects:
``[[float, float], ...]``.
"""

Point2DLike_Array: TypeAlias = Point2D_Array | Sequence[Point2DLike]
"""``shape: (M, 2)``

An array of :class:`.Point2DLike` objects: ``[[float, float], ...]``.

This represents anything which can be converted to a :class:`.Point2D_Array`
NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

Point3D: TypeAlias = npt.NDArray[PointDType]
"""``shape: (3,)``

A NumPy array representing a 3-dimensional point: ``[float, float, float]``.
"""

Point3DLike: TypeAlias = Point3D | tuple[float, float, float]
"""``shape: (3,)``

A 3-dimensional point: ``[float, float, float]``.

This represents anything which can be converted to a :class:`.Point3D` NumPy
array.
"""

Point3D_Array: TypeAlias = npt.NDArray[PointDType]
"""``shape: (M, 3)``

A NumPy array representing a sequence of :class:`.Point3D` objects:
``[[float, float, float], ...]``.
"""

Point3DLike_Array: TypeAlias = Point3D_Array | Sequence[Point3DLike]
"""``shape: (M, 3)``

An array of :class:`.Point3DLike` objects: ``[[float, float, float], ...]``.

This represents anything which can be converted to a :class:`.Point3D_Array`
NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

PointND: TypeAlias = npt.NDArray[PointDType]
"""``shape: (N,)``

A NumPy array representing an N-dimensional point: ``[float, ...]``.
"""

PointNDLike: TypeAlias = PointND | Sequence[float]
"""``shape: (N,)``

An N-dimensional point: ``[float, ...]``.

This represents anything which can be converted to a :class:`.PointND` NumPy
array.
"""

PointND_Array: TypeAlias = npt.NDArray[PointDType]
"""``shape: (M, N)``

A NumPy array representing a sequence of :class:`.PointND` objects:
``[[float, ...], ...]``.
"""

PointNDLike_Array: TypeAlias = PointND_Array | Sequence[PointNDLike]
"""``shape: (M, N)``

An array of :class:`.PointNDLike` objects: ``[[float, ...], ...]``.

This represents anything which can be converted to a :class:`.PointND_Array`
NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""


"""
[CATEGORY]
Vector types
"""

Vector2D: TypeAlias = npt.NDArray[PointDType]
"""``shape: (2,)``

A NumPy array representing a 2-dimensional vector: ``[float, float]``.

.. caution::
    Do not confuse with the :class:`~.Vector` or :class:`~.Arrow`
    VMobjects!
"""

Vector2DLike: TypeAlias = npt.NDArray[PointDType] | tuple[float, float]
"""``shape: (2,)``

A 2-dimensional vector: ``[float, float]``.

This represents anything which can be converted to a :class:`.Vector2D` NumPy
array.

.. caution::
    Do not confuse with the :class:`~.Vector` or :class:`~.Arrow`
    VMobjects!
"""

Vector2D_Array: TypeAlias = npt.NDArray[PointDType]
"""``shape: (M, 2)``

A NumPy array representing a sequence of :class:`.Vector2D` objects:
``[[float, float], ...]``.
"""

Vector2DLike_Array: TypeAlias = Vector2D_Array | Sequence[Vector2DLike]
"""``shape: (M, 2)``

An array of :class:`.Vector2DLike` objects: ``[[float, float], ...]``.

This represents anything which can be converted to a :class:`.Vector2D_Array`
NumPy array.
"""

Vector3D: TypeAlias = npt.NDArray[PointDType]
"""``shape: (3,)``

A NumPy array representing a 3-dimensional vector: ``[float, float, float]``.

.. caution::
    Do not confuse with the :class:`~.Vector` or :class:`~.Arrow3D`
    VMobjects!
"""

Vector3DLike: TypeAlias = npt.NDArray[PointDType] | tuple[float, float, float]
"""``shape: (3,)``

A 3-dimensional vector: ``[float, float, float]``.

This represents anything which can be converted to a :class:`.Vector3D` NumPy
array.

.. caution::
    Do not confuse with the :class:`~.Vector` or :class:`~.Arrow3D`
    VMobjects!
"""

Vector3D_Array: TypeAlias = npt.NDArray[PointDType]
"""``shape: (M, 3)``

An NumPy array representing a sequence of :class:`.Vector3D` objects:
``[[float, float, float], ...]``.
"""

Vector3DLike_Array: TypeAlias = npt.NDArray[PointDType] | Sequence[Vector3DLike]
"""``shape: (M, 3)``

An array of :class:`.Vector3DLike` objects: ``[[float, float, float], ...]``.

This represents anything which can be converted to a :class:`.Vector3D_Array`
NumPy array.
"""

VectorND: TypeAlias = npt.NDArray[PointDType]
"""``shape (N,)``

A NumPy array representing an :math:`N`-dimensional vector: ``[float, ...]``.

.. caution::
    Do not confuse with the :class:`~.Vector` VMobject! This type alias
    is named "VectorND" instead of "Vector" to avoid potential name
    collisions.
"""

VectorNDLike: TypeAlias = npt.NDArray[PointDType] | Sequence[float]
"""``shape (N,)``

An :math:`N`-dimensional vector: ``[float, ...]``.

This represents anything which can be converted to a :class:`.VectorND` NumPy
array.

.. caution::
    Do not confuse with the :class:`~.Vector` VMobject! This type alias
    is named "VectorND" instead of "Vector" to avoid potential name
    collisions.
"""

VectorND_Array: TypeAlias = npt.NDArray[PointDType]
"""``shape (M, N)``

A NumPy array representing a sequence of :class:`.VectorND` objects:
``[[float, ...], ...]``.
"""

VectorNDLike_Array: TypeAlias = npt.NDArray[PointDType] | Sequence[VectorNDLike]
"""``shape (M, N)``

An array of :class:`.VectorNDLike` objects: ``[[float, ...], ...]``.

This represents anything which can be converted to a :class:`.VectorND_Array`
NumPy array.
"""

RowVector: TypeAlias = npt.NDArray[PointDType]
"""``shape: (1, N)``

A row vector: ``[[float, ...]]``.
"""

ColVector: TypeAlias = npt.NDArray[PointDType]
"""``shape: (N, 1)``

A column vector: ``[[float], [float], ...]``.
"""


"""
[CATEGORY]
Matrix types
"""

MatrixMN: TypeAlias = npt.NDArray[PointDType]
"""``shape: (M, N)``

A matrix: ``[[float, ...], [float, ...], ...]``.
"""

Zeros: TypeAlias = MatrixMN
"""``shape: (M, N)``

A :class:`.MatrixMN` filled with zeros, typically created with
``numpy.zeros((M, N))``.
"""


"""
[CATEGORY]
Bézier types
"""

QuadraticBezierPoints: TypeAlias = Point3D_Array
"""``shape: (3, 3)``

A :class:`.Point3D_Array` of three 3D control points for a single quadratic Bézier
curve:
``[[float, float, float], [float, float, float], [float, float, float]]``.
"""

QuadraticBezierPointsLike: TypeAlias = (
    QuadraticBezierPoints | tuple[Point3DLike, Point3DLike, Point3DLike]
)
"""``shape: (3, 3)``

A :class:`.Point3DLike_Array` of three 3D control points for a single quadratic Bézier
curve:
``[[float, float, float], [float, float, float], [float, float, float]]``.

This represents anything which can be converted to a
:class:`.QuadraticBezierPoints` NumPy array.
"""

QuadraticBezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
"""``shape: (N, 3, 3)``

A NumPy array containing :math:`N` :class:`.QuadraticBezierPoints` objects:
``[[[float, float, float], [float, float, float], [float, float, float]], ...]``.
"""

QuadraticBezierPointsLike_Array: TypeAlias = (
    QuadraticBezierPoints_Array | Sequence[QuadraticBezierPointsLike]
)
"""``shape: (N, 3, 3)``

A sequence of :math:`N` :class:`.QuadraticBezierPointsLike` objects:
``[[[float, float, float], [float, float, float], [float, float, float]], ...]``.

This represents anything which can be converted to a
:class:`.QuadraticBezierPoints_Array` NumPy array.
"""

QuadraticBezierPath: TypeAlias = Point3D_Array
"""``shape: (3*N, 3)``

A :class:`.Point3D_Array` of :math:`3N` points, where each one of the
:math:`N` consecutive blocks of 3 points represents a quadratic
Bézier curve:
``[[float, float, float], ...], ...]``.

Please refer to the documentation of the function you are using for
further type information.
"""

QuadraticBezierPathLike: TypeAlias = Point3DLike_Array
"""``shape: (3*N, 3)``

A :class:`.Point3DLike_Array` of :math:`3N` points, where each one of the
:math:`N` consecutive blocks of 3 points represents a quadratic
Bézier curve:
``[[float, float, float], ...], ...]``.

This represents anything which can be converted to a
:class:`.QuadraticBezierPath` NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

QuadraticSpline: TypeAlias = QuadraticBezierPath
"""``shape: (3*N, 3)``

A special case of :class:`.QuadraticBezierPath` where all the :math:`N`
quadratic Bézier curves are connected, forming a quadratic spline:
``[[float, float, float], ...], ...]``.

Please refer to the documentation of the function you are using for
further type information.
"""

QuadraticSplineLike: TypeAlias = QuadraticBezierPathLike
"""``shape: (3*N, 3)``

A special case of :class:`.QuadraticBezierPathLike` where all the :math:`N`
quadratic Bézier curves are connected, forming a quadratic spline:
``[[float, float, float], ...], ...]``.

This represents anything which can be converted to a :class:`.QuadraticSpline`
NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

CubicBezierPoints: TypeAlias = Point3D_Array
"""``shape: (4, 3)``

A :class:`.Point3D_Array` of four 3D control points for a single cubic Bézier curve:
``[[float, float, float], [float, float, float], [float, float, float], [float, float, float]]``.
"""

CubicBezierPointsLike: TypeAlias = (
    CubicBezierPoints | tuple[Point3DLike, Point3DLike, Point3DLike, Point3DLike]
)
"""``shape: (4, 3)``

A :class:`.Point3DLike_Array` of 4 control points for a single cubic Bézier curve:
``[[float, float, float], [float, float, float], [float, float, float], [float, float, float]]``.

This represents anything which can be converted to a :class:`.CubicBezierPoints`
NumPy array.
"""

CubicBezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
"""``shape: (N, 4, 3)``

A NumPy array containing :math:`N` :class:`.CubicBezierPoints` objects:
``[[[float, float, float], [float, float, float], [float, float, float], [float, float, float]], ...]``.
"""

CubicBezierPointsLike_Array: TypeAlias = (
    CubicBezierPoints_Array | Sequence[CubicBezierPointsLike]
)
"""``shape: (N, 4, 3)``

A sequence of :math:`N` :class:`.CubicBezierPointsLike` objects:
``[[[float, float, float], [float, float, float], [float, float, float], [float, float, float]], ...]``.

This represents anything which can be converted to a
:class:`.CubicBezierPoints_Array` NumPy array.
"""

CubicBezierPath: TypeAlias = Point3D_Array
"""``shape: (4*N, 3)``

A :class:`.Point3D_Array` of :math:`4N` points, where each one of the
:math:`N` consecutive blocks of 4 points represents a cubic Bézier
curve:
``[[float, float, float], ...], ...]``.

Please refer to the documentation of the function you are using for
further type information.
"""

CubicBezierPathLike: TypeAlias = Point3DLike_Array
"""``shape: (4*N, 3)``

A :class:`.Point3DLike_Array` of :math:`4N` points, where each one of the
:math:`N` consecutive blocks of 4 points represents a cubic Bézier
curve:
``[[float, float, float], ...], ...]``.

This represents anything which can be converted to a
:class:`.CubicBezierPath` NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

CubicSpline: TypeAlias = CubicBezierPath
"""``shape: (4*N, 3)``

A special case of :class:`.CubicBezierPath` where all the :math:`N` cubic
Bézier curves are connected, forming a quadratic spline:
``[[float, float, float], ...], ...]``.

Please refer to the documentation of the function you are using for
further type information.
"""

CubicSplineLike: TypeAlias = CubicBezierPathLike
"""``shape: (4*N, 3)``

A special case of :class:`.CubicBezierPath` where all the :math:`N` cubic
Bézier curves are connected, forming a quadratic spline:
``[[float, float, float], ...], ...]``.

This represents anything which can be converted to a
:class:`.CubicSpline` NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

BezierPoints: TypeAlias = Point3D_Array
r"""``shape: (PPC, 3)``

A :class:`.Point3D_Array` of :math:`\text{PPC}` control points
(:math:`\text{PPC: Points Per Curve} = n + 1`) for a single
:math:`n`-th degree Bézier curve:
``[[float, float, float], ...]``.

Please refer to the documentation of the function you are using for
further type information.
"""

BezierPointsLike: TypeAlias = Point3DLike_Array
r"""``shape: (PPC, 3)``

A :class:`.Point3DLike_Array` of :math:`\text{PPC}` control points
(:math:`\text{PPC: Points Per Curve} = n + 1`) for a single
:math:`n`-th degree Bézier curve:
``[[float, float, float], ...]``.

This represents anything which can be converted to a
:class:`.BezierPoints` NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

BezierPoints_Array: TypeAlias = npt.NDArray[PointDType]
r"""``shape: (N, PPC, 3)``

A NumPy array of :math:`N` :class:`.BezierPoints` objects containing
:math:`\text{PPC}` :class:`.Point3D` objects each
(:math:`\text{PPC: Points Per Curve} = n + 1`):
``[[[float, float, float], ...], ...]``.

Please refer to the documentation of the function you are using for
further type information.
"""

BezierPointsLike_Array: TypeAlias = BezierPoints_Array | Sequence[BezierPointsLike]
r"""``shape: (N, PPC, 3)``

A sequence of :math:`N` :class:`.BezierPointsLike` objects containing
:math:`\text{PPC}` :class:`.Point3DLike` objects each
(:math:`\text{PPC: Points Per Curve} = n + 1`):
``[[[float, float, float], ...], ...]``.

This represents anything which can be converted to a
:class:`.BezierPoints_Array` NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

BezierPath: TypeAlias = Point3D_Array
r"""``shape: (PPC*N, 3)``

A :class:`.Point3D_Array` of :math:`\text{PPC} \cdot N` points, where each
one of the :math:`N` consecutive blocks of :math:`\text{PPC}` control
points (:math:`\text{PPC: Points Per Curve} = n + 1`) represents a
Bézier curve of :math:`n`-th degree:
``[[float, float, float], ...], ...]``.

Please refer to the documentation of the function you are using for
further type information.
"""

BezierPathLike: TypeAlias = Point3DLike_Array
r"""``shape: (PPC*N, 3)``

A :class:`.Point3DLike_Array` of :math:`\text{PPC} \cdot N` points, where each
one of the :math:`N` consecutive blocks of :math:`\text{PPC}` control
points (:math:`\text{PPC: Points Per Curve} = n + 1`) represents a
Bézier curve of :math:`n`-th degree:
``[[float, float, float], ...], ...]``.

This represents anything which can be converted to a
:class:`.BezierPath` NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

Spline: TypeAlias = BezierPath
r"""``shape: (PPC*N, 3)``

A special case of :class:`.BezierPath` where all the :math:`N` Bézier curves
consisting of :math:`\text{PPC}` :class:`.Point3D` objects
(:math:`\text{PPC: Points Per Curve} = n + 1`) are connected, forming
an :math:`n`-th degree spline:
``[[float, float, float], ...], ...]``.

Please refer to the documentation of the function you are using for
further type information.
"""

SplineLike: TypeAlias = BezierPathLike
r"""``shape: (PPC*N, 3)``

A special case of :class:`.BezierPathLike` where all the :math:`N` Bézier curves
consisting of :math:`\text{PPC}` :class:`.Point3D` objects
(:math:`\text{PPC: Points Per Curve} = n + 1`) are connected, forming
an :math:`n`-th degree spline:
``[[float, float, float], ...], ...]``.

This represents anything which can be converted to a
:class:`.Spline` NumPy array.

Please refer to the documentation of the function you are using for
further type information.
"""

FlatBezierPoints: TypeAlias = npt.NDArray[PointDType] | tuple[float, ...]
"""``shape: (3*PPC*N,)``

A flattened array of Bézier control points:
``[float, ...]``.
"""


"""
[CATEGORY]
Function types
"""

# Due to current limitations
# (see https://github.com/python/mypy/issues/14656 / 8263),
# we don't specify the first argument type (Mobject).
# Nor are we able to specify the return type (Animation) since we cannot import
# that here.
FunctionOverride: TypeAlias = Callable
"""Function type returning an :class:`~.Animation` for the specified
:class:`~.Mobject`.
"""

PathFuncType: TypeAlias = Callable[[Point3DLike, Point3DLike, float], Point3DLike]
"""Function mapping two :class:`.Point3D` objects and an alpha value to a new
:class:`.Point3D`.
"""

MappingFunction: TypeAlias = Callable[[Point3D], Point3D]
"""A function mapping a :class:`.Point3D` to another :class:`.Point3D`."""

MultiMappingFunction: TypeAlias = Callable[[Point3D_Array], Point3D_Array]
"""A function mapping a :class:`.Point3D_Array` to another
:class:`.Point3D_Array`.
"""

"""
[CATEGORY]
Image types
"""

PixelArray: TypeAlias = npt.NDArray[ManimInt]
"""``shape: (height, width) | (height, width, 3) | (height, width, 4)``

A rasterized image with a height of ``height`` pixels and a width of
``width`` pixels.

Every value in the array is an integer from 0 to 255.

Every pixel is represented either by a single integer indicating its
lightness (for greyscale images), an :class:`.RGB_Array_Int` or an
`RGBA_Array_Int`.
"""

GrayscalePixelArray: TypeAlias = PixelArray
"""``shape: (height, width)``

A 100% opaque grayscale :class:`.PixelArray`, where every pixel value is a
`ManimInt` indicating its lightness (black -> gray -> white).
"""

RGBPixelArray: TypeAlias = PixelArray
"""``shape: (height, width, 3)``

A 100% opaque :class:`.PixelArray` in color, where every pixel value is an
`RGB_Array_Int` object.
"""

RGBAPixelArray: TypeAlias = PixelArray
"""``shape: (height, width, 4)``

A :class:`.PixelArray` in color where pixels can be transparent. Every pixel
value is an :class:`.RGBA_Array_Int` object.
"""


"""
[CATEGORY]
Path types
"""

StrPath: TypeAlias = str | PathLike[str]
"""A string or :class:`.os.PathLike` representing a path to a
directory or file.
"""

StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
"""A string, bytes or :class:`.os.PathLike` object representing a path
to a directory or file.
"""
