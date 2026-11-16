===================
Choosing Type Hints
===================
In order to provide the best user experience,
it's important that type hints are chosen correctly.
With the large variety of types provided by Manim, choosing
which one to use can be difficult. This guide aims to
aid you in the process of choosing the right type for the scenario.


The first step is figuring out which category your type hint fits into.

Coordinates
-----------
Coordinates encompass two main categories: points, and vectors.


Points
~~~~~~
The purpose of points is pretty straightforward: they represent a point
in space. For example:

.. code-block:: python

   def print_point2D(coord: Point2DLike) -> None:
       x, y = coord
       print(f"Point at {x=},{y=}")


   def print_point3D(coord: Point3DLike) -> None:
       x, y, z = coord
       print(f"Point at {x=},{y=},{z=}")


   def print_point_array(coords: Point2DLike_Array | Point3DLike_Array) -> None:
       for coord in coords:
           if len(coord) == 2:
               # it's a Point2DLike
               print_point2D(coord)
           else:
               # it's a Point3DLike
               print_point3D(coord)

   def shift_point_up(coord: Point3DLike) -> Point3D:
       result = np.asarray(coord)
       result += UP
       print(f"New point: {result}")
       return result

Notice that the last function, ``shift_point_up()``, accepts a
:class:`~.Point3DLike` as a parameter and returns a :class:`~.Point3D`. A
:class:`~.Point3D` always represents a NumPy array consisting of 3 floats,
whereas a :class:`~.Point3DLike` can represent anything resembling a 3D point:
either a NumPy array or a tuple/list of 3 floats, hence the ``Like`` word. The
same happens with :class:`~.Point2D`, :class:`~.Point2D_Array` and
:class:`~.Point3D_Array`, and their ``Like`` counterparts
:class:`~.Point2DLike`, :class:`~.Point2DLike_Array` and
:class:`~.Point3DLike_Array`.

The rule for typing functions is: **make parameter types as broad as possible,
and return types as specific as possible.** Therefore, for functions which are
intended to be called by users, **we should always, if possible, accept**
``Like`` **types as parameters and return NumPy, non-** ``Like`` **types.** The
main reason is to be more flexible with users who might want to pass tuples or
lists as arguments rather than NumPy arrays, because it's more convenient. The
last function, ``shift_point_up()``, is an example of it.

Internal functions which are *not* meant to be called by users may accept
non-``Like`` parameters if necessary.

Vectors
~~~~~~~
Vectors share many similarities to points. However, they have a different
connotation. Vectors should be used to represent direction. For example,
consider this slightly contrived function:

.. code-block:: python

   M = TypeVar("M", bound=Mobject)  # allow any mobject
   def shift_mobject(mob: M, direction: Vector3D, scale_factor: float = 1) -> M:
       return mob.shift(direction * scale_factor)

Here we see an important example of the difference. ``direction`` should not be
typed as a :class:`~.Point3D`, because it represents a direction along
which to shift a :class:`~.Mobject`, not a position in space.

As a general rule, if a parameter is called ``direction`` or ``axis``,
it should be type hinted as some form of :class:`~.VectorND` or
:class:`~.VectorNDLike`.

Colors
------
The interface Manim provides for working with colors is :class:`.ManimColor`.
The main color types Manim supports are RGB, RGBA, and HSV. You will want
to add type hints to a function depending on which type it uses. If any color will work,
you will need something like:

.. code-block:: python

   if TYPE_CHECKING:
       from manim.utils.color import ParsableManimColor

   # type hint stuff with ParsableManimColor



Béziers
-------
Manim internally represents a :class:`.Mobject` by a collection of points. In the case of :class:`.VMobject`,
the most commonly used subclass of :class:`.Mobject`, these points represent Bézier curves,
which are a way of representing a curve using a sequence of points.

.. note::

   To learn more about Béziers, take a look at https://pomax.github.io/bezierinfo/


Manim supports two different renderers, which each have different representations of
Béziers: Cairo uses cubic Bézier curves, while OpenGL uses quadratic Bézier curves.

Type hints like :class:`~.typing.BezierPoints` represent a single bezier curve, and :class:`~.typing.BezierPath`
represents multiple Bézier curves. A :class:`~.typing.Spline` is when the Bézier curves in a :class:`~.typing.BezierPath`
forms a single connected curve. Manim also provides more specific type aliases when working with
quadratic or cubic curves, and they are prefixed with their respective type (e.g. :class:`~.typing.CubicBezierPoints`,
is a :class:`~.typing.BezierPoints` consisting of exactly 4 points representing a cubic Bézier curve).


Functions
---------
Throughout the codebase, many different types of functions are used. The most obvious example
is a rate function, which takes in a float and outputs a float (``Callable[[float], float]``).
Another example is for overriding animations. One will often need to map a :class:`.Mobject`
to an overridden :class:`.Animation`, and for that we have the :class:`~.typing.FunctionOverride` type hint.

:class:`~.typing.PathFuncType` and :class:`~.typing.MappingFunction` are more niche, but are related to moving objects
along a path, or applying functions. If you need to use it, you'll know.


Images
------
There are several representations of images in Manim. The most common is
the representation as a NumPy array of floats representing the pixels of an image.
This is especially common when it comes to the OpenGL renderer.

This is the use case of the :class:`~.typing.PixelArray` type hint. Sometimes, Manim may use ``PIL.Image.Image``,
which is not the same as :class:`~.typing.PixelArray`. In this case, use the ``PIL.Image.Image`` typehint.
Of course, if a more specific type of image is needed, it can be annotated as such.
