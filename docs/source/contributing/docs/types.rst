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

   def status2D(coord: Point2D) -> None:
       x, y = coord
       print(f"Point at {x=},{y=}")


   def status3D(coord: Point3D) -> None:
       x, y, z = coord
       print(f"Point at {x=},{y=},{z=}")


   def get_statuses(coords: Point2D_Array | Point3D_Array) -> None:
       for coord in coords:
           if len(coord) == 2:
               # it's a Point2D
               status2D(coord)
           else:
               # it's a point3D
               status3D(coord)

It's important to realize that the status functions accepted both
tuples/lists of the correct length, and ``NDArray``'s of the correct shape.
If they only accepted ``NDArray``'s, we would use their ``Internal`` counterparts:
:class:`~.typing.InternalPoint2D`, :class:`~.typing.InternalPoint3D`, :class:`~.typing.InternalPoint2D_Array` and :class:`~.typing.InternalPoint3D_Array`.

In general, the type aliases prefixed with ``Internal`` should never be used on
user-facing classes and functions, but should be reserved for internal behavior.

Vectors
~~~~~~~
Vectors share many similarities to points. However, they have a different
connotation. Vectors should be used to represent direction. For example,
consider this slightly contrived function:

.. code-block:: python

   def shift_mobject(mob: Mobject, direction: Vector3D, scale_factor: float = 1) -> mob:
       return mob.shift(direction * scale_factor)

Here we see an important example of the difference. ``direction`` can not, and
should not, be typed as a :class:`~.typing.Point3D` because the function does not accept tuples/lists,
like ``direction=(0, 1, 0)``. You could type it as :class:`~.typing.InternalPoint3D` and
the type checker and linter would be happy; however, this makes the code harder
to understand.

As a general rule, if a parameter is called ``direction`` or ``axis``,
it should be type hinted as some form of :class:`~.VectorND`.

.. warning::

   This is not always true. For example, as of Manim 0.18.0, the direction
   parameter of the :class:`.Vector` Mobject should be ``Point2D | Point3D``,
   as it can also accept ``tuple[float, float]`` and ``tuple[float, float, float]``.

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

This is the use case of the :class:`~.typing.Image` type hint. Sometimes, Manim may use ``PIL.Image``,
in which case one should use that type hint instead.
Of course, if a more specific type of image is needed, it can be annotated as such.
