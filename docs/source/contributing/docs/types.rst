==================
Choosing Typehints
==================
In order to provide the best user experience,
it's important that typehints are chosen correctly.
With the large variety of types provided by Manim, choosing
which one to use can be difficult. This guide aims to
aid you in the process of choosing the right type for the scenario.


The first step is figuring out which category your typehint fits into.

Coordinates
-----------
Coordinates encompasses two "main" categories: points, and vectors.


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

It's important to realize that the status functions worked with both
tuples/lists of the correct length, and ``NDArray``'s of the correct shape.
If they only worked with ``NDArray``'s, we would use their "Internal" counterparts:
``InternalPoint2D``, ``InternalPoint3D``, ``InternalPoint2D_Array`` and ``InternalPoint3D_Array``.

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
should not, be typed as a ``Point3D`` because it does not work with something
like ``direction=(0, 1, 0)``. You could type it as ``InternalPoint3D`` and
the typechecker and linter would be happy; however, this makes the code harder
to understand.

As a general rule, if a parameter is called ``direction`` or ``axis``,
it should be typehinted as some form of ``Vector``.

.. warning::

   This is not always true. For example as of Manim 0.18.0, the direction
   parameter of the Vector mobject should be ``Point2D | Point3D``
   as it can also accept ``tuple[float, float]`` and ``tuple[float, float, float]``.

Colors
------
The interface manim provides for working with colors is :class:`.ManimColor`.
The main color types Manim supports are RGB, RGBA, and HSV. You will want
to typehint a function depending on which type it uses. If any color will work,
you will need something like:

.. code-block:: python

   if TYPE_CHECKING:
       from manim.utils.color import ParsableManimColor

   # typehint stuff with ParsableManimColor



Béziers
-------


Functions
---------
