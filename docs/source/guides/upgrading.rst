Upgrading from Manim 0.22
=========================

This page lists what code written for Manim 0.22 needs to change.

Mobject arrays are replaced, not changed in place
-------------------------------------------------

The arrays that mobjects are drawn from can be replaced, but not changed in place:
``points``, the color arrays of vectorized mobjects (``fill_rgbas``,
``stroke_rgbas``, ``background_stroke_rgbas``) and their ``sheen_direction``, and
the ``pixel_array`` of an :class:`.ImageMobject`. Renderers keep track of which
mobjects changed by the assignments to these arrays, so changing one in place
raises numpy's error ``ValueError: assignment destination is read-only``:

.. code-block:: python

    mob.points[0] = ORIGIN  # raises
    mob.points += RIGHT  # raises

Assign a new array instead:

.. code-block:: python

    points = mob.points.copy()
    points[0] = ORIGIN
    mob.points = points

    mob.points = mob.points + RIGHT

The assigned array is copied, so changing it afterwards does not change the
mobject. To give the array another shape, assign a reshaped copy
(``mob.points = mob.points.reshape(...)``); changing ``shape`` or ``dtype`` of
the array itself is not supported.

The same holds for the other properties mobjects are drawn from, such as
``stroke_width`` or ``z_index``. A subclass that sets one of them in its class
body raises a ``TypeError`` as soon as the class is defined:

.. code-block:: python

    class ThickLine(Line):
        stroke_width = 8  # raises TypeError

Set the value in ``__init__`` instead:

.. code-block:: python

    class ThickLine(Line):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stroke_width = 8

Removed 3D shading options
--------------------------

Shading of vectorized mobjects in 3D is drawn from the mobjects' own fields now. The
options that only served the old shading code are gone, without a deprecation
period; passing them raises ``TypeError``:

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``ThreeDAxes(light_source=...)``, ``ThreeDAxes(gloss=...)``
     - Nothing: the axes are drawn in their own colors, without a sheen
   * - ``set_shade_in_3d(value, z_index_as_group=True)``, ``z_index_group``
     - ``set_shade_in_3d(value)``; each path is ordered by its own points

Stroke widths are numbers
-------------------------

``stroke_width`` and ``background_stroke_width`` of vectorized mobjects accept only
numbers and are stored as floats; assigning a string, ``None`` or a bool raises
``TypeError``. Negative widths are stored as 0, so reading the attribute gives the
same value as :meth:`.VMobject.get_stroke_width`.

Fixed mobjects are marked on the mobjects
-----------------------------------------

Whether a mobject is fixed in the frame or keeps facing the camera is stored on
the mobject and its family: read ``mob.fixed_in_frame`` and
``mob.fixed_orientation``. :class:`.ThreeDCamera` no longer keeps its own records,
so its ``fixed_in_frame_mobjects`` and ``fixed_orientation_mobjects`` attributes
are gone.
