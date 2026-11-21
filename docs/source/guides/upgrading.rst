Upgrading from Manim 0.22
=========================

This page lists changes that need changes in code written for Manim 0.22.

Mobject arrays are read-only
----------------------------

The arrays that mobjects are drawn from are read-only: ``points``, the color
arrays of vectorized mobjects (``fill_rgbas``, ``stroke_rgbas``,
``background_stroke_rgbas``) and their ``sheen_direction``, and the
``pixel_array`` of an :class:`.ImageMobject`. Renderers keep track of which
mobjects changed by the assignments to these arrays, so writing into one in place
raises ``ValueError: assignment destination is read-only``:

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
``stroke_width`` or ``z_index``: a subclass that sets one of them in its class
body, for example ``stroke_width = 8``, raises a ``TypeError`` when it is
defined; set the value in ``__init__`` instead.
