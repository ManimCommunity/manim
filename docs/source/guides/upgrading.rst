Upgrading from Manim 0.22
=========================

This page lists changes that need changes in code written for Manim 0.22.

Mobject arrays are read-only
----------------------------

The ``points`` of a mobject are read-only. Renderers keep track of which
mobjects changed by the assignments to ``points``, so writing into the array in
place raises ``ValueError: assignment destination is read-only``:

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

A subclass that sets ``points`` in its class body raises a ``TypeError`` when it
is defined; set the value in ``__init__`` instead.
