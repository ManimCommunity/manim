Units
#####

Manim uses *Manim units* for lengths in a scene. The visible frame is
``config.frame_width`` Manim units wide and ``config.frame_height`` Manim units
tall. The :mod:`manim.utils.unit` module provides helpers for converting common
measurements into these scene units.

The helpers are exposed through the ``unit`` namespace:

.. code-block:: pycon

   >>> from manim import *
   >>> 50 * unit.Pixels
   0.37037037037037035
   >>> 90 * unit.Degrees
   1.5707963267948966
   >>> 10 * unit.Percent(X_AXIS)
   1.4222222222222223

Pixels
******

``unit.Pixels`` converts a number of screen pixels to Manim units. This is
useful when a value should stay tied to the rendered pixel resolution instead of
to a mathematical coordinate.

.. code-block:: pycon

   >>> from manim import *
   >>> config.pixel_width
   1920
   >>> config.frame_width
   14.222222222222221
   >>> 50 * unit.Pixels
   0.37037037037037035

The conversion depends on the current frame and pixel configuration, so changing
``config.frame_width`` or ``config.pixel_width`` changes the value returned by
``unit.Pixels``.

Degrees
*******

Angles in Manim are given in radians. ``unit.Degrees`` converts degrees to
radians:

.. code-block:: pycon

   >>> from manim import *
   >>> 45 * unit.Degrees
   0.7853981633974483
   >>> 180 * unit.Degrees == PI
   True

Percent
*******

``unit.Percent`` converts a percentage of the current frame width or height to
Manim units. Pass ``X_AXIS`` to use ``config.frame_width`` and ``Y_AXIS`` to use
``config.frame_height``:

.. code-block:: pycon

   >>> from manim import *
   >>> 10 * unit.Percent(X_AXIS)
   1.4222222222222223
   >>> 25 * unit.Percent(Y_AXIS)
   2.0

The Z axis does not have a frame length, so ``unit.Percent(Z_AXIS)`` is not
defined.

Munits
******

``unit.Munits`` represents a Manim unit directly. It is equal to ``1`` and can
be used when mixing explicit unit conversions in the same expression:

.. code-block:: pycon

   >>> from manim import *
   >>> 3 * unit.Munits
   3
