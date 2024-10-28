"""Utilities for working with colors and predefined color constants.

Color data structure
--------------------

.. autosummary::
   :toctree: ../reference

   core


Predefined colors
-----------------

There are several predefined colors available in Manim:

- The colors listed in :mod:`.color.manim_colors` are loaded into
  Manim's global name space.
- The colors in :mod:`.color.AS2700`, :mod:`.color.BS381`,
  :mod:`.color.DVIPSNAMES`, :mod:`.color.SVGNAMES`, :mod:`.color.X11` and
  :mod:`.color.XKCD` need to be accessed via their module (which are available
  in Manim's global name space), or imported separately. For example:

  .. code:: pycon

     >>> from manim import XKCD
     >>> XKCD.AVOCADO
     ManimColor('#90B134')

  Or, alternatively:

  .. code:: pycon

     >>> from manim.utils.color.XKCD import AVOCADO
     >>> AVOCADO
     ManimColor('#90B134')

The following modules contain the predefined color constants:

.. autosummary::
   :toctree: ../reference

   manim_colors
   AS2700
   BS381
   DVIPSNAMES
   SVGNAMES
   XKCD
   X11

"""

from __future__ import annotations

from . import AS2700, BS381, DVIPSNAMES, SVGNAMES, X11, XKCD
from .core import *
from .manim_colors import *

_all_color_dict: dict[str, ManimColor] = {
    k: v for k, v in globals().items() if isinstance(v, ManimColor)
}
