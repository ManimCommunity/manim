r"""dvips Colors

This module contains the colors defined in the dvips driver, which are commonly accessed
as named colors in LaTeX via the ``\usepackage[dvipsnames]{xcolor}`` package.

To use the colors from this list, access them directly from the module (which
is exposed to Manim's global name space):

.. code:: pycon

    >>> from manim import DVIPSNAMES
    >>> DVIPSNAMES.DARKORCHID
    ManimColor('#A4538A')

List of Color Constants
-----------------------

These hex values are derived from those specified in the ``xcolor`` package
documentation (see https://ctan.org/pkg/xcolor):

.. automanimcolormodule:: manim.utils.color.DVIPSNAMES

"""

from __future__ import annotations

from .core import ManimColor

AQUAMARINE = ManimColor("#00B5BE")
BITTERSWEET = ManimColor("#C04F17")
APRICOT = ManimColor("#FBB982")
BLACK = ManimColor("#221E1F")
BLUE = ManimColor("#2D2F92")
BLUEGREEN = ManimColor("#00B3B8")
BLUEVIOLET = ManimColor("#473992")
BRICKRED = ManimColor("#B6321C")
BROWN = ManimColor("#792500")
BURNTORANGE = ManimColor("#F7921D")
CADETBLUE = ManimColor("#74729A")
CARNATIONPINK = ManimColor("#F282B4")
CERULEAN = ManimColor("#00A2E3")
CORNFLOWERBLUE = ManimColor("#41B0E4")
CYAN = ManimColor("#00AEEF")
DANDELION = ManimColor("#FDBC42")
DARKORCHID = ManimColor("#A4538A")
EMERALD = ManimColor("#00A99D")
FORESTGREEN = ManimColor("#009B55")
FUCHSIA = ManimColor("#8C368C")
GOLDENROD = ManimColor("#FFDF42")
GRAY = ManimColor("#949698")
GREEN = ManimColor("#00A64F")
GREENYELLOW = ManimColor("#DFE674")
JUNGLEGREEN = ManimColor("#00A99A")
LAVENDER = ManimColor("#F49EC4")
LIMEGREEN = ManimColor("#8DC73E")
MAGENTA = ManimColor("#EC008C")
MAHOGANY = ManimColor("#A9341F")
MAROON = ManimColor("#AF3235")
MELON = ManimColor("#F89E7B")
MIDNIGHTBLUE = ManimColor("#006795")
MULBERRY = ManimColor("#A93C93")
NAVYBLUE = ManimColor("#006EB8")
OLIVEGREEN = ManimColor("#3C8031")
ORANGE = ManimColor("#F58137")
ORANGERED = ManimColor("#ED135A")
ORCHID = ManimColor("#AF72B0")
PEACH = ManimColor("#F7965A")
PERIWINKLE = ManimColor("#7977B8")
PINEGREEN = ManimColor("#008B72")
PLUM = ManimColor("#92268F")
PROCESSBLUE = ManimColor("#00B0F0")
PURPLE = ManimColor("#99479B")
RAWSIENNA = ManimColor("#974006")
RED = ManimColor("#ED1B23")
REDORANGE = ManimColor("#F26035")
REDVIOLET = ManimColor("#A1246B")
RHODAMINE = ManimColor("#EF559F")
ROYALBLUE = ManimColor("#0071BC")
ROYALPURPLE = ManimColor("#613F99")
RUBINERED = ManimColor("#ED017D")
SALMON = ManimColor("#F69289")
SEAGREEN = ManimColor("#3FBC9D")
SEPIA = ManimColor("#671800")
SKYBLUE = ManimColor("#46C5DD")
SPRINGGREEN = ManimColor("#C6DC67")
TAN = ManimColor("#DA9D76")
TEALBLUE = ManimColor("#00AEB3")
THISTLE = ManimColor("#D883B7")
TURQUOISE = ManimColor("#00B4CE")
VIOLET = ManimColor("#58429B")
VIOLETRED = ManimColor("#EF58A0")
WHITE = ManimColor("#FFFFFF")
WILDSTRAWBERRY = ManimColor("#EE2967")
YELLOW = ManimColor("#FFF200")
YELLOWGREEN = ManimColor("#98CC70")
YELLOWORANGE = ManimColor("#FAA21A")
