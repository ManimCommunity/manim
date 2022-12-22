from typing import Dict, List

from .core import *
from .manim_colors import *

_colors: List[ManimColor] = [x for x in globals().values() if isinstance(x, ManimColor)]

from . import AS2700, BS381, X11, XKCD

_all_color_dict: Dict[str, ManimColor] = {
    k: v for k, v in globals().items() if isinstance(v, ManimColor)
}
