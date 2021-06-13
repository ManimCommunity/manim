try:
    from dearpygui import core as dpg
except ImportError:
    pass

from ..mobject.opengl_geometry import *
from ..mobject.opengl_mobject import *
from ..mobject.opengl_three_dimensions import *
from ..mobject.svg.opengl_svg_mobject import *
from ..mobject.svg.opengl_tex_mobject import *
from ..mobject.svg.opengl_text_mobject import *
from ..mobject.types.opengl_surface import *
from ..mobject.types.opengl_vectorized_mobject import *
from ..renderer.shader import *
from ..utils.opengl import *
