from ...constants import *
from ..svg.svg_mobject import SVGMobject
from ..types.opengl_vectorized_mobject import OpenGLVMobject
from .opengl_svg_path import OpenGLSVGPathMobject
from .style_utils import cascade_element_style, parse_style


class OpenGLSVGMobject(SVGMobject):
    def __init__(
        self,
        file_name=None,
        should_center=True,
        height=2,
        width=None,
        unpack_groups=True,  # if False, creates a hierarchy of VGroups
        stroke_width=DEFAULT_STROKE_WIDTH,
        fill_opacity=1.0,
        should_subdivide_sharp_curves=False,
        should_remove_null_curves=False,
        **kwargs,
    ):
        self.def_map = {}
        self.file_name = file_name or self.file_name
        self.ensure_valid_file()
        self.should_center = should_center
        self.unpack_groups = unpack_groups
        self.path_string_config = {
            "should_subdivide_sharp_curves": should_subdivide_sharp_curves,
            "should_remove_null_curves": should_remove_null_curves,
        }
        OpenGLVMobject.__init__(
            self, stroke_width=stroke_width, fill_opacity=fill_opacity, **kwargs
        )
        self.move_into_position(width, height)

    def init_points(self):
        self.generate_points()

    def path_string_to_mobject(self, path_string: str, style: dict):
        return OpenGLSVGPathMobject(
            path_string, **self.path_string_config, **parse_style(style)
        )
