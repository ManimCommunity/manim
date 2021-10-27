#!/usr/bin/env python


import pkg_resources

__version__ = pkg_resources.get_distribution(__name__).version


import sys

# Importing the config module should be the first thing we do, since other
# modules depend on the global config dict for initialization.
from ._config import *

# Workaround to set the renderer passed via CLI args *before* importing
# Manim's classes (as long as the metaclass approach for switching
# between OpenGL and cairo rendering is in place, classes depend
# on the value of config.renderer).
for i, arg in enumerate(sys.argv):
    if arg.startswith("--renderer"):
        if "=" in arg:
            _, parsed_renderer = arg.split("=")
        else:
            parsed_renderer = sys.argv[i + 1]
        config.renderer = parsed_renderer
    elif arg == "--use_opengl_renderer":
        config.renderer = "opengl"
    elif arg == "--use_webgl_renderer":
        config.renderer = "webgl"

# many scripts depend on this -> has to be loaded first
from .utils.commands import *  # isort:skip

from .animation.animation import *
from .animation.composition import *
from .animation.creation import *
from .animation.fading import *
from .animation.growing import *
from .animation.indication import *
from .animation.movement import *
from .animation.numbers import *
from .animation.rotation import *
from .animation.specialized import *
from .animation.transform import *
from .animation.transform_matching_parts import *
from .animation.update import *
from .camera.camera import *
from .camera.mapping_camera import *
from .camera.moving_camera import *
from .camera.multi_camera import *
from .camera.three_d_camera import *
from .constants import *
from .mobject.boolean_ops import *
from .mobject.changing import *
from .mobject.coordinate_systems import *
from .mobject.frame import *
from .mobject.functions import *
from .mobject.geometry import *
from .mobject.graph import *
from .mobject.logo import *
from .mobject.matrix import *
from .mobject.mobject import *
from .mobject.mobject_update_utils import *
from .mobject.number_line import *
from .mobject.numbers import *
from .mobject.polyhedra import *
from .mobject.probability import *
from .mobject.shape_matchers import *
from .mobject.svg.brace import *
from .mobject.svg.code_mobject import *
from .mobject.svg.style_utils import *
from .mobject.svg.svg_mobject import *
from .mobject.svg.svg_path import *
from .mobject.svg.tex_mobject import *
from .mobject.svg.text_mobject import *
from .mobject.table import *
from .mobject.three_d_utils import *
from .mobject.three_dimensions import *
from .mobject.types.dot_cloud import *
from .mobject.types.image_mobject import *
from .mobject.types.opengl_point_cloud_mobject import *
from .mobject.types.point_cloud_mobject import *
from .mobject.types.vectorized_mobject import *
from .mobject.value_tracker import *
from .mobject.vector_field import *
from .renderer.cairo_renderer import *
from .scene.moving_camera_scene import *
from .scene.reconfigurable_scene import *
from .scene.sample_space_scene import *
from .scene.scene import *
from .scene.scene_file_writer import *
from .scene.section import *
from .scene.three_d_scene import *
from .scene.vector_space_scene import *
from .scene.zoomed_scene import *
from .utils import color, rate_functions, unit
from .utils.bezier import *
from .utils.color import *
from .utils.config_ops import *
from .utils.debug import *
from .utils.file_ops import *
from .utils.images import *
from .utils.iterables import *
from .utils.paths import *
from .utils.rate_functions import *
from .utils.simple_functions import *
from .utils.sounds import *
from .utils.space_ops import *
from .utils.strings import *
from .utils.tex import *
from .utils.tex_templates import *

try:
    from IPython import get_ipython

    from .utils.ipython_magic import ManimMagic
except ImportError:
    pass
else:
    ipy = get_ipython()
    if ipy is not None:
        ipy.register_magics(ManimMagic)

from .plugins import *
