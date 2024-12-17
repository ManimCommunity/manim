#!/usr/bin/env python
from __future__ import annotations

from importlib.metadata import version

__version__ = version(__name__)


# isort: off

# Importing the config module should be the first thing we do, since other
# modules depend on the global config dict for initialization.
from manim._config import *

# many scripts depend on this -> has to be loaded first
from manim.utils.commands import *

# isort: on
import numpy as np

from manim.animation.animation import *
from manim.animation.changing import *
from manim.animation.composition import *
from manim.animation.creation import *
from manim.animation.fading import *
from manim.animation.growing import *
from manim.animation.indication import *
from manim.animation.movement import *
from manim.animation.numbers import *
from manim.animation.rotation import *
from manim.animation.specialized import *
from manim.animation.speedmodifier import *
from manim.animation.transform import *
from manim.animation.transform_matching_parts import *
from manim.animation.updaters.mobject_update_utils import *
from manim.animation.updaters.update import *
from manim.constants import *
from manim.file_writer import *
from manim.manager import *
from manim.mobject.frame import *
from manim.mobject.geometry.arc import *
from manim.mobject.geometry.boolean_ops import *
from manim.mobject.geometry.labeled import *
from manim.mobject.geometry.line import *
from manim.mobject.geometry.polygram import *
from manim.mobject.geometry.shape_matchers import *
from manim.mobject.geometry.tips import *
from manim.mobject.graph import *
from manim.mobject.graphing.coordinate_systems import *
from manim.mobject.graphing.functions import *
from manim.mobject.graphing.number_line import *
from manim.mobject.graphing.probability import *
from manim.mobject.graphing.scale import *
from manim.mobject.logo import *
from manim.mobject.matrix import *
from manim.mobject.mobject import *
from manim.mobject.opengl.dot_cloud import *
from manim.mobject.opengl.opengl_point_cloud_mobject import *
from manim.mobject.opengl.opengl_vectorized_mobject import *
from manim.mobject.svg.brace import *
from manim.mobject.svg.svg_mobject import *
from manim.mobject.table import *
from manim.mobject.text.code_mobject import *
from manim.mobject.text.numbers import *
from manim.mobject.text.tex_mobject import *
from manim.mobject.text.text_mobject import *
from manim.mobject.three_d.polyhedra import *
from manim.mobject.three_d.three_d_utils import *
from manim.mobject.three_d.three_dimensions import *
from manim.mobject.types.image_mobject import *
from manim.mobject.types.point_cloud_mobject import *
from manim.mobject.types.vectorized_mobject import *
from manim.mobject.value_tracker import *
from manim.mobject.vector_field import *
from manim.scene.scene import *
from manim.scene.sections import *
from manim.scene.vector_space_scene import *
from manim.utils import color, rate_functions, unit
from manim.utils.bezier import *
from manim.utils.color import *
from manim.utils.config_ops import *
from manim.utils.debug import *
from manim.utils.file_ops import *
from manim.utils.images import *
from manim.utils.iterables import *
from manim.utils.paths import *
from manim.utils.rate_functions import *
from manim.utils.simple_functions import *
from manim.utils.sounds import *
from manim.utils.space_ops import *
from manim.utils.tex import *
from manim.utils.tex_templates import *

try:
    from IPython import get_ipython

    from manim.utils.ipython_magic import ManimMagic
except ImportError:
    pass
else:
    ipy = get_ipython()
    if ipy is not None:
        ipy.register_magics(ManimMagic)

from manim.plugins import *
