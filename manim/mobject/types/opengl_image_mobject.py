import numpy as np
from PIL import Image

from ...constants import *
from ...utils.bezier import inverse_interpolate
from ...utils.images import get_full_raster_image_path
from ...utils.iterables import listify
from ..opengl_mobject import OpenGLMobject


class OpenGLImageMobject(OpenGLMobject):
    shader_dtype = [
        ("point", np.float32, (3,)),
        ("im_coords", np.float32, (2,)),
        ("opacity", np.float32, (1,)),
    ]
    shader_folder = "image"

    def __init__(self, filename, width=None, height=None, opacity=1, **kwargs):
        path = get_full_raster_image_path(filename)
        self.image = Image.open(path)
        self.size = self.image.size
        if width is None and height is None:
            width = self.size[0] / self.size[1]
            height = 1
        if height is None:
            height = width * self.size[1] / self.size[0]
        if width is None:
            width = height * self.size[0] / self.size[1]

        self.tmp_width = width
        self.tmp_height = height
        super().__init__(
            opacity=opacity,
            texture_paths={"Texture": path},
            width=width,
            height=height,
            **kwargs
        )

    def init_data(self):
        w = self.tmp_width / 2
        h = self.tmp_height / 2
        self.data = {
            "points": np.array(
                [
                    UP * h + LEFT * w,
                    DOWN * h + LEFT * w,
                    UP * h + RIGHT * w,
                    UP * h + RIGHT * w,
                    DOWN * h + RIGHT * w,
                    DOWN * h + LEFT * w,
                ]
            ),
            "im_coords": np.array([(0, 0), (0, 1), (1, 0), (1, 0), (1, 1), (0, 1)]),
            "opacity": np.array([[self.opacity]], dtype=np.float32),
        }

    def set_opacity(self, opacity, recurse=True):
        for mob in self.get_family(recurse):
            mob.data["opacity"] = np.array([[o] for o in listify(opacity)])
        return self

    def point_to_rgb(self, point):
        x0, y0 = self.get_corner(UL)[:2]
        x1, y1 = self.get_corner(DR)[:2]
        x_alpha = inverse_interpolate(x0, x1, point[0])
        y_alpha = inverse_interpolate(y0, y1, point[1])
        if not (0 <= x_alpha <= 1) and (0 <= y_alpha <= 1):
            # TODO, raise smarter exception
            raise Exception("Cannot sample color from outside an image")

        pw, ph = self.image.size
        rgb = self.image.getpixel(
            (
                int((pw - 1) * x_alpha),
                int((ph - 1) * y_alpha),
            )
        )
        return np.array(rgb) / 255

    def get_shader_data(self):
        shader_data = super().get_shader_data()
        self.read_data_to_shader(shader_data, "im_coords", "im_coords")
        self.read_data_to_shader(shader_data, "opacity", "opacity")
        return shader_data
