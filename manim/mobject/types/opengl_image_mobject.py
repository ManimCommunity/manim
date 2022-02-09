from typing import Union

import numpy as np
from PIL import Image

from ...utils.images import get_full_raster_image_path
from ..types.opengl_surface import OpenGLSurface, OpenGLTexturedSurface


class OpenGLImageMobject(OpenGLTexturedSurface):
    def __init__(
        self,
        filename_or_array: Union[str, np.ndarray],
        width: float = None,
        height: float = None,
        opacity: float = 1,
        gloss: float = 0,
        shadow: float = 0,
        **kwargs,
    ):
        if type(filename_or_array) == str:
            path = get_full_raster_image_path(filename_or_array)
            self.image = Image.open(path)
            self.size = self.image.size
        else:
            self.image = filename_or_array
            self.size = self.image.shape[1::-1]

        if width is None and height is None:
            width = self.size[0] / self.size[1]
            height = 1
        if height is None:
            height = width * self.size[1] / self.size[0]
        if width is None:
            width = height * self.size[0] / self.size[1]

        surface = OpenGLSurface(
            lambda u, v: np.array([u, v, 0]),
            [-width / 2, width / 2],
            [-height / 2, height / 2],
            opacity=opacity,
            gloss=gloss,
            shadow=shadow,
        )

        super().__init__(
            surface,
            filename_or_array,
            opacity=opacity,
            gloss=gloss,
            shadow=shadow,
            **kwargs,
        )
