from __future__ import annotations

__all__ = [
    "OpenGLImageMobject",
]

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from PIL.Image import Resampling

from manim.mobject.opengl.opengl_surface import OpenGLSurface, OpenGLTexturedSurface
from manim.utils.images import get_full_raster_image_path

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["OpenGLImageMobject"]


class OpenGLImageMobject(OpenGLTexturedSurface):
    def __init__(
        self,
        filename_or_array: str | Path | npt.NDArray,
        width: float | None = None,
        height: float | None = None,
        image_mode: str = "RGBA",
        resampling_algorithm: Resampling = Resampling.BICUBIC,
        opacity: float = 1,
        gloss: float = 0,
        shadow: float = 0,
        **kwargs: Any,
    ):
        self.image = filename_or_array
        self.resampling_algorithm = resampling_algorithm
        if isinstance(filename_or_array, np.ndarray):
            self.size = filename_or_array.shape[1::-1]
        elif isinstance(filename_or_array, (str, Path)):
            path = get_full_raster_image_path(filename_or_array)
            self.size = Image.open(path).size

        if width is None and height is None:
            width = 4 * self.size[0] / self.size[1]
            height = 4
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
            self.image,
            image_mode=image_mode,
            opacity=opacity,
            gloss=gloss,
            shadow=shadow,
            **kwargs,
        )

    def get_image_from_file(
        self,
        image_file: str | Path | np.ndarray,
        image_mode: str,
    ) -> Image.Image:
        if isinstance(image_file, (str, Path)):
            return super().get_image_from_file(image_file, image_mode)
        else:
            return (
                Image.fromarray(image_file.astype("uint8"))
                .convert(image_mode)
                .resize(
                    image_file.shape[:2]
                    * 200,  # assumption of 200 ppmu (pixels per manim unit) would suffice
                    resample=self.resampling_algorithm,
                )
            )
