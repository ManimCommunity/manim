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

# The OpenGL renderer samples textures with nearest-neighbor filtering. To apply
# ``resampling_algorithm`` to image arrays anyway, they are upscaled with it
# before they are uploaded as textures.

# Maximum factor by which an image array is upscaled.
_MAX_UPSCALE_FACTOR = 200
# Maximum length (in pixels) of the longer side of an upscaled image array.
_MAX_UPSCALED_SIDE_LENGTH = 4096


def _get_upscale_factor(width: int, height: int) -> int:
    """Return the factor by which an image array of the given size is upscaled.

    The factor is at most ``_MAX_UPSCALE_FACTOR``, and the longer side of the
    upscaled image is at most ``_MAX_UPSCALED_SIDE_LENGTH`` pixels. Image arrays
    are never downscaled.
    """
    return max(
        1, min(_MAX_UPSCALE_FACTOR, _MAX_UPSCALED_SIDE_LENGTH // max(width, height))
    )


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
            image = Image.fromarray(image_file.astype("uint8")).convert(image_mode)
            width, height = image.size
            scale = _get_upscale_factor(width, height)
            return image.resize(
                (width * scale, height * scale),
                resample=self.resampling_algorithm,
            )
