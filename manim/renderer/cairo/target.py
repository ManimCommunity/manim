"""Private raster-target primitives for :mod:`manim.renderer.cairo`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cairo
import numpy as np
from PIL import Image

from manim.utils.color import color_to_int_rgba
from manim.utils.images import get_full_raster_image_path

if TYPE_CHECKING:
    from manim.typing import RGBAPixelArray

    from .camera import Camera


@dataclass(frozen=True, slots=True)
class _CairoRasterSettings:
    """Immutable dimensions and scaling used by one Cairo raster target."""

    pixel_width: int
    pixel_height: int
    base_pixel_width: int
    base_pixel_height: int
    cairo_line_width_multiple: float = 0.01

    def __post_init__(self) -> None:
        if (
            min(
                self.pixel_width,
                self.pixel_height,
                self.base_pixel_width,
                self.base_pixel_height,
            )
            <= 0
        ):
            raise ValueError("Cairo raster dimensions must be positive.")

    def resized(self, *, pixel_width: int, pixel_height: int) -> _CairoRasterSettings:
        return _CairoRasterSettings(
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            base_pixel_width=self.base_pixel_width,
            base_pixel_height=self.base_pixel_height,
            cairo_line_width_multiple=self.cairo_line_width_multiple,
        )


def _rgba_image(
    image: Image.Image | Path | str, size: tuple[int, int]
) -> RGBAPixelArray:
    """The pixels of ``image`` (a file or a PIL image) as RGBA, resized to ``size``."""
    if isinstance(image, (str, Path)):
        with Image.open(get_full_raster_image_path(image)) as source:
            rgba = source.convert("RGBA")
    else:
        rgba = image.convert("RGBA")
    if rgba.size != size:
        rgba = rgba.resize(size)
    return np.asarray(rgba, dtype=np.uint8).copy()


class _ImageFiles:
    """Image files read for one renderer, by path and size, shared by its targets."""

    def __init__(self) -> None:
        self._images: dict[tuple[str, tuple[int, int]], RGBAPixelArray] = {}

    def get(
        self, image: Image.Image | Path | str, size: tuple[int, int]
    ) -> RGBAPixelArray:
        # Only files are cached: an image object can change in place, and its
        # string form names a memory address that another image can reuse.
        if not isinstance(image, (str, Path)):
            return _rgba_image(image, size)
        key = (str(image), size)
        cached = self._images.get(key)
        if cached is None:
            cached = self._images[key] = _rgba_image(image, size)
            cached.flags.writeable = False
        return cached


class _CairoRenderTarget:
    """Own one top-left-origin RGBA target and its PyCairo context."""

    def __init__(
        self, settings: _CairoRasterSettings, image_files: _ImageFiles | None = None
    ) -> None:
        self.settings = settings
        #: Image files read for this target, shared with the renderer's other targets.
        self.image_files = _ImageFiles() if image_files is None else image_files
        self._pixels = np.zeros(
            (settings.pixel_height, settings.pixel_width, 4),
            dtype=np.uint8,
        )
        self._background_key: tuple[object, ...] | None = None
        self._background = np.zeros_like(self._pixels)
        self._context_key: tuple[float, ...] | None = None
        self._context: cairo.Context | None = None
        self._scratch_target: _CairoRenderTarget | None = None
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The Cairo render target is closed.")

    @property
    def pixels(self) -> RGBAPixelArray:
        self._ensure_open()
        return self._pixels

    def _camera_background_key(self, camera: Camera) -> tuple[object, ...]:
        return (
            camera.background_image,
            tuple(
                camera.background_color.to_rgba_with_alpha(
                    camera.background_opacity,
                ),
            ),
        )

    @property
    def _size(self) -> tuple[int, int]:
        return self.settings.pixel_width, self.settings.pixel_height

    def _load_background(self, camera: Camera) -> RGBAPixelArray:
        if camera.background_image is None:
            background = np.empty_like(self._pixels)
            background[:, :] = color_to_int_rgba(
                camera.background_color,
                camera.background_opacity,
            )
            return background
        return self.image_files.get(camera.background_image, self._size)

    def reset(self, camera: Camera) -> None:
        self._ensure_open()
        key = self._camera_background_key(camera)
        if key != self._background_key:
            self._background = self._load_background(camera)
            self._background_key = key
        np.copyto(self._pixels, self._background)

    def clear(self) -> None:
        """Clear this target to transparent black without allocating an array."""
        self.pixels.fill(0)

    def get_scratch_target(self) -> _CairoRenderTarget:
        """Return a reusable same-sized target for intermediate composition."""
        self._ensure_open()
        if self._scratch_target is None:
            self._scratch_target = _CairoRenderTarget(self.settings, self.image_files)
        return self._scratch_target

    def set_pixels(self, pixels: RGBAPixelArray) -> None:
        self._ensure_open()
        if pixels.shape != self._pixels.shape:
            raise ValueError(
                f"Cairo target pixels must have shape {self._pixels.shape}; "
                f"got {pixels.shape}.",
            )
        if pixels.dtype != np.uint8:
            raise TypeError("Cairo target pixels must use uint8.")
        np.copyto(self._pixels, pixels)

    def get_context(self, camera: Camera) -> cairo.Context:
        self._ensure_open()
        settings = self.settings
        center = camera.get_view_transform_center()
        view_key = (
            float(center[0]),
            float(center[1]),
            float(camera.frame_width),
            float(camera.frame_height),
        )
        if self._context is not None and self._context_key == view_key:
            return self._context

        surface = cairo.ImageSurface.create_for_data(
            self._pixels.data,
            cairo.FORMAT_ARGB32,
            settings.pixel_width,
            settings.pixel_height,
        )
        context = cairo.Context(surface)
        context.set_matrix(
            cairo.Matrix(
                settings.pixel_width / camera.frame_width,
                0,
                0,
                -(settings.pixel_height / camera.frame_height),
                (settings.pixel_width / 2)
                - center[0] * (settings.pixel_width / camera.frame_width),
                (settings.pixel_height / 2)
                + center[1] * (settings.pixel_height / camera.frame_height),
            ),
        )
        self._context = context
        self._context_key = view_key
        return context

    def get_background_image(self, image: Image.Image | Path | str) -> RGBAPixelArray:
        """The pixels of a vectorized mobject's background image at this target's
        size (read-only for files, which are cached).
        """
        self._ensure_open()
        return self.image_files.get(image, self._size)

    def read_pixels(self) -> RGBAPixelArray:
        return self.pixels.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context = None
        self._context_key = None
        if self._scratch_target is not None:
            self._scratch_target.close()
            self._scratch_target = None
        self._background_key = None
        self._pixels = np.empty((0, 0, 4), dtype=np.uint8)
        self._background = self._pixels
