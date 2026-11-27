"""Private raster-target primitives for :mod:`manim.renderer.cairo`."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import cairo
import numpy as np
from PIL import Image

from manim.renderer._view import view_transform_center
from manim.utils.color import color_to_int_rgba
from manim.utils.images import get_full_raster_image_path

if TYPE_CHECKING:
    from manim.renderer.protocol import CameraState, RenderConfig
    from manim.typing import RGBAPixelArray


#: What a target's background is made from: the image, or the colour with its opacity.
BackgroundKey: TypeAlias = tuple[str | None, tuple[float, float, float, float]]


def background_key(config: RenderConfig) -> BackgroundKey:
    """What the background of a frame with ``config`` is made from; frames with equal
    keys start from the same pixels.
    """
    return (
        config.background_image,
        tuple(config.background_color.to_rgba_with_alpha(config.background_opacity)),
    )


@dataclass(frozen=True, slots=True)
class _CairoRasterSettings:
    """The size of one Cairo raster target."""

    pixel_width: int
    pixel_height: int

    def __post_init__(self) -> None:
        if min(self.pixel_width, self.pixel_height) <= 0:
            raise ValueError("Cairo raster dimensions must be positive.")


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

    def clear(self) -> None:
        """Forget the images read so far."""
        self._images.clear()


#: The view: the context matrix's ``xx``, ``yy``, ``x0`` and ``y0``.
View = tuple[float, float, float, float]


class _PathCache:
    """Cairo paths by node key, replayed while the node keeps its key.

    Entries live for two generations: a path drawn in one frame is kept for the next,
    and dropped if it was not drawn again.

    Cairo stores paths in device space with 1/256 pixel resolution, so a path recorded
    under one view and replayed under another is rounded twice. A path is therefore
    only replayed under the same view.
    """

    def __init__(self) -> None:
        self._current: dict[Hashable, tuple[View, cairo.Path]] = {}
        self._previous: dict[Hashable, tuple[View, cairo.Path]] = {}

    def next_generation(self) -> None:
        self._previous, self._current = self._current, {}

    def get(self, key: Hashable, view: View) -> cairo.Path | None:
        entry = self._current.get(key)
        if entry is None:
            entry = self._previous.pop(key, None)
            if entry is None:
                return None
            self._current[key] = entry
        cached_view, path = entry
        return path if cached_view == view else None

    def put(self, key: Hashable, view: View, path: cairo.Path) -> None:
        self._current[key] = (view, path)

    def clear(self) -> None:
        self._current.clear()
        self._previous.clear()


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
        self._current_background: tuple[object, ...] | None = None
        self._background = np.zeros_like(self._pixels)
        self._context_key: tuple[float, ...] | None = None
        self._context: cairo.Context | None = None
        self._scratch_target: _CairoRenderTarget | None = None
        self.path_cache = _PathCache()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The Cairo render target is closed.")

    @property
    def pixels(self) -> RGBAPixelArray:
        self._ensure_open()
        return self._pixels

    @property
    def _size(self) -> tuple[int, int]:
        return self.settings.pixel_width, self.settings.pixel_height

    def _load_background(self, config: RenderConfig) -> RGBAPixelArray:
        if config.background_image is None:
            background = np.empty_like(self._pixels)
            background[:, :] = color_to_int_rgba(
                config.background_color,
                config.background_opacity,
            )
            return background
        return self.image_files.get(config.background_image, self._size)

    def reset(self, config: RenderConfig) -> None:
        """Fill the target with the background of ``config``."""
        self._ensure_open()
        key = background_key(config)
        if key != self._current_background:
            self._background = self._load_background(config)
            self._current_background = key
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

    def get_context(self, camera: CameraState) -> cairo.Context:
        self._ensure_open()
        settings = self.settings
        center = view_transform_center(camera)
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
        self.path_cache.clear()
        self._current_background = None
        self._pixels = np.empty((0, 0, 4), dtype=np.uint8)
        self._background = self._pixels
