"""Semantic camera state for nested Cairo camera views."""

from __future__ import annotations

__all__ = ["MultiCamera"]

from collections.abc import Iterable
from typing import Any

from manim.mobject.mobject import Mobject
from manim.mobject.types.image_mobject import ImageMobjectFromCamera

from .camera import Camera


class MultiCamera(Camera):
    """Describe a primary view with camera-backed image mobjects."""

    def __init__(
        self,
        image_mobjects_from_cameras: Iterable[ImageMobjectFromCamera] | None = None,
        **kwargs: Any,
    ) -> None:
        self.image_mobjects_from_cameras: list[ImageMobjectFromCamera] = []
        if image_mobjects_from_cameras is not None:
            for image_mobject in image_mobjects_from_cameras:
                self.add_image_mobject_from_camera(image_mobject)
        super().__init__(**kwargs)

    def add_image_mobject_from_camera(
        self,
        image_mobject_from_camera: ImageMobjectFromCamera,
    ) -> None:
        """Register a camera-backed image for renderer-owned composition."""
        if not isinstance(image_mobject_from_camera.camera, Camera):
            raise TypeError("Nested Cairo views require a Cairo Camera.")
        self.image_mobjects_from_cameras.append(image_mobject_from_camera)

    def get_mobjects_indicating_movement(self) -> list[Mobject]:
        """Return controls whose movement changes a primary or nested view."""

        def collect(camera: Camera, visited: set[int]) -> list[Mobject]:
            if id(camera) in visited:
                return []
            visited.add(id(camera))
            if not isinstance(camera, MultiCamera):
                return camera.get_mobjects_indicating_movement()

            indicators = Camera.get_mobjects_indicating_movement(camera)
            for image_mobject in camera.image_mobjects_from_cameras:
                indicators.extend(collect(image_mobject.camera, visited))
            return indicators

        return collect(self, set())
