"""Compatibility name for the now-movable default Cairo camera."""

from __future__ import annotations

__all__ = ["MovingCamera"]

from .camera import Camera


class MovingCamera(Camera):
    """Compatibility subclass of :class:`~manim.camera.camera.Camera`.

    The default Cairo camera now owns the same animatable frame and
    :meth:`~manim.camera.camera.Camera.auto_zoom` behavior.
    """
