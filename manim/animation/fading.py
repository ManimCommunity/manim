"""Fading in and out of view.

.. manim:: Fading

    class Fading(Scene):
        def construct(self):
            tex_in = Tex("Fade", "In").scale(3)
            tex_out = Tex("Fade", "Out").scale(3)
            self.play(FadeIn(tex_in, shift=DOWN, scale=0.66))
            self.play(ReplacementTransform(tex_in, tex_out))
            self.play(FadeOut(tex_out, shift=DOWN * 2, scale=1.5))

"""

from __future__ import annotations

__all__ = [
    "FadeOut",
    "FadeIn",
]

import numpy as np
from typing import Any, Union

from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from ..animation.transform import Transform
from ..constants import ORIGIN
from ..mobject.mobject import Group, Mobject
from ..scene.scene import Scene


class _Fade(Transform):
    """Base class for fading Mobjects in or out."""

    def __init__(
        self,
        *mobjects: Mobject,
        shift: np.ndarray | None = None,
        target_position: np.ndarray | Mobject | None = None,
        scale: float = 1,
        **kwargs: Any,
    ) -> None:
        
        if not mobjects:
            raise ValueError("At least one mobject must be provided for fading.")
        
        for mob in mobjects:
            if not isinstance(mob, Mobject):
                raise TypeError(f"Expected Mobject instances, got {type(mob)}")

        mobject = mobjects[0] if len(mobjects) == 1 else Group(*mobjects)
        
        self.point_target = False
        
        if shift is None:
            if target_position is not None:
                if isinstance(target_position, (Mobject, OpenGLMobject)):
                    target_position = target_position.get_center()
                elif not isinstance(target_position, np.ndarray):
                    raise TypeError(
                        "target_position must be a Mobject or np.ndarray"
                    )
                shift = np.array(target_position) - mobject.get_center()
                self.point_target = True
            else:
                shift = ORIGIN
        else:
            if not isinstance(shift, np.ndarray):
                raise TypeError("shift must be of type np.ndarray")

        if not isinstance(scale, (int, float)):
            raise TypeError("scale must be a number")

        self.shift_vector = shift
        self.scale_factor = scale

        super().__init__(mobject, **kwargs)

    def _create_faded_mobject(self, fadeIn: bool) -> Mobject:
        """Create a faded, shifted and scaled copy of the mobject."""
        
        faded_mobject = self.mobject.copy()  # type: ignore[assignment]
        
        if not isinstance(faded_mobject, Mobject):
            raise RuntimeError("Failed to create faded mobject copy.")

        faded_mobject.fade(1)
        
        direction_modifier = -1 if fadeIn and not self.point_target else 1
        
        faded_mobject.shift(self.shift_vector * direction_modifier)
        faded_mobject.scale(self.scale_factor)
        
        return faded_mobject


class FadeIn(_Fade):
    """Fade in Mobjects with optional shift, target_position, or scale."""

    def __init__(self, *mobjects: Mobject, **kwargs: Any) -> None:
        super().__init__(*mobjects, introducer=True, **kwargs)

    def create_target(self) -> Mobject:
        return self.mobject  # type: ignore[return-value]

    def create_starting_mobject(self) -> Mobject:
        return self._create_faded_mobject(fadeIn=True)


class FadeOut(_Fade):
    """Fade out Mobjects with optional shift, target_position, or scale."""

    def __init__(self, *mobjects: Mobject, **kwargs: Any) -> None:
        super().__init__(*mobjects, remover=True, **kwargs)

    def create_target(self) -> Mobject:
        return self._create_faded_mobject(fadeIn=False)

    def clean_up_from_scene(self, scene: Scene) -> None:
        """Remove the mobject from the scene after fading out."""
        super().clean_up_from_scene(scene)
        try:
            self.interpolate(0)
        except Exception as e:
            raise RuntimeError(f"Error during cleanup interpolation: {e}")
