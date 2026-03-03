from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject as Mobject
    from manim.utils.rate_functions import RateFunction

    from .scene_buffer import SceneBuffer

M = TypeVar("M", bound="Mobject", default="Mobject")


__all__ = ("AnimationProtocol",)


class AnimationProtocol(Protocol):
    """A protocol that all animations must implement."""

    buffer: SceneBuffer
    """The interface to the scene. This can be used to add, remove, or replace mobjects on the scene."""

    apply_buffer: bool
    """Normally, the buffer is only applied at the beginning and end of an animation.

    To apply it mid animation, set :attr:`apply_buffer` to ``True``."""

    def begin(self) -> object:
        """Called before the animation starts.

        This is where all setup for the animation should be done, such
        as creating copies/targets of the mobject to animate, etc.
        """

    def finish(self) -> object:
        """Called after the animation finishes.

        This is where all cleanup should happen, such as removing
        mobjects from the scene, etc.
        """

    def interpolate(self, alpha: float) -> object:
        """This is called every frame of the animation.

        This method should update the animation to the given ``alpha`` value.

        Parameters
        ----------
            alpha : a value in the interval :math:`[0, 1]` representing the proportion of the animation that has passed.
        """

    def get_run_time(self) -> float:
        """Compute and return the run time of the animation."""
        raise NotImplementedError

    def update_rate_info(
        self,
        run_time: float | None,
        rate_func: RateFunction | None,
        lag_ratio: float | None,
    ) -> object:
        """Update the rate information for the animation.

        If any value is ``None``, it should not update
        the animation's corresponding attribute.
        """

    def update_mobjects(self, dt: float) -> object:
        """Update the mobjects during the animation.

        This method is called every frame of the animation
        """


class MobjectAnimation(AnimationProtocol, Protocol[M]):
    mobject: M
    """The mobject that is being animated."""

    suspend_mobject_updating: bool
    """Whether to suspend updating the mobject during the animation."""
