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


__all__ = [
    "FadeOut",
    "FadeIn",
    "FadeInFrom",
    "FadeOutAndShift",
    "FadeOutToPoint",
    "FadeInFromPoint",
    "FadeInFromLarge",
    "VFadeIn",
    "VFadeOut",
    "VFadeInThenOut",
]

from typing import Callable, Optional, Union

import numpy as np

from manim.mobject.opengl_mobject import OpenGLMobject

from ..animation.transform import Transform
from ..constants import DOWN, ORIGIN
from ..mobject.mobject import Group, Mobject
from ..scene.scene import Scene
from ..utils.deprecation import deprecated
from ..utils.rate_functions import there_and_back


class _Fade(Transform):
    """Fade :class:`~.Mobject` s in or out.

    Parameters
    ----------
    mobjects
        The mobjects to be faded.
    shift
        The vector by which the mobject shifts while being faded.
    target_position
        The position to/from which the mobject moves while being faded in. In case
        another mobject is given as target position, its center is used.
    scale
        The factor by which the mobject is scaled initially before being rescaling to
        its original size while being faded in.

    """

    def __init__(
        self,
        *mobjects: Mobject,
        shift: Optional[np.ndarray] = None,
        target_position: Optional[Union[np.ndarray, Mobject]] = None,
        scale: float = 1,
        **kwargs
    ) -> None:
        if not mobjects:
            raise ValueError("At least one mobject must be passed.")
        if len(mobjects) == 1:
            mobject = mobjects[0]
        else:
            mobject = Group(*mobjects)

        self.point_target = False
        if shift is None:
            if target_position is not None:
                if isinstance(target_position, (Mobject, OpenGLMobject)):
                    target_position = target_position.get_center()
                shift = target_position - mobject.get_center()
                self.point_target = True
            else:
                shift = ORIGIN
        self.shift_vector = shift
        self.scale_factor = scale
        super().__init__(mobject, **kwargs)

    def _create_faded_mobject(self, fadeIn: bool) -> Mobject:
        """Create a faded, shifted and scaled copy of the mobject.

        Parameters
        ----------
        fadeIn
            Whether the faded mobject is used to fade in.

        Returns
        -------
        Mobject
            The faded, shifted and scaled copy of the mobject.
        """
        faded_mobject = self.mobject.copy()
        faded_mobject.fade(1)
        direction_modifier = -1 if fadeIn and not self.point_target else 1
        faded_mobject.shift(self.shift_vector * direction_modifier)
        faded_mobject.scale(self.scale_factor)
        return faded_mobject


class FadeIn(_Fade):
    """Fade in :class:`~.Mobject` s.

    Parameters
    ----------
    mobjects
        The mobjects to be faded in.
    shift
        The vector by which the mobject shifts while being faded in.
    target_position
        The position from which the mobject starts while being faded in. In case
        another mobject is given as target position, its center is used.
    scale
        The factor by which the mobject is scaled initially before being rescaling to
        its original size while being faded in.

    Examples
    --------

    .. manim :: FadeInExample

        class FadeInExample(Scene):
            def construct(self):
                dot = Dot(UP * 2 + LEFT)
                self.add(dot)
                tex = Tex(
                    "FadeIn with ", "shift ", " or target\\_position", " and scale"
                ).scale(1)
                animations = [
                    FadeIn(tex[0]),
                    FadeIn(tex[1], shift=DOWN),
                    FadeIn(tex[2], target_position=dot),
                    FadeIn(tex[3], scale=1.5),
                ]
                self.play(AnimationGroup(*animations, lag_ratio=0.5))

    """

    def create_target(self):
        return self.mobject

    def create_starting_mobject(self):
        return self._create_faded_mobject(fadeIn=True)


class FadeOut(_Fade):
    """Fade out :class:`~.Mobject` s.

    Parameters
    ----------
    mobjects
        The mobjects to be faded out.
    shift
        The vector by which the mobject shifts while being faded out.
    target_position
        The position to which the mobject moves while being faded out. In case another
        mobject is given as target position, its center is used.
    scale
        The factor by which the mobject is scaled while being faded out.

    Examples
    --------

    .. manim :: FadeInExample

        class FadeInExample(Scene):
            def construct(self):
                dot = Dot(UP * 2 + LEFT)
                self.add(dot)
                tex = Tex(
                    "FadeOut with ", "shift ", " or target\\_position", " and scale"
                ).scale(1)
                animations = [
                    FadeOut(tex[0]),
                    FadeOut(tex[1], shift=DOWN),
                    FadeOut(tex[2], target_position=dot),
                    FadeOut(tex[3], scale=0.5),
                ]
                self.play(AnimationGroup(*animations, lag_ratio=0.5))


    """

    def __init__(self, *mobjects: Mobject, **kwargs) -> None:
        super().__init__(*mobjects, remover=True, **kwargs)

    def create_target(self):
        return self._create_faded_mobject(fadeIn=False)

    def clean_up_from_scene(self, scene: Scene = None) -> None:
        super().clean_up_from_scene(scene)
        self.interpolate(0)


@deprecated(
    since="v0.6.0",
    until="v0.8.0",
    replacement="FadeIn",
    message="You can set a shift amount there.",
)
class FadeInFrom(FadeIn):
    def __init__(
        self, mobject: "Mobject", direction: np.ndarray = DOWN, **kwargs
    ) -> None:
        super().__init__(mobject, shift=-direction, **kwargs)


@deprecated(
    since="v0.6.0",
    until="v0.8.0",
    replacement="FadeOut",
    message="You can set a shift amount there.",
)
class FadeOutAndShift(FadeIn):
    def __init__(
        self, mobject: "Mobject", direction: np.ndarray = DOWN, **kwargs
    ) -> None:
        super().__init__(mobject, shift=direction, **kwargs)


@deprecated(
    since="v0.6.0",
    until="v0.8.0",
    replacement="FadeOut",
    message="You can set a target position there.",
)
class FadeOutToPoint(FadeOut):
    def __init__(
        self, mobject: "Mobject", point: Union["Mobject", np.ndarray] = ORIGIN, **kwargs
    ) -> None:
        super().__init__(mobject, target_position=point, **kwargs)


@deprecated(
    since="v0.6.0",
    until="v0.8.0",
    replacement="FadeIn",
    message="You can set a target position and scaling factor there.",
)
class FadeInFromPoint(FadeIn):
    def __init__(
        self, mobject: "Mobject", point: Union["Mobject", np.ndarray], **kwargs
    ) -> None:
        super().__init__(mobject, target_position=point, scale=0, **kwargs)


@deprecated(
    since="v0.6.0",
    until="v0.8.0",
    replacement="FadeIn",
    message="You can set a scaling factor there.",
)
class FadeInFromLarge(FadeIn):
    def __init__(self, mobject: "Mobject", scale_factor: float = 2, **kwargs) -> None:
        super().__init__(mobject, scale=scale_factor, **kwargs)


@deprecated(since="v0.6.0", until="v0.8.0", replacement="FadeIn")
class VFadeIn(FadeIn):
    def __init__(self, mobject: "Mobject", **kwargs) -> None:
        super().__init__(mobject, **kwargs)


@deprecated(since="v0.6.0", until="v0.8.0", replacement="FadeOut")
class VFadeOut(FadeOut):
    def __init__(self, mobject: "Mobject", **kwargs) -> None:
        super().__init__(mobject, **kwargs)


@deprecated(since="v0.6.0", until="v0.8.0")
class VFadeInThenOut(FadeIn):
    def __init__(
        self,
        mobject: "Mobject",
        rate_func: Callable[[float], float] = there_and_back,
        **kwargs
    ):
        super().__init__(mobject, remover=True, rate_func=rate_func, **kwargs)
