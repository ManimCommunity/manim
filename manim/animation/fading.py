# """Fading in and out of view.

# .. manim:: Example
#     :hide_source:

#     class Example(Scene):
#         def construct(self):
#             s1 = Square().set_color(BLUE)
#             s2 = Square().set_color(BLUE)
#             s3 = Square().set_color(BLUE)
#             s4 = Square().set_color(BLUE)
#             s5 = Square().set_color(BLUE)
#             s6 = Square().set_color(RED)
#             s7 = Square().set_color(RED)
#             s8 = Square().set_color(RED)
#             VGroup(s1, s2, s3, s4).set_x(0).arrange(buff=1.9).shift(UP)
#             VGroup(s5, s6, s7, s8).set_x(0).arrange(buff=1.9).shift(2 * DOWN)
#             t1 = Text("FadeIn").scale(0.5).next_to(s1, UP)
#             t2 = Text("FadeInFromPoint").scale(0.5).next_to(s2, UP)
#             t3 = Text("FadeInFrom").scale(0.5).next_to(s3, UP)
#             t4 = Text("VFadeIn").scale(0.5).next_to(s4, UP)
#             t5 = Text("FadeInFromLarge").scale(0.4).next_to(s5, UP)
#             t6 = Text("FadeOut").scale(0.45).next_to(s6, UP)
#             t7 = Text("FadeOutAndShift").scale(0.45).next_to(s7, UP)
#             t8 = Text("VFadeOut").scale(0.45).next_to(s8, UP)

#             objs = [ManimBanner().scale(0.25) for _ in range(1, 9)]
#             objs[0].move_to(s1.get_center())
#             objs[1].move_to(s2.get_center())
#             objs[2].move_to(s3.get_center())
#             objs[3].move_to(s4.get_center())
#             objs[4].move_to(s5.get_center())
#             objs[5].move_to(s6.get_center())
#             objs[6].move_to(s7.get_center())
#             objs[7].move_to(s8.get_center())
#             self.add(s1, s2, s3, s4, s5, s6, s7, s8, t1, t2, t3, t4, t5, t6, t7, t8)
#             self.add(*objs)

#             self.play(
#                 FadeIn(objs[0]),
#                 FadeInFromPoint(objs[1], s2.get_center()),
#                 FadeInFrom(objs[2], LEFT*0.2),
#                 VFadeIn(objs[3]),
#                 FadeInFromLarge(objs[4]),
#             )
#             self.wait(0.3)
#             self.play(
#                 FadeOut(objs[5]),
#                 FadeOutAndShift(objs[6], DOWN),
#                 VFadeOut(objs[7])
#             )

# """


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

from ..animation.transform import Transform
from ..constants import DOWN, ORIGIN
from ..mobject.mobject import Mobject
from ..scene.scene import Scene
from ..utils.deprecation import deprecated
from ..utils.rate_functions import there_and_back


class _Fade(Transform):
    def __init__(
        self,
        mobject: Mobject,
        shift: Optional[np.ndarray] = None,
        target_position: Optional[Union[np.ndarray, Mobject]] = None,
        scale: float = 1,
        **kwargs
    ) -> None:
        self.point_target = False
        if shift is None:
            if target_position is not None:
                if isinstance(target_position, Mobject):
                    target_position = target_position.get_center()
                shift = target_position - mobject.get_center()
                self.point_target = True
            else:
                shift = ORIGIN
        self.shift_vector = shift
        print(self.shift_vector)
        self.scale_factor = scale
        super().__init__(mobject, **kwargs)

    def _create_faded_mobject(self, fadeIn: bool):
        faded_mobject = self.mobject.copy()
        faded_mobject.set_opacity(0)
        direction_modifier = -1 if fadeIn and not self.point_target else 1
        faded_mobject.shift(self.shift_vector * direction_modifier)
        faded_mobject.scale(self.scale_factor)
        return faded_mobject


class FadeIn(_Fade):
    def create_target(self):
        return self.mobject

    def create_starting_mobject(self):
        return self._create_faded_mobject(fadeIn=True)


class FadeOut(_Fade):
    def __init__(self, mobject: Mobject, **kwargs) -> None:
        super().__init__(mobject, remover=True, **kwargs)

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
        super().__init__(mobject, shift=direction, **kwargs)


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
