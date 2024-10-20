from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from manim.animation.animation import Animation
    from manim.mobject.mobject import Mobject
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject


T_co = TypeVar("T_co", covariant=True, bound="Mobject | OpenGLMobject")

__all__ = [
    "_AnimationBuilder",
    "_UpdaterBuilder",
]


class _AnimationBuilder(Generic[T_co]):
    def __init__(self, mobject: T_co):
        self.mobject = mobject
        self.mobject.generate_target()

        self.overridden_animation = None
        self.is_chaining = False
        self.methods = []

        # Whether animation args can be passed
        self.cannot_pass_args = False
        self.anim_args = {}

    def __call__(self, **kwargs) -> Self:
        if self.cannot_pass_args:
            raise ValueError(
                "Animation arguments must be passed before accessing methods and can only be passed once",
            )

        self.anim_args = kwargs
        self.cannot_pass_args = True

        return self

    def __getattr__(self, method_name: str):
        method = getattr(self.mobject.target, method_name)
        has_overridden_animation = hasattr(method, "_override_animate")

        if (self.is_chaining and has_overridden_animation) or self.overridden_animation:
            raise NotImplementedError(
                "Method chaining is currently not supported for "
                "overridden animations",
            )

        def update_target(*method_args, **method_kwargs):
            if has_overridden_animation:
                self.overridden_animation = method._override_animate(
                    self.mobject,
                    *method_args,
                    anim_args=self.anim_args,
                    **method_kwargs,
                )
            else:
                self.methods.append([method, method_args, method_kwargs])
                method(*method_args, **method_kwargs)
            return self

        self.is_chaining = True
        self.cannot_pass_args = True

        return update_target

    def build(self) -> Animation:
        from manim.animation.transform import _MethodAnimation

        if self.overridden_animation:
            anim = self.overridden_animation
        else:
            anim = _MethodAnimation(self.mobject, self.methods)

        for attr, value in self.anim_args.items():
            setattr(anim, attr, value)

        return anim


class _UpdaterBuilder(Generic[T_co]):
    """Syntactic sugar for adding updaters to mobjects."""

    def __init__(self, mobject: T_co):
        self._mobject = mobject

    def __getattr__(self, name: str, /):
        # just return a function that will add the updater
        def add_updater(*method_args, **method_kwargs):
            self._mobject.add_updater(
                lambda m: getattr(m, name)(*method_args, **method_kwargs),
                call_updater=True,
            )
            return self

        return add_updater
