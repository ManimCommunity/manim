"""Animate mobjects."""


__all__ = ["Animation", "Wait", "override_animation"]


from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

if TYPE_CHECKING:
    from manim.scene.scene import Scene

from .. import logger
from ..mobject import mobject, opengl_mobject
from ..mobject.mobject import Mobject
from ..mobject.opengl_mobject import OpenGLMobject
from ..utils.rate_functions import smooth

DEFAULT_ANIMATION_RUN_TIME: float = 1.0
DEFAULT_ANIMATION_LAG_RATIO: float = 0.0


class Animation:
    overrides: "Dict[Type[Animation], List[Dict[Type[Mobject], str]]]" = {}

    def __new__(
        cls, mobject: Optional[Mobject], *args, use_default: bool = False, **kwargs
    ):
        if (
            not use_default
            and cls in cls.overrides
            and type(mobject) in cls.overrides[cls]
        ):
            func = cls.overrides[cls][type(mobject)]
            anim = func(mobject, *args, **kwargs)
            return anim
        return super().__new__(cls)

    def __init__(
        self,
        mobject: Optional[Mobject],
        # If lag_ratio is 0, the animation is applied to all submobjects
        # at the same time
        # If 1, it is applied to each successively.
        # If 0 < lag_ratio < 1, its applied to each
        # with lagged start times
        lag_ratio: float = DEFAULT_ANIMATION_LAG_RATIO,
        run_time: float = DEFAULT_ANIMATION_RUN_TIME,
        rate_func: Callable[[float], float] = smooth,
        name: str = None,
        remover: bool = False,  # remove a mobject from the screen?
        suspend_mobject_updating: bool = True,
        **kwargs,
    ) -> None:
        self._typecheck_input(mobject)
        self.run_time: float = run_time
        self.rate_func: Callable[[float], float] = rate_func
        self.name: Optional[str] = name
        self.remover: bool = remover
        self.suspend_mobject_updating: bool = suspend_mobject_updating
        self.lag_ratio: float = lag_ratio
        self.starting_mobject: Mobject = Mobject()
        self.mobject: Mobject = mobject if mobject is not None else Mobject()
        if kwargs:
            logger.debug("Animation received extra kwargs: %s", kwargs)

        if hasattr(self, "CONFIG"):
            logger.error(
                (
                    "CONFIG has been removed from ManimCommunity.",
                    "Please use keyword arguments instead.",
                )
            )

    def _typecheck_input(self, mobject: Union[Mobject, None]) -> None:
        if mobject is None:
            logger.debug("Animation with empty mobject")
        elif not isinstance(mobject, Mobject) and not isinstance(
            mobject, OpenGLMobject
        ):
            raise TypeError("Animation only works on Mobjects")

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f"{self.__class__.__name__}({str(self.mobject)})"

    def __repr__(self) -> str:
        return str(self)

    def begin(self) -> None:
        # This is called right as an animation is being
        # played.  As much initialization as possible,
        # especially any mobject copying, should live in
        # this method
        self.starting_mobject = self.create_starting_mobject()
        if self.suspend_mobject_updating:
            # All calls to self.mobject's internal updaters
            # during the animation, either from this Animation
            # or from the surrounding scene, should do nothing.
            # It is, however, okay and desirable to call
            # the internal updaters of self.starting_mobject,
            # or any others among self.get_all_mobjects()
            self.mobject.suspend_updating()
        self.interpolate(0)

    def finish(self) -> None:
        self.interpolate(1)
        if self.suspend_mobject_updating and self.mobject is not None:
            self.mobject.resume_updating()

    def clean_up_from_scene(self, scene: "Scene") -> None:
        if self.is_remover():
            scene.remove(self.mobject)

    def create_starting_mobject(self) -> Mobject:
        # Keep track of where the mobject starts
        return self.mobject.copy()

    def get_all_mobjects(self) -> Tuple[Mobject, Mobject]:
        """
        Ordering must match the ordering of arguments to interpolate_submobject
        """
        return self.mobject, self.starting_mobject

    def get_all_families_zipped(self) -> Iterable[Tuple]:
        return zip(
            *[mob.family_members_with_points() for mob in self.get_all_mobjects()]
        )

    def update_mobjects(self, dt: float) -> None:
        """
        Updates things like starting_mobject, and (for
        Transforms) target_mobject.  Note, since typically
        (always?) self.mobject will have its updating
        suspended during the animation, this will do
        nothing to self.mobject.
        """
        for mob in self.get_all_mobjects_to_update():
            mob.update(dt)

    def get_all_mobjects_to_update(self) -> list:
        # The surrounding scene typically handles
        # updating of self.mobject.  Besides, in
        # most cases its updating is suspended anyway
        return list(filter(lambda m: m is not self.mobject, self.get_all_mobjects()))

    def copy(self) -> "Animation":
        return deepcopy(self)

    # Methods for interpolation, the mean of an Animation
    def interpolate(self, alpha: float) -> None:
        alpha = min(max(alpha, 0), 1)
        self.interpolate_mobject(self.rate_func(alpha))

    def update(self, alpha: float) -> None:
        """
        This method shouldn't exist, but it's here to
        keep many old scenes from breaking
        """
        logger.warning(
            "animation.update() has been deprecated. "
            "Please use animation.interpolate() instead."
        )
        self.interpolate(alpha)

    def interpolate_mobject(self, alpha: float) -> None:
        families = list(self.get_all_families_zipped())
        for i, mobs in enumerate(families):
            sub_alpha = self.get_sub_alpha(alpha, i, len(families))
            self.interpolate_submobject(*mobs, sub_alpha)

    def interpolate_submobject(
        self,
        submobject: Mobject,
        starting_submobject: Mobject,
        # target_copy: Mobject, #Todo: fix - signature of interpolate_submobject differes in Transform().
        alpha: float,
    ) -> "Animation":
        # Typically implemented by subclass
        pass

    def get_sub_alpha(self, alpha: float, index: int, num_submobjects: int) -> float:
        # TODO, make this more understandable, and/or combine
        # its functionality with AnimationGroup's method
        # build_animations_with_timings
        lag_ratio = self.lag_ratio
        full_length = (num_submobjects - 1) * lag_ratio + 1
        value = alpha * full_length
        lower = index * lag_ratio
        return min(max((value - lower), 0), 1)

    # Getters and setters
    def set_run_time(self, run_time: float) -> "Animation":
        self.run_time = run_time
        return self

    def get_run_time(self) -> float:
        return self.run_time

    def set_rate_func(
        self,
        rate_func: Callable[[float], float],
    ) -> "Animation":
        self.rate_func = rate_func
        return self

    def get_rate_func(
        self,
    ) -> Callable[[float], float]:
        return self.rate_func

    def set_name(self, name: str) -> "Animation":
        self.name = name
        return self

    def is_remover(self) -> bool:
        return self.remover


def prepare_animation(
    anim: Union["Animation", "mobject._AnimationBuilder"]
) -> "Animation":
    r"""Returns either an unchanged animation, or the animation built
    from a passed animation factory.

    Examples
    --------

    ::

        >>> from manim import Square, FadeIn
        >>> s = Square()
        >>> prepare_animation(FadeIn(s))
        FadeIn(Square)

    ::

        >>> prepare_animation(s.animate.scale(2).rotate(42))
        _MethodAnimation(Square)

    ::

        >>> prepare_animation(42)
        Traceback (most recent call last):
        ...
        TypeError: Object 42 cannot be converted to an animation

    """
    if isinstance(anim, mobject._AnimationBuilder):
        return anim.build()

    if isinstance(anim, opengl_mobject._AnimationBuilder):
        return anim.build()

    if isinstance(anim, Animation):
        return anim

    raise TypeError(f"Object {anim} cannot be converted to an animation")


class Wait(Animation):
    def __init__(
        self, run_time: float = 1, stop_condition=None, **kwargs
    ):  # what is stop_condition?
        self.duration: float = run_time
        self.stop_condition = stop_condition
        self.is_static_wait: bool = False
        super().__init__(None, run_time=run_time, **kwargs)
        # quick fix to work in opengl setting:
        self.mobject.shader_wrapper_list = []

    def begin(self) -> None:
        pass

    def finish(self) -> None:
        pass

    def clean_up_from_scene(self, scene: "Scene") -> None:
        pass

    def update_mobjects(self, dt: float) -> None:
        pass

    def interpolate(self, alpha: float) -> None:
        pass


def _all_subclasses(cls, root=True):
    lst = cls.__subclasses__() + [
        s for c in cls.__subclasses__() for s in _all_subclasses(c, False)
    ]
    if root:
        return list(dict.fromkeys([cls] + lst))
    else:
        return lst


def _setup():
    overrides = {}
    i = 0
    for mobject_class in _all_subclasses(Mobject):
        for method_name in dir(mobject_class):
            i += 1
            method = getattr(mobject_class, method_name)
            if hasattr(method, "_override_animation"):
                animation_class = method._override_animation

                if animation_class not in overrides:
                    overrides[animation_class] = {}
                if mobject_class in overrides[animation_class]:
                    msg = (
                        f"The animation {animation_class.__name__} for "
                        f"{mobject_class.__name__} is overridden by more than one method:"
                        f" {overrides[animation_class][mobject_class].__qualname__} "
                        f"and {method.__qualname__}. If one of these methods is "
                        f"inherited make sure they are named equally."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
                overrides[animation_class][mobject_class] = method
    Animation._overrides = overrides


def override_animation(animationClass):
    def decorator(func):
        func._override_animation = animationClass
        return func

    return decorator


override_animation.setup = _setup
