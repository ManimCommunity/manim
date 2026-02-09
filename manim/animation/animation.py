"""Animate mobjects."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Self, assert_never, cast, overload

import numpy as np
from typing_extensions import TypeVar

from manim.mobject.opengl.opengl_mobject import OpenGLMobject

from .. import logger
from ..mobject import mobject
from ..mobject.mobject import Group, Mobject
from ..mobject.opengl import opengl_mobject
from ..utils.rate_functions import linear, smooth
from .protocol import AnimationProtocol, MobjectAnimation
from .scene_buffer import SceneBuffer, SceneOperation

if TYPE_CHECKING:
    from typing import Self

    from manim.scene.scene import Scene

M = TypeVar("M", bound=OpenGLMobject)


__all__ = ["Animation", "Wait", "override_animation"]


DEFAULT_ANIMATION_RUN_TIME: float = 1.0
DEFAULT_ANIMATION_LAG_RATIO: float = 0.0


class Animation(AnimationProtocol):
    """An animation.

    Animations have a fixed time span.

    Parameters
    ----------
    mobject
        The mobject to be animated. This is not required for all types of animations.
    lag_ratio
        Defines the delay after which the animation is applied to submobjects. This lag
        is relative to the duration of the animation.

        This does not influence the total
        runtime of the animation. Instead the runtime of individual animations is
        adjusted so that the complete animation has the defined run time.

    run_time
        The duration of the animation in seconds.
    rate_func
        The function defining the animation progress based on the relative runtime (see  :mod:`~.rate_functions`) .

        For example ``rate_func(0.5)`` is the proportion of the animation that is done
        after half of the animations run time.

    reverse_rate_function
        Reverses the rate function of the animation. Setting ``reverse_rate_function``
        does not have any effect on ``remover`` or ``introducer``. These need to be
        set explicitly if an introducer-animation should be turned into a remover one
        and vice versa.
    name
        The name of the animation. This gets displayed while rendering the animation.
        Defaults to <class-name>(<Mobject-name>).
    remover
        Whether the given mobject should be removed from the scene after this animation.
    suspend_mobject_updating
        Whether updaters of the mobject should be suspended during the animation.


    .. NOTE::

        In the current implementation of this class, the specified rate function is applied
        within :meth:`.Animation.interpolate` call as part of the call to
        :meth:`.Animation.interpolate_submobject`. For subclasses of :class:`.Animation`
        that are implemented by overriding :meth:`interpolate`, the rate function
        has to be applied manually (e.g., by passing ``self.rate_func(alpha)`` instead
        of just ``alpha``).


    Examples
    --------

    .. manim:: LagRatios

        class LagRatios(Scene):
            def construct(self):
                ratios = [0, 0.1, 0.5, 1, 2]  # demonstrated lag_ratios

                # Create dot groups
                group = VGroup(*[Dot() for _ in range(4)]).arrange_submobjects()
                groups = VGroup(*[group.copy() for _ in ratios]).arrange_submobjects(buff=1)
                self.add(groups)

                # Label groups
                self.add(Text("lag_ratio = ", font_size=36).next_to(groups, UP, buff=1.5))
                for group, ratio in zip(groups, ratios):
                    self.add(Text(str(ratio), font_size=36).next_to(group, UP))

                #Animate groups with different lag_ratios
                self.play(AnimationGroup(*[
                    group.animate(lag_ratio=ratio, run_time=1.5).shift(DOWN * 2)
                    for group, ratio in zip(groups, ratios)
                ]))

                # lag_ratio also works recursively on nested submobjects:
                self.play(groups.animate(run_time=1, lag_ratio=0.1).shift(UP * 2))

    """

    def __new__(
        cls,
        mobject=None,
        *args,
        use_override=True,
        **kwargs,
    ) -> Self:
        if isinstance(mobject, Mobject) and use_override:
            func = mobject.animation_override_for(cls)
            if func is not None:
                anim = func(mobject, *args, **kwargs)
                logger.debug(
                    f"The {cls.__name__} animation has been overridden for "
                    f"{type(mobject).__name__} mobjects. use_override = False can "
                    f" be used as keyword argument to prevent animation overriding.",
                )
                return anim
        return super().__new__(cls)

    def __init__(
        self,
        mobject: OpenGLMobject | None,
        lag_ratio: float = DEFAULT_ANIMATION_LAG_RATIO,
        run_time: float = DEFAULT_ANIMATION_RUN_TIME,
        rate_func: Callable[[float], float] = smooth,
        reverse_rate_function: bool = False,
        name: str = "",
        remover: bool = False,  # remove a mobject from the screen at end of animation
        suspend_mobject_updating: bool = True,
        introducer: bool = False,
        *,
        _on_finish: Callable[[SceneBuffer], object] = lambda _: None,
        use_override: bool = True,  # included here to avoid TypeError if passed from a subclass' constructor
    ) -> None:
        self._typecheck_input(mobject)
        self.run_time: float = run_time
        self.rate_func: Callable[[float], float] = rate_func
        self.reverse_rate_function: bool = reverse_rate_function
        self.name: str = name
        self.remover: bool = remover
        self.introducer: bool = introducer
        self.suspend_mobject_updating: bool = suspend_mobject_updating
        self.lag_ratio: float = lag_ratio
        self._on_finish = _on_finish

        self.buffer = SceneBuffer()
        self.apply_buffer = False  # ask scene to apply buffer

        self.starting_mobject: OpenGLMobject = OpenGLMobject()
        self.mobject: OpenGLMobject = (
            mobject if mobject is not None else OpenGLMobject()
        )

        if hasattr(self, "CONFIG"):
            logger.error(
                (
                    "CONFIG has been removed from ManimCommunity.",
                    "Please use keyword arguments instead.",
                ),
            )

    @property
    def run_time(self) -> float:
        return self._run_time

    @run_time.setter
    def run_time(self, value: float) -> None:
        if value < 0:
            raise ValueError(
                f"The run_time of {self.__class__.__name__} cannot be "
                f"negative. The given value was {value}."
            )
        self._run_time = value

    def _typecheck_input(self, mobject: Mobject | None) -> None:
        if mobject is None:
            logger.debug("Animation with empty mobject")
        elif not isinstance(mobject, (Mobject, OpenGLMobject)):
            raise TypeError("Animation only works on Mobjects")

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f"{self.__class__.__name__}({str(self.mobject)})"

    def __repr__(self) -> str:
        return str(self)

    def update_rate_info(
        self,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
    ):
        self.run_time = run_time or self.run_time
        self.rate_func = rate_func or self.rate_func
        self.lag_ratio = lag_ratio or self.lag_ratio
        return self

    def begin(self) -> None:
        """Begin the animation.

        This method is called right as an animation is being played. As much
        initialization as possible, especially any mobject copying, should live in this
        method.

        """
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

        # TODO: Figure out a way to check
        # if self.mobject in scene.get_mobject_family
        if self.introducer:
            self.buffer.add(self.mobject)

    def finish(self) -> None:
        """Finish the animation.

        This method gets called when the animation is over.

        """
        self.interpolate(1)
        if self.suspend_mobject_updating and self.mobject is not None:
            self.mobject.resume_updating()

        # TODO: remove on_finish
        self._on_finish(self.buffer)
        if self.remover:
            self.buffer.remove(self.mobject)

    def create_starting_mobject(self) -> OpenGLMobject:
        # Keep track of where the mobject starts
        return self.mobject.copy()

    def get_all_mobjects(self) -> Sequence[OpenGLMobject]:
        """Get all mobjects involved in the animation.

        Ordering must match the ordering of arguments to interpolate_submobject

        Returns
        -------
        Sequence[Mobject]
            The sequence of mobjects.
        """
        return self.mobject, self.starting_mobject

    def get_all_families_zipped(self) -> Iterable[tuple]:
        return zip(*(mob.get_family() for mob in self.get_all_mobjects()), strict=False)

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

    def process_subanimation_buffer(self, buffer: SceneBuffer):
        """
        This is used in animations that are proxies around
        other animations, like :class:`.AnimationGroup`
        """
        for op, args, kwargs in buffer:
            match op:
                case SceneOperation.ADD:
                    self.buffer.add(*args, **kwargs)
                case SceneOperation.REMOVE:
                    self.buffer.remove(*args, **kwargs)
                case SceneOperation.REPLACE:
                    self.buffer.replace(*args, **kwargs)
                case _:
                    assert_never(op)
        buffer.clear()

    def get_all_mobjects_to_update(self) -> Sequence[OpenGLMobject]:
        """Get all mobjects to be updated during the animation.

        Returns
        -------
        List[Mobject]
            The list of mobjects to be updated during the animation.
        """
        # The surrounding scene typically handles
        # updating of self.mobject.  Besides, in
        # most cases its updating is suspended anyway
        return [m for m in self.get_all_mobjects() if m is not self.mobject]

    def copy(self) -> Self:
        """Create a copy of the animation.

        Returns
        -------
        Animation
            A copy of ``self``
        """
        return deepcopy(self)

    # Methods for interpolation, the mean of an Animation

    # TODO: stop using alpha as parameter name in different meanings.
    def interpolate(self, alpha: float) -> None:
        """Interpolates the mobject of the :class:`Animation` based on alpha value.

        Parameters
        ----------
        alpha
            A float between 0 and 1 expressing the ratio to which the animation
            is completed. For example, alpha-values of 0, 0.5, and 1 correspond
            to the animation being completed 0%, 50%, and 100%, respectively.
        """
        families = tuple(self.get_all_families_zipped())
        for i, mobs in enumerate(families):
            sub_alpha = self.get_sub_alpha(alpha, i, len(families))
            self.interpolate_submobject(*mobs, sub_alpha)  # type: ignore

    def interpolate_submobject(
        self,
        submobject: OpenGLMobject,
        starting_submobject: OpenGLMobject,
        # target_copy: Mobject, #Todo: fix - signature of interpolate_submobject differs in Transform().
        alpha: float,
    ) -> Animation:
        raise NotImplementedError("Implement in subclass")

    def get_sub_alpha(self, alpha: float, index: int, num_submobjects: int) -> float:
        """Get the animation progress of any submobjects subanimation.

        Parameters
        ----------
        alpha
            The overall animation progress
        index
            The index of the subanimation.
        num_submobjects
            The total count of subanimations.

        Returns
        -------
        float
            The progress of the subanimation.
        """
        # TODO, make this more understandable, and/or combine
        # its functionality with AnimationGroup's method
        # build_animations_with_timings
        lag_ratio = self.lag_ratio
        full_length = (num_submobjects - 1) * lag_ratio + 1
        value = alpha * full_length
        lower = index * lag_ratio
        raw_sub_alpha = np.clip((value - lower), 0, 1)
        if self.reverse_rate_function:
            return self.rate_func(1 - raw_sub_alpha)
        else:
            return self.rate_func(raw_sub_alpha)

    # Getters and setters
    def set_run_time(self, run_time: float) -> Self:
        """Set the run time of the animation.

        Parameters
        ----------
        run_time
            The new time the animation should take in seconds.

        .. note::

            The run_time of an animation should not be changed while it is already
            running.

        Returns
        -------
        Animation
            ``self``
        """
        self.run_time = run_time
        return self

    # TODO: is this getter even necessary?
    def get_run_time(self) -> float:
        """Get the run time of the animation.

        Returns
        -------
        float
            The time the animation takes in seconds.
        """
        return self.run_time

    def set_rate_func(
        self,
        rate_func: Callable[[float], float],
    ) -> Self:
        """Set the rate function of the animation.

        Parameters
        ----------
        rate_func
            The new function defining the animation progress based on the
            relative runtime (see :mod:`~.rate_functions`).

        Returns
        -------
        Animation
            ``self``
        """
        self.rate_func = rate_func
        return self

    def get_rate_func(
        self,
    ) -> Callable[[float], float]:
        """Get the rate function of the animation.

        Returns
        -------
        Callable[[float], float]
            The rate function of the animation.
        """
        return self.rate_func

    def set_name(self, name: str) -> Self:
        """Set the name of the animation.

        Parameters
        ----------
        name
            The new name of the animation.

        Returns
        -------
        Animation
            ``self``
        """
        self.name = name
        return self

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        cls._original__init__ = cls.__init__

    _original__init__ = __init__  # needed if set_default() is called with no kwargs directly from Animation

    @classmethod
    def set_default(cls, **kwargs) -> None:
        """Sets the default values of keyword arguments.

        If this method is called without any additional keyword
        arguments, the original default values of the initialization
        method of this class are restored.

        Parameters
        ----------

        kwargs
            Passing any keyword argument will update the default
            values of the keyword arguments of the initialization
            function of this class.

        Examples
        --------

        .. manim:: ChangeDefaultAnimation

            class ChangeDefaultAnimation(Scene):
                def construct(self):
                    Rotate.set_default(run_time=2, rate_func=rate_functions.linear)
                    Indicate.set_default(color=None)

                    S = Square(color=BLUE, fill_color=BLUE, fill_opacity=0.25)
                    self.add(S)
                    self.play(Rotate(S, PI))
                    self.play(Indicate(S))

                    Rotate.set_default()
                    Indicate.set_default()

        """
        if kwargs:
            cls.__init__ = partialmethod(cls.__init__, **kwargs)
        else:
            cls.__init__ = cls._original__init__


@overload
def prepare_animation(anim: MobjectAnimation[M]) -> MobjectAnimation[M]: ...


@overload
def prepare_animation(
    anim: AnimationProtocol
    | opengl_mobject._AnimationBuilder
    | opengl_mobject.OpenGLMobject,
) -> AnimationProtocol: ...


def prepare_animation(
    anim: AnimationProtocol
    | opengl_mobject._AnimationBuilder
    | opengl_mobject.OpenGLMobject,
) -> AnimationProtocol:
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
    if isinstance(anim, (mobject._AnimationBuilder, opengl_mobject._AnimationBuilder)):
        return anim.build()

    # if it has these three methods it probably is an AnimationProtocol
    # but we don't use isinstance because it's slow
    try:
        for method in ("begin", "finish", "update_mobjects"):
            getattr(anim, method)
        return cast(AnimationProtocol, anim)
    except AttributeError:
        raise TypeError(f"Object {anim} cannot be converted to an animation") from None


class Wait(Animation):
    """A "no operation" animation.

    Parameters
    ----------
    run_time
        The amount of time that should pass.
    stop_condition
        A function without positional arguments that evaluates to a boolean.
        The function is evaluated after every new frame has been rendered.
        Playing the animation stops after the return value is truthy, or
        after the specified ``run_time`` has passed.
    frozen_frame
        Controls whether or not the wait animation is static, i.e., corresponds
        to a frozen frame. If ``False`` is passed, the render loop still
        progresses through the animation as usual and (among other things)
        continues to call updater functions. If ``None`` (the default value),
        the :meth:`.Scene.play` call tries to determine whether the Wait call
        can be static or not itself via :meth:`.Scene.should_mobjects_update`.
    kwargs
        Keyword arguments to be passed to the parent class, :class:`.Animation`.
    """

    def __init__(
        self,
        run_time: float = 1,
        stop_condition: Callable[[], bool] | None = None,
        frozen_frame: bool | None = None,
        rate_func: Callable[[float], float] = linear,
        **kwargs,
    ):
        if stop_condition and frozen_frame:
            raise ValueError("A static Wait animation cannot have a stop condition.")

        self.stop_condition = stop_condition
        self.is_static_wait: bool = bool(frozen_frame)
        super().__init__(None, run_time=run_time, rate_func=rate_func, **kwargs)

    def begin(self) -> None:
        pass

    def finish(self) -> None:
        pass

    def update_mobjects(self, dt: float) -> None:
        pass

    def interpolate(self, alpha: float) -> None:
        pass


class Add(Animation):
    """Add Mobjects to a scene, without animating them in any other way. This
    is similar to the :meth:`.Scene.add` method, but :class:`Add` is an
    animation which can be grouped into other animations.

    Parameters
    ----------
    mobjects
        One :class:`~.Mobject` or more to add to a scene.
    run_time
        The duration of the animation after adding the ``mobjects``. Defaults
        to 0, which means this is an instant animation without extra wait time
        after adding them.
    **kwargs
        Additional arguments to pass to the parent :class:`Animation` class.

    Examples
    --------

    .. manim:: DefaultAddScene

        class DefaultAddScene(Scene):
            def construct(self):
                text_1 = Text("I was added with Add!")
                text_2 = Text("Me too!")
                text_3 = Text("And me!")
                texts = VGroup(text_1, text_2, text_3).arrange(DOWN)
                rect = SurroundingRectangle(texts, buff=0.5)

                self.play(
                    Create(rect, run_time=3.0),
                    Succession(
                        Wait(1.0),
                        # You can Add a Mobject in the middle of an animation...
                        Add(text_1),
                        Wait(1.0),
                        # ...or multiple Mobjects at once!
                        Add(text_2, text_3),
                    ),
                )
                self.wait()

    .. manim:: AddWithRunTimeScene

        class AddWithRunTimeScene(Scene):
            def construct(self):
                # A 5x5 grid of circles
                circles = VGroup(
                    *[Circle(radius=0.5) for _ in range(25)]
                ).arrange_in_grid(5, 5)

                self.play(
                    Succession(
                        # Add a run_time of 0.2 to wait for 0.2 seconds after
                        # adding the circle, instead of using Wait(0.2) after Add!
                        *[Add(circle, run_time=0.2) for circle in circles],
                        rate_func=smooth,
                    )
                )
                self.wait()
    """

    def __init__(
        self, *mobjects: Mobject, run_time: float = 0.0, **kwargs: Any
    ) -> None:
        mobject = mobjects[0] if len(mobjects) == 1 else Group(*mobjects)
        super().__init__(mobject, run_time=run_time, introducer=True, **kwargs)

    def begin(self) -> None:
        pass

    def finish(self) -> None:
        pass

    def clean_up_from_scene(self, scene: Scene) -> None:
        pass

    def update_mobjects(self, dt: float) -> None:
        pass

    def interpolate(self, alpha: float) -> None:
        pass


def override_animation(
    animation_class: type[Animation],
) -> Callable[[Callable], Callable]:
    """Decorator used to mark methods as overrides for specific :class:`~.Animation` types.

    Should only be used to decorate methods of classes derived from :class:`~.Mobject`.
    ``Animation`` overrides get inherited to subclasses of the ``Mobject`` who defined
    them. They don't override subclasses of the ``Animation`` they override.

    See Also
    --------
    :meth:`~.Mobject.add_animation_override`

    Parameters
    ----------
    animation_class
        The animation to be overridden.

    Returns
    -------
    Callable[[Callable], Callable]
        The actual decorator. This marks the method as overriding an animation.

    Examples
    --------

    .. manim:: OverrideAnimationExample

        class MySquare(Square):
            @override_animation(FadeIn)
            def _fade_in_override(self, **kwargs):
                return Create(self, **kwargs)

        class OverrideAnimationExample(Scene):
            def construct(self):
                self.play(FadeIn(MySquare()))

    """
    _F = TypeVar("_F", bound=Callable)

    def decorator(func: _F) -> _F:
        func._override_animation = animation_class  # type: ignore
        return func

    return decorator
