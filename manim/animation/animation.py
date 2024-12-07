"""Animate mobjects."""

from __future__ import annotations

from manim.mobject.opengl.opengl_mobject import OpenGLMobject

from .. import config, logger
from ..constants import RendererType
from ..mobject import mobject
from ..mobject.mobject import Group, Mobject
from ..mobject.opengl import opengl_mobject
from ..utils.rate_functions import linear, smooth

__all__ = ["Animation", "Wait", "Add", "override_animation"]


from collections.abc import Iterable, Sequence
from copy import deepcopy
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Callable

from typing_extensions import Self

if TYPE_CHECKING:
    from manim.scene.scene import Scene


DEFAULT_ANIMATION_RUN_TIME: float = 1.0
DEFAULT_ANIMATION_LAG_RATIO: float = 0.0


class Animation:
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
        within :meth:`.Animation.interpolate_mobject` call as part of the call to
        :meth:`.Animation.interpolate_submobject`. For subclasses of :class:`.Animation`
        that are implemented by overriding :meth:`interpolate_mobject`, the rate function
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
                    f"The {cls.__name__} animation has been is overridden for "
                    f"{type(mobject).__name__} mobjects. use_override = False can "
                    f" be used as keyword argument to prevent animation overriding.",
                )
                return anim
        return super().__new__(cls)

    def __init__(
        self,
        mobject: Mobject | None,
        lag_ratio: float = DEFAULT_ANIMATION_LAG_RATIO,
        run_time: float = DEFAULT_ANIMATION_RUN_TIME,
        rate_func: Callable[[float], float] = smooth,
        reverse_rate_function: bool = False,
        name: str = None,
        remover: bool = False,  # remove a mobject from the screen?
        suspend_mobject_updating: bool = True,
        introducer: bool = False,
        *,
        _on_finish: Callable[[], None] = lambda _: None,
        **kwargs,
    ) -> None:
        self._typecheck_input(mobject)
        self.run_time: float = run_time
        self.rate_func: Callable[[float], float] = rate_func
        self.reverse_rate_function: bool = reverse_rate_function
        self.name: str | None = name
        self.remover: bool = remover
        self.introducer: bool = introducer
        self.suspend_mobject_updating: bool = suspend_mobject_updating
        self.lag_ratio: float = lag_ratio
        self._on_finish: Callable[[Scene], None] = _on_finish
        if config["renderer"] == RendererType.OPENGL:
            self.starting_mobject: OpenGLMobject = OpenGLMobject()
            self.mobject: OpenGLMobject = (
                mobject if mobject is not None else OpenGLMobject()
            )
        else:
            self.starting_mobject: Mobject = Mobject()
            self.mobject: Mobject = mobject if mobject is not None else Mobject()
        if kwargs:
            logger.debug("Animation received extra kwargs: %s", kwargs)

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

    def finish(self) -> None:
        # TODO: begin and finish should require a scene as parameter.
        # That way Animation.clean_up_from_screen and Scene.add_mobjects_from_animations
        # could be removed as they fulfill basically the same purpose.
        """Finish the animation.

        This method gets called when the animation is over.

        """
        self.interpolate(1)
        if self.suspend_mobject_updating and self.mobject is not None:
            self.mobject.resume_updating()

    def clean_up_from_scene(self, scene: Scene) -> None:
        """Clean up the :class:`~.Scene` after finishing the animation.

        This includes to :meth:`~.Scene.remove` the Animation's
        :class:`~.Mobject` if the animation is a remover.

        Parameters
        ----------
        scene
            The scene the animation should be cleaned up from.
        """
        self._on_finish(scene)
        if self.is_remover():
            scene.remove(self.mobject)

    def _setup_scene(self, scene: Scene) -> None:
        """Setup up the :class:`~.Scene` before starting the animation.

        This includes to :meth:`~.Scene.add` the Animation's
        :class:`~.Mobject` if the animation is an introducer.

        Parameters
        ----------
        scene
            The scene the animation should be cleaned up from.
        """
        if scene is None:
            return
        if (
            self.is_introducer()
            and self.mobject not in scene.get_mobject_family_members()
        ):
            scene.add(self.mobject)

    def create_starting_mobject(self) -> Mobject:
        # Keep track of where the mobject starts
        return self.mobject.copy()

    def get_all_mobjects(self) -> Sequence[Mobject]:
        """Get all mobjects involved in the animation.

        Ordering must match the ordering of arguments to interpolate_submobject

        Returns
        -------
        Sequence[Mobject]
            The sequence of mobjects.
        """
        return self.mobject, self.starting_mobject

    def get_all_families_zipped(self) -> Iterable[tuple]:
        if config["renderer"] == RendererType.OPENGL:
            return zip(*(mob.get_family() for mob in self.get_all_mobjects()))
        return zip(
            *(mob.family_members_with_points() for mob in self.get_all_mobjects())
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

    def get_all_mobjects_to_update(self) -> list[Mobject]:
        """Get all mobjects to be updated during the animation.

        Returns
        -------
        List[Mobject]
            The list of mobjects to be updated during the animation.
        """
        # The surrounding scene typically handles
        # updating of self.mobject.  Besides, in
        # most cases its updating is suspended anyway
        return list(filter(lambda m: m is not self.mobject, self.get_all_mobjects()))

    def copy(self) -> Animation:
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
        """Set the animation progress.

        This method gets called for every frame during an animation.

        Parameters
        ----------
        alpha
            The relative time to set the animation to, 0 meaning the start, 1 meaning
            the end.
        """
        self.interpolate_mobject(alpha)

    def interpolate_mobject(self, alpha: float) -> None:
        """Interpolates the mobject of the :class:`Animation` based on alpha value.

        Parameters
        ----------
        alpha
            A float between 0 and 1 expressing the ratio to which the animation
            is completed. For example, alpha-values of 0, 0.5, and 1 correspond
            to the animation being completed 0%, 50%, and 100%, respectively.
        """
        families = list(self.get_all_families_zipped())
        for i, mobs in enumerate(families):
            sub_alpha = self.get_sub_alpha(alpha, i, len(families))
            self.interpolate_submobject(*mobs, sub_alpha)

    def interpolate_submobject(
        self,
        submobject: Mobject,
        starting_submobject: Mobject,
        # target_copy: Mobject, #Todo: fix - signature of interpolate_submobject differs in Transform().
        alpha: float,
    ) -> Animation:
        # Typically implemented by subclass
        pass

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
        if self.reverse_rate_function:
            return self.rate_func(1 - (value - lower))
        else:
            return self.rate_func(value - lower)

    # Getters and setters
    def set_run_time(self, run_time: float) -> Animation:
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
    ) -> Animation:
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

    def set_name(self, name: str) -> Animation:
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

    def is_remover(self) -> bool:
        """Test if the animation is a remover.

        Returns
        -------
        bool
            ``True`` if the animation is a remover, ``False`` otherwise.
        """
        return self.remover

    def is_introducer(self) -> bool:
        """Test if the animation is an introducer.

        Returns
        -------
        bool
            ``True`` if the animation is an introducer, ``False`` otherwise.
        """
        return self.introducer

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        cls._original__init__ = cls.__init__

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


def prepare_animation(
    anim: Animation | mobject._AnimationBuilder,
) -> Animation:
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

        self.duration: float = run_time
        self.stop_condition = stop_condition
        self.is_static_wait: bool = frozen_frame
        super().__init__(None, run_time=run_time, rate_func=rate_func, **kwargs)
        # quick fix to work in opengl setting:
        self.mobject.shader_wrapper_list = []

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

    def decorator(func):
        func._override_animation = animation_class
        return func

    return decorator
