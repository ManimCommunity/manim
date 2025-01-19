"""Base classes for objects that can be displayed."""

from __future__ import annotations

__all__ = ["Mobject", "Group", "override_animate"]


import copy
import inspect
import itertools as it
import math
import operator as op
import random
import sys
import types
import warnings
from collections.abc import Iterable
from functools import partialmethod, reduce
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL

from .. import config, logger
from ..constants import *
from ..utils.color import (
    BLACK,
    WHITE,
    YELLOW_C,
    ManimColor,
    ParsableManimColor,
    color_gradient,
    interpolate_color,
)
from ..utils.exceptions import MultiAnimationOverrideException
from ..utils.iterables import list_update, remove_list_redundancies
from ..utils.paths import straight_path
from ..utils.space_ops import angle_between_vectors, normalize, rotation_matrix

if TYPE_CHECKING:
    from typing import Any, Callable, Literal

    from typing_extensions import Self, TypeAlias

    from manim.typing import (
        FunctionOverride,
        MappingFunction,
        MultiMappingFunction,
        PathFuncType,
        PixelArray,
        Point3D,
        Point3DLike,
        Point3DLike_Array,
        Vector3D,
    )

    from ..animation.animation import Animation

    TimeBasedUpdater: TypeAlias = Callable[["Mobject", float], object]
    NonTimeBasedUpdater: TypeAlias = Callable[["Mobject"], object]
    Updater: TypeAlias = NonTimeBasedUpdater | TimeBasedUpdater


class Mobject:
    """Mathematical Object: base class for objects that can be displayed on screen.

    There is a compatibility layer that allows for
    getting and setting generic attributes with ``get_*``
    and ``set_*`` methods. See :meth:`set` for more details.

    Attributes
    ----------
    submobjects : List[:class:`Mobject`]
        The contained objects.
    points : :class:`numpy.ndarray`
        The points of the objects.

        .. seealso::

            :class:`~.VMobject`

    """

    animation_overrides = {}

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        cls.animation_overrides: dict[
            type[Animation],
            FunctionOverride,
        ] = {}
        cls._add_intrinsic_animation_overrides()
        cls._original__init__ = cls.__init__

    def __init__(
        self,
        color: ParsableManimColor | list[ParsableManimColor] = WHITE,
        name: str | None = None,
        dim: int = 3,
        target=None,
        z_index: float = 0,
    ) -> None:
        self.name = self.__class__.__name__ if name is None else name
        self.dim = dim
        self.target = target
        self.z_index = z_index
        self.point_hash = None
        self.submobjects = []
        self.updaters: list[Updater] = []
        self.updating_suspended = False
        self.color = ManimColor.parse(color)

        self.reset_points()
        self.generate_points()
        self.init_colors()

    def _assert_valid_submobjects(self, submobjects: Iterable[Mobject]) -> Self:
        """Check that all submobjects are actually instances of
        :class:`Mobject`, and that none of them is ``self`` (a
        :class:`Mobject` cannot contain itself).

        This is an auxiliary function called when adding Mobjects to the
        :attr:`submobjects` list.

        This function is intended to be overridden by subclasses such as
        :class:`VMobject`, which should assert that only other VMobjects
        may be added into it.

        Parameters
        ----------
        submobjects
            The list containing values to validate.

        Returns
        -------
        :class:`Mobject`
            The Mobject itself.

        Raises
        ------
        TypeError
            If any of the values in `submobjects` is not a :class:`Mobject`.
        ValueError
            If there was an attempt to add a :class:`Mobject` as its own
            submobject.
        """
        return self._assert_valid_submobjects_internal(submobjects, Mobject)

    def _assert_valid_submobjects_internal(
        self, submobjects: list[Mobject], mob_class: type[Mobject]
    ) -> Self:
        for i, submob in enumerate(submobjects):
            if not isinstance(submob, mob_class):
                error_message = (
                    f"Only values of type {mob_class.__name__} can be added "
                    f"as submobjects of {type(self).__name__}, but the value "
                    f"{submob} (at index {i}) is of type "
                    f"{type(submob).__name__}."
                )
                # Intended for subclasses such as VMobject, which
                # cannot have regular Mobjects as submobjects
                if isinstance(submob, Mobject):
                    error_message += (
                        " You can try adding this value into a Group instead."
                    )
                raise TypeError(error_message)
            if submob is self:
                raise ValueError(
                    f"Cannot add {type(self).__name__} as a submobject of "
                    f"itself (at index {i})."
                )
        return self

    @classmethod
    def animation_override_for(
        cls,
        animation_class: type[Animation],
    ) -> FunctionOverride | None:
        """Returns the function defining a specific animation override for this class.

        Parameters
        ----------
        animation_class
            The animation class for which the override function should be returned.

        Returns
        -------
        Optional[Callable[[Mobject, ...], Animation]]
            The function returning the override animation or ``None`` if no such animation
            override is defined.
        """
        if animation_class in cls.animation_overrides:
            return cls.animation_overrides[animation_class]

        return None

    @classmethod
    def _add_intrinsic_animation_overrides(cls) -> None:
        """Initializes animation overrides marked with the :func:`~.override_animation`
        decorator.
        """
        for method_name in dir(cls):
            # Ignore dunder methods
            if method_name.startswith("__"):
                continue

            method = getattr(cls, method_name)
            if hasattr(method, "_override_animation"):
                animation_class = method._override_animation
                cls.add_animation_override(animation_class, method)

    @classmethod
    def add_animation_override(
        cls,
        animation_class: type[Animation],
        override_func: FunctionOverride,
    ) -> None:
        """Add an animation override.

        This does not apply to subclasses.

        Parameters
        ----------
        animation_class
            The animation type to be overridden
        override_func
            The function returning an animation replacing the default animation. It gets
            passed the parameters given to the animation constructor.

        Raises
        ------
        MultiAnimationOverrideException
            If the overridden animation was already overridden.
        """
        if animation_class not in cls.animation_overrides:
            cls.animation_overrides[animation_class] = override_func
        else:
            raise MultiAnimationOverrideException(
                f"The animation {animation_class.__name__} for "
                f"{cls.__name__} is overridden by more than one method: "
                f"{cls.animation_overrides[animation_class].__qualname__} and "
                f"{override_func.__qualname__}.",
            )

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

        ::

            >>> from manim import Square, GREEN
            >>> Square.set_default(color=GREEN, fill_opacity=0.25)
            >>> s = Square(); s.color, s.fill_opacity
            (ManimColor('#83C167'), 0.25)
            >>> Square.set_default()
            >>> s = Square(); s.color, s.fill_opacity
            (ManimColor('#FFFFFF'), 0.0)

        .. manim:: ChangedDefaultTextcolor
            :save_last_frame:

            config.background_color = WHITE

            class ChangedDefaultTextcolor(Scene):
                def construct(self):
                    Text.set_default(color=BLACK)
                    self.add(Text("Changing default values is easy!"))

                    # we revert the colour back to the default to prevent a bug in the docs.
                    Text.set_default(color=WHITE)

        """
        if kwargs:
            cls.__init__ = partialmethod(cls.__init__, **kwargs)
        else:
            cls.__init__ = cls._original__init__

    @property
    def animate(self) -> _AnimationBuilder | Self:
        """Used to animate the application of any method of :code:`self`.

        Any method called on :code:`animate` is converted to an animation of applying
        that method on the mobject itself.

        For example, :code:`square.set_fill(WHITE)` sets the fill color of a square,
        while :code:`square.animate.set_fill(WHITE)` animates this action.

        Multiple methods can be put in a single animation once via chaining:

        ::

            self.play(my_mobject.animate.shift(RIGHT).rotate(PI))

        .. warning::

            Passing multiple animations for the same :class:`Mobject` in one
            call to :meth:`~.Scene.play` is discouraged and will most likely
            not work properly. Instead of writing an animation like

            ::

                self.play(
                    my_mobject.animate.shift(RIGHT), my_mobject.animate.rotate(PI)
                )

            make use of method chaining.

        Keyword arguments that can be passed to :meth:`.Scene.play` can be passed
        directly after accessing ``.animate``, like so::

            self.play(my_mobject.animate(rate_func=linear).shift(RIGHT))

        This is especially useful when animating simultaneous ``.animate`` calls that
        you want to behave differently::

            self.play(
                mobject1.animate(run_time=2).rotate(PI),
                mobject2.animate(rate_func=there_and_back).shift(RIGHT),
            )

        .. seealso::

            :func:`override_animate`


        Examples
        --------

        .. manim:: AnimateExample

            class AnimateExample(Scene):
                def construct(self):
                    s = Square()
                    self.play(Create(s))
                    self.play(s.animate.shift(RIGHT))
                    self.play(s.animate.scale(2))
                    self.play(s.animate.rotate(PI / 2))
                    self.play(Uncreate(s))


        .. manim:: AnimateChainExample

            class AnimateChainExample(Scene):
                def construct(self):
                    s = Square()
                    self.play(Create(s))
                    self.play(s.animate.shift(RIGHT).scale(2).rotate(PI / 2))
                    self.play(Uncreate(s))

        .. manim:: AnimateWithArgsExample

            class AnimateWithArgsExample(Scene):
                def construct(self):
                    s = Square()
                    c = Circle()

                    VGroup(s, c).arrange(RIGHT, buff=2)
                    self.add(s, c)

                    self.play(
                        s.animate(run_time=2).rotate(PI / 2),
                        c.animate(rate_func=there_and_back).shift(RIGHT),
                    )

        .. warning::

            ``.animate``
             will interpolate the :class:`~.Mobject` between its points prior to
             ``.animate`` and its points after applying ``.animate`` to it. This may
             result in unexpected behavior when attempting to interpolate along paths,
             or rotations.
             If you want animations to consider the points between, consider using
             :class:`~.ValueTracker` with updaters instead.

        """
        return _AnimationBuilder(self)

    def __deepcopy__(self, clone_from_id) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        clone_from_id[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, clone_from_id))
        result.original_id = str(id(self))
        return result

    def __repr__(self) -> str:
        return str(self.name)

    def reset_points(self) -> None:
        """Sets :attr:`points` to be an empty array."""
        self.points = np.zeros((0, self.dim))

    def init_colors(self) -> object:
        """Initializes the colors.

        Gets called upon creation. This is an empty method that can be implemented by
        subclasses.
        """

    def generate_points(self) -> object:
        """Initializes :attr:`points` and therefore the shape.

        Gets called upon creation. This is an empty method that can be implemented by
        subclasses.
        """

    def add(self, *mobjects: Mobject) -> Self:
        """Add mobjects as submobjects.

        The mobjects are added to :attr:`submobjects`.

        Subclasses of mobject may implement ``+`` and ``+=`` dunder methods.

        Parameters
        ----------
        mobjects
            The mobjects to add.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Raises
        ------
        :class:`ValueError`
            When a mobject tries to add itself.
        :class:`TypeError`
            When trying to add an object that is not an instance of :class:`Mobject`.


        Notes
        -----
        A mobject cannot contain itself, and it cannot contain a submobject
        more than once.  If the parent mobject is displayed, the newly-added
        submobjects will also be displayed (i.e. they are automatically added
        to the parent Scene).

        See Also
        --------
        :meth:`remove`
        :meth:`add_to_back`

        Examples
        --------
        ::

            >>> outer = Mobject()
            >>> inner = Mobject()
            >>> outer = outer.add(inner)

        Duplicates are not added again::

            >>> outer = outer.add(inner)
            >>> len(outer.submobjects)
            1

        Only Mobjects can be added::

            >>> outer.add(3)
            Traceback (most recent call last):
            ...
            TypeError: Only values of type Mobject can be added as submobjects of Mobject, but the value 3 (at index 0) is of type int.

        Adding an object to itself raises an error::

            >>> outer.add(outer)
            Traceback (most recent call last):
            ...
            ValueError: Cannot add Mobject as a submobject of itself (at index 0).

        A given mobject cannot be added as a submobject
        twice to some parent::

            >>> parent = Mobject(name="parent")
            >>> child = Mobject(name="child")
            >>> parent.add(child, child)
            [...] WARNING  ...
            parent
            >>> parent.submobjects
            [child]

        """
        self._assert_valid_submobjects(mobjects)
        unique_mobjects = remove_list_redundancies(mobjects)
        if len(mobjects) != len(unique_mobjects):
            logger.warning(
                "Attempted adding some Mobject as a child more than once, "
                "this is not possible. Repetitions are ignored.",
            )

        self.submobjects = list_update(self.submobjects, unique_mobjects)
        return self

    def insert(self, index: int, mobject: Mobject) -> None:
        """Inserts a mobject at a specific position into self.submobjects

        Effectively just calls  ``self.submobjects.insert(index, mobject)``,
        where ``self.submobjects`` is a list.

        Highly adapted from ``Mobject.add``.

        Parameters
        ----------
        index
            The index at which
        mobject
            The mobject to be inserted.
        """
        self._assert_valid_submobjects([mobject])
        self.submobjects.insert(index, mobject)

    def __add__(self, mobject: Mobject):
        raise NotImplementedError

    def __iadd__(self, mobject: Mobject):
        raise NotImplementedError

    def add_to_back(self, *mobjects: Mobject) -> Self:
        """Add all passed mobjects to the back of the submobjects.

        If :attr:`submobjects` already contains the given mobjects, they just get moved
        to the back instead.

        Parameters
        ----------
        mobjects
            The mobjects to add.

        Returns
        -------
        :class:`Mobject`
            ``self``


        .. note::

            Technically, this is done by adding (or moving) the mobjects to
            the head of :attr:`submobjects`. The head of this list is rendered
            first, which places the corresponding mobjects behind the
            subsequent list members.

        Raises
        ------
        :class:`ValueError`
            When a mobject tries to add itself.
        :class:`TypeError`
            When trying to add an object that is not an instance of :class:`Mobject`.

        Notes
        -----
        A mobject cannot contain itself, and it cannot contain a submobject
        more than once.  If the parent mobject is displayed, the newly-added
        submobjects will also be displayed (i.e. they are automatically added
        to the parent Scene).

        See Also
        --------
        :meth:`remove`
        :meth:`add`

        """
        self._assert_valid_submobjects(mobjects)
        self.remove(*mobjects)
        # dict.fromkeys() removes duplicates while maintaining order
        self.submobjects = list(dict.fromkeys(mobjects)) + self.submobjects
        return self

    def remove(self, *mobjects: Mobject) -> Self:
        """Remove :attr:`submobjects`.

        The mobjects are removed from :attr:`submobjects`, if they exist.

        Subclasses of mobject may implement ``-`` and ``-=`` dunder methods.

        Parameters
        ----------
        mobjects
            The mobjects to remove.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See Also
        --------
        :meth:`add`

        """
        for mobject in mobjects:
            if mobject in self.submobjects:
                self.submobjects.remove(mobject)
        return self

    def __sub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def set(self, **kwargs) -> Self:
        """Sets attributes.

        I.e. ``my_mobject.set(foo=1)`` applies ``my_mobject.foo = 1``.

        This is a convenience to be used along with :attr:`animate` to
        animate setting attributes.

        In addition to this method, there is a compatibility
        layer that allows ``get_*`` and ``set_*`` methods to
        get and set generic attributes. For instance::

            >>> mob = Mobject()
            >>> mob.set_foo(0)
            Mobject
            >>> mob.get_foo()
            0
            >>> mob.foo
            0

        This compatibility layer does not interfere with any
        ``get_*`` or ``set_*`` methods that are explicitly
        defined.

        .. warning::

            This compatibility layer is for backwards compatibility
            and is not guaranteed to stay around. Where applicable,
            please prefer getting/setting attributes normally or with
            the :meth:`set` method.

        Parameters
        ----------
        **kwargs
            The attributes and corresponding values to set.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------
        ::

            >>> mob = Mobject()
            >>> mob.set(foo=0)
            Mobject
            >>> mob.foo
            0
        """
        for attr, value in kwargs.items():
            setattr(self, attr, value)

        return self

    def __getattr__(self, attr: str) -> types.MethodType:
        # Add automatic compatibility layer
        # between properties and get_* and set_*
        # methods.
        #
        # In python 3.9+ we could change this
        # logic to use str.remove_prefix instead.

        if attr.startswith("get_"):
            # Remove the "get_" prefix
            to_get = attr[4:]

            def getter(self):
                warnings.warn(
                    "This method is not guaranteed to stay around. Please prefer "
                    "getting the attribute normally.",
                    DeprecationWarning,
                    stacklevel=2,
                )

                return getattr(self, to_get)

            # Return a bound method
            return types.MethodType(getter, self)

        if attr.startswith("set_"):
            # Remove the "set_" prefix
            to_set = attr[4:]

            def setter(self, value):
                warnings.warn(
                    "This method is not guaranteed to stay around. Please prefer "
                    "setting the attribute normally or with Mobject.set().",
                    DeprecationWarning,
                    stacklevel=2,
                )

                setattr(self, to_set, value)

                return self

            # Return a bound method
            return types.MethodType(setter, self)

        # Unhandled attribute, therefore error
        raise AttributeError(f"{type(self).__name__} object has no attribute '{attr}'")

    @property
    def width(self) -> float:
        """The width of the mobject.

        Returns
        -------
        :class:`float`

        Examples
        --------
        .. manim:: WidthExample

            class WidthExample(Scene):
                def construct(self):
                    decimal = DecimalNumber().to_edge(UP)
                    rect = Rectangle(color=BLUE)
                    rect_copy = rect.copy().set_stroke(GRAY, opacity=0.5)

                    decimal.add_updater(lambda d: d.set_value(rect.width))

                    self.add(rect_copy, rect, decimal)
                    self.play(rect.animate.set(width=7))
                    self.wait()

        See also
        --------
        :meth:`length_over_dim`

        """
        # Get the length across the X dimension
        return self.length_over_dim(0)

    @width.setter
    def width(self, value: float):
        self.scale_to_fit_width(value)

    @property
    def height(self) -> float:
        """The height of the mobject.

        Returns
        -------
        :class:`float`

        Examples
        --------
        .. manim:: HeightExample

            class HeightExample(Scene):
                def construct(self):
                    decimal = DecimalNumber().to_edge(UP)
                    rect = Rectangle(color=BLUE)
                    rect_copy = rect.copy().set_stroke(GRAY, opacity=0.5)

                    decimal.add_updater(lambda d: d.set_value(rect.height))

                    self.add(rect_copy, rect, decimal)
                    self.play(rect.animate.set(height=5))
                    self.wait()

        See also
        --------
        :meth:`length_over_dim`

        """
        # Get the length across the Y dimension
        return self.length_over_dim(1)

    @height.setter
    def height(self, value: float):
        self.scale_to_fit_height(value)

    @property
    def depth(self) -> float:
        """The depth of the mobject.

        Returns
        -------
        :class:`float`

        See also
        --------
        :meth:`length_over_dim`

        """
        # Get the length across the Z dimension
        return self.length_over_dim(2)

    @depth.setter
    def depth(self, value: float):
        self.scale_to_fit_depth(value)

    # Can't be staticmethod because of point_cloud_mobject.py
    def get_array_attrs(self) -> list[Literal["points"]]:
        return ["points"]

    def apply_over_attr_arrays(self, func: MultiMappingFunction) -> Self:
        for attr in self.get_array_attrs():
            setattr(self, attr, func(getattr(self, attr)))
        return self

    # Displaying

    def get_image(self, camera=None) -> PixelArray:
        if camera is None:
            from ..camera.camera import Camera

            camera = Camera()
        camera.capture_mobject(self)
        return camera.get_image()

    def show(self, camera=None) -> None:
        self.get_image(camera=camera).show()

    def save_image(self, name: str | None = None) -> None:
        """Saves an image of only this :class:`Mobject` at its position to a png
        file.
        """
        self.get_image().save(
            Path(config.get_dir("video_dir")).joinpath((name or str(self)) + ".png"),
        )

    def copy(self) -> Self:
        """Create and return an identical copy of the :class:`Mobject` including all
        :attr:`submobjects`.

        Returns
        -------
        :class:`Mobject`
            The copy.

        Note
        ----
        The clone is initially not visible in the Scene, even if the original was.
        """
        return copy.deepcopy(self)

    def generate_target(self, use_deepcopy: bool = False) -> Self:
        self.target = None  # Prevent unbounded linear recursion
        if use_deepcopy:
            self.target = copy.deepcopy(self)
        else:
            self.target = self.copy()
        return self.target

    # Updating

    def update(self, dt: float = 0, recursive: bool = True) -> Self:
        """Apply all updaters.

        Does nothing if updating is suspended.

        Parameters
        ----------
        dt
            The parameter ``dt`` to pass to the update functions. Usually this is the
            time in seconds since the last call of ``update``.
        recursive
            Whether to recursively update all submobjects.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See Also
        --------
        :meth:`add_updater`
        :meth:`get_updaters`

        """
        if self.updating_suspended:
            return self
        for updater in self.updaters:
            if "dt" in inspect.signature(updater).parameters:
                updater(self, dt)
            else:
                updater(self)
        if recursive:
            for submob in self.submobjects:
                submob.update(dt, recursive)
        return self

    def get_time_based_updaters(self) -> list[TimeBasedUpdater]:
        """Return all updaters using the ``dt`` parameter.

        The updaters use this parameter as the input for difference in time.

        Returns
        -------
        List[:class:`Callable`]
            The list of time based updaters.

        See Also
        --------
        :meth:`get_updaters`
        :meth:`has_time_based_updater`

        """
        return [
            updater
            for updater in self.updaters
            if "dt" in inspect.signature(updater).parameters
        ]

    def has_time_based_updater(self) -> bool:
        """Test if ``self`` has a time based updater.

        Returns
        -------
        :class:`bool`
            ``True`` if at least one updater uses the ``dt`` parameter, ``False``
            otherwise.

        See Also
        --------
        :meth:`get_time_based_updaters`

        """
        return any(
            "dt" in inspect.signature(updater).parameters for updater in self.updaters
        )

    def get_updaters(self) -> list[Updater]:
        """Return all updaters.

        Returns
        -------
        List[:class:`Callable`]
            The list of updaters.

        See Also
        --------
        :meth:`add_updater`
        :meth:`get_time_based_updaters`

        """
        return self.updaters

    def get_family_updaters(self) -> list[Updater]:
        return list(it.chain(*(sm.get_updaters() for sm in self.get_family())))

    def add_updater(
        self,
        update_function: Updater,
        index: int | None = None,
        call_updater: bool = False,
    ) -> Self:
        """Add an update function to this mobject.

        Update functions, or updaters in short, are functions that are applied to the
        Mobject in every frame.

        Parameters
        ----------
        update_function
            The update function to be added.
            Whenever :meth:`update` is called, this update function gets called using
            ``self`` as the first parameter.
            The updater can have a second parameter ``dt``. If it uses this parameter,
            it gets called using a second value ``dt``, usually representing the time
            in seconds since the last call of :meth:`update`.
        index
            The index at which the new updater should be added in ``self.updaters``.
            In case ``index`` is ``None`` the updater will be added at the end.
        call_updater
            Whether or not to call the updater initially. If ``True``, the updater will
            be called using ``dt=0``.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------
        .. manim:: NextToUpdater

            class NextToUpdater(Scene):
                def construct(self):
                    def dot_position(mobject):
                        mobject.set_value(dot.get_center()[0])
                        mobject.next_to(dot)

                    dot = Dot(RIGHT*3)
                    label = DecimalNumber()
                    label.add_updater(dot_position)
                    self.add(dot, label)

                    self.play(Rotating(dot, about_point=ORIGIN, angle=TAU, run_time=TAU, rate_func=linear))

        .. manim:: DtUpdater

            class DtUpdater(Scene):
                def construct(self):
                    square = Square()

                    #Let the square rotate 90° per second
                    square.add_updater(lambda mobject, dt: mobject.rotate(dt*90*DEGREES))
                    self.add(square)
                    self.wait(2)

        See also
        --------
        :meth:`get_updaters`
        :meth:`remove_updater`
        :class:`~.UpdateFromFunc`
        """
        if index is None:
            self.updaters.append(update_function)
        else:
            self.updaters.insert(index, update_function)
        if call_updater:
            parameters = inspect.signature(update_function).parameters
            if "dt" in parameters:
                update_function(self, 0)
            else:
                update_function(self)
        return self

    def remove_updater(self, update_function: Updater) -> Self:
        """Remove an updater.

        If the same updater is applied multiple times, every instance gets removed.

        Parameters
        ----------
        update_function
            The update function to be removed.


        Returns
        -------
        :class:`Mobject`
            ``self``

        See also
        --------
        :meth:`clear_updaters`
        :meth:`add_updater`
        :meth:`get_updaters`

        """
        while update_function in self.updaters:
            self.updaters.remove(update_function)
        return self

    def clear_updaters(self, recursive: bool = True) -> Self:
        """Remove every updater.

        Parameters
        ----------
        recursive
            Whether to recursively call ``clear_updaters`` on all submobjects.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See also
        --------
        :meth:`remove_updater`
        :meth:`add_updater`
        :meth:`get_updaters`

        """
        self.updaters = []
        if recursive:
            for submob in self.submobjects:
                submob.clear_updaters()
        return self

    def match_updaters(self, mobject: Mobject) -> Self:
        """Match the updaters of the given mobject.

        Parameters
        ----------
        mobject
            The mobject whose updaters get matched.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Note
        ----
        All updaters from submobjects are removed, but only updaters of the given
        mobject are matched, not those of it's submobjects.

        See also
        --------
        :meth:`add_updater`
        :meth:`clear_updaters`

        """
        self.clear_updaters()
        for updater in mobject.get_updaters():
            self.add_updater(updater)
        return self

    def suspend_updating(self, recursive: bool = True) -> Self:
        """Disable updating from updaters and animations.


        Parameters
        ----------
        recursive
            Whether to recursively suspend updating on all submobjects.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See also
        --------
        :meth:`resume_updating`
        :meth:`add_updater`

        """
        self.updating_suspended = True
        if recursive:
            for submob in self.submobjects:
                submob.suspend_updating(recursive)
        return self

    def resume_updating(self, recursive: bool = True) -> Self:
        """Enable updating from updaters and animations.

        Parameters
        ----------
        recursive
            Whether to recursively enable updating on all submobjects.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See also
        --------
        :meth:`suspend_updating`
        :meth:`add_updater`

        """
        self.updating_suspended = False
        if recursive:
            for submob in self.submobjects:
                submob.resume_updating(recursive)
        self.update(dt=0, recursive=recursive)
        return self

    # Transforming operations

    def apply_to_family(self, func: Callable[[Mobject], None]) -> None:
        """Apply a function to ``self`` and every submobject with points recursively.

        Parameters
        ----------
        func
            The function to apply to each mobject. ``func`` gets passed the respective
            (sub)mobject as parameter.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See also
        --------
        :meth:`family_members_with_points`

        """
        for mob in self.family_members_with_points():
            func(mob)

    def shift(self, *vectors: Vector3D) -> Self:
        """Shift by the given vectors.

        Parameters
        ----------
        vectors
            Vectors to shift by. If multiple vectors are given, they are added
            together.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See also
        --------
        :meth:`move_to`
        """
        total_vector = reduce(op.add, vectors)
        for mob in self.family_members_with_points():
            mob.points = mob.points.astype("float")
            mob.points += total_vector

        return self

    def scale(self, scale_factor: float, **kwargs) -> Self:
        r"""Scale the size by a factor.

        Default behavior is to scale about the center of the mobject.

        Parameters
        ----------
        scale_factor
            The scaling factor :math:`\alpha`. If :math:`0 < |\alpha| < 1`, the mobject
            will shrink, and for :math:`|\alpha| > 1` it will grow. Furthermore,
            if :math:`\alpha < 0`, the mobject is also flipped.
        kwargs
            Additional keyword arguments passed to
            :meth:`apply_points_function_about_point`.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------

        .. manim:: MobjectScaleExample
            :save_last_frame:

            class MobjectScaleExample(Scene):
                def construct(self):
                    f1 = Text("F")
                    f2 = Text("F").scale(2)
                    f3 = Text("F").scale(0.5)
                    f4 = Text("F").scale(-1)

                    vgroup = VGroup(f1, f2, f3, f4).arrange(6 * RIGHT)
                    self.add(vgroup)

        See also
        --------
        :meth:`move_to`

        """
        self.apply_points_function_about_point(
            lambda points: scale_factor * points, **kwargs
        )
        return self

    def rotate_about_origin(self, angle: float, axis: Vector3D = OUT, axes=[]) -> Self:
        """Rotates the :class:`~.Mobject` about the ORIGIN, which is at [0,0,0]."""
        return self.rotate(angle, axis, about_point=ORIGIN)

    def rotate(
        self,
        angle: float,
        axis: Vector3D = OUT,
        about_point: Point3DLike | None = None,
        **kwargs,
    ) -> Self:
        """Rotates the :class:`~.Mobject` about a certain point."""
        rot_matrix = rotation_matrix(angle, axis)
        self.apply_points_function_about_point(
            lambda points: np.dot(points, rot_matrix.T), about_point, **kwargs
        )
        return self

    def flip(self, axis: Vector3D = UP, **kwargs) -> Self:
        """Flips/Mirrors an mobject about its center.

        Examples
        --------

        .. manim:: FlipExample
            :save_last_frame:

            class FlipExample(Scene):
                def construct(self):
                    s= Line(LEFT, RIGHT+UP).shift(4*LEFT)
                    self.add(s)
                    s2= s.copy().flip()
                    self.add(s2)

        """
        return self.rotate(TAU / 2, axis, **kwargs)

    def stretch(self, factor: float, dim: int, **kwargs) -> Self:
        def func(points: Point3D_Array) -> Point3D_Array:
            points[:, dim] *= factor
            return points

        self.apply_points_function_about_point(func, **kwargs)
        return self

    def apply_function(self, function: MappingFunction, **kwargs) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if len(kwargs) == 0:
            kwargs["about_point"] = ORIGIN

        def multi_mapping_function(points: Point3D_Array) -> Point3D_Array:
            result: Point3D_Array = np.apply_along_axis(function, 1, points)
            return result

        self.apply_points_function_about_point(multi_mapping_function, **kwargs)
        return self

    def apply_function_to_position(self, function: MappingFunction) -> Self:
        self.move_to(function(self.get_center()))
        return self

    def apply_function_to_submobject_positions(self, function: MappingFunction) -> Self:
        for submob in self.submobjects:
            submob.apply_function_to_position(function)
        return self

    def apply_matrix(self, matrix, **kwargs) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if ("about_point" not in kwargs) and ("about_edge" not in kwargs):
            kwargs["about_point"] = ORIGIN
        full_matrix = np.identity(self.dim)
        matrix = np.array(matrix)
        full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        self.apply_points_function_about_point(
            lambda points: np.dot(points, full_matrix.T), **kwargs
        )
        return self

    def apply_complex_function(
        self, function: Callable[[complex], complex], **kwargs
    ) -> Self:
        """Applies a complex function to a :class:`Mobject`.
        The x and y Point3Ds correspond to the real and imaginary parts respectively.

        Example
        -------

        .. manim:: ApplyFuncExample

            class ApplyFuncExample(Scene):
                def construct(self):
                    circ = Circle().scale(1.5)
                    circ_ref = circ.copy()
                    circ.apply_complex_function(
                        lambda x: np.exp(x*1j)
                    )
                    t = ValueTracker(0)
                    circ.add_updater(
                        lambda x: x.become(circ_ref.copy().apply_complex_function(
                            lambda x: np.exp(x+t.get_value()*1j)
                        )).set_color(BLUE)
                    )
                    self.add(circ_ref)
                    self.play(TransformFromCopy(circ_ref, circ))
                    self.play(t.animate.set_value(TAU), run_time=3)
        """

        def R3_func(point):
            x, y, z = point
            xy_complex = function(complex(x, y))
            return [xy_complex.real, xy_complex.imag, z]

        return self.apply_function(R3_func)

    def reverse_points(self) -> Self:
        for mob in self.family_members_with_points():
            mob.apply_over_attr_arrays(lambda arr: np.array(list(reversed(arr))))
        return self

    def repeat(self, count: int) -> Self:
        """This can make transition animations nicer"""

        def repeat_array(array):
            return reduce(lambda a1, a2: np.append(a1, a2, axis=0), [array] * count)

        for mob in self.family_members_with_points():
            mob.apply_over_attr_arrays(repeat_array)
        return self

    # In place operations.
    # Note, much of these are now redundant with default behavior of
    # above methods

    # TODO: name is inconsistent with OpenGLMobject.apply_points_function()
    def apply_points_function_about_point(
        self,
        func: MultiMappingFunction,
        about_point: Point3DLike | None = None,
        about_edge: Vector3D | None = None,
    ) -> Self:
        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
            about_point = self.get_critical_point(about_edge)
        for mob in self.family_members_with_points():
            mob.points -= about_point
            mob.points = func(mob.points)
            mob.points += about_point
        return self

    def pose_at_angle(self, **kwargs):
        self.rotate(TAU / 14, RIGHT + UP, **kwargs)
        return self

    # Positioning methods

    def center(self) -> Self:
        """Moves the center of the mobject to the center of the scene.

        Returns
        -------
        :class:`.Mobject`
            The centered mobject.
        """
        self.shift(-self.get_center())
        return self

    def align_on_border(
        self, direction: Vector3D, buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER
    ) -> Self:
        """Direction just needs to be a vector pointing towards side or
        corner in the 2d plane.
        """
        target_point = np.sign(direction) * (
            config["frame_x_radius"],
            config["frame_y_radius"],
            0,
        )
        point_to_align = self.get_critical_point(direction)
        shift_val = target_point - point_to_align - buff * np.array(direction)
        shift_val = shift_val * abs(np.sign(direction))
        self.shift(shift_val)
        return self

    def to_corner(
        self, corner: Vector3D = DL, buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER
    ) -> Self:
        """Moves this :class:`~.Mobject` to the given corner of the screen.

        Returns
        -------
        :class:`.Mobject`
            The newly positioned mobject.

        Examples
        --------

        .. manim:: ToCornerExample
            :save_last_frame:

            class ToCornerExample(Scene):
                def construct(self):
                    c = Circle()
                    c.to_corner(UR)
                    t = Tex("To the corner!")
                    t2 = MathTex("x^3").shift(DOWN)
                    self.add(c,t,t2)
                    t.to_corner(DL, buff=0)
                    t2.to_corner(UL, buff=1.5)
        """
        return self.align_on_border(corner, buff)

    def to_edge(
        self, edge: Vector3D = LEFT, buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER
    ) -> Self:
        """Moves this :class:`~.Mobject` to the given edge of the screen,
        without affecting its position in the other dimension.

        Returns
        -------
        :class:`.Mobject`
            The newly positioned mobject.

        Examples
        --------

        .. manim:: ToEdgeExample
            :save_last_frame:

            class ToEdgeExample(Scene):
                def construct(self):
                    tex_top = Tex("I am at the top!")
                    tex_top.to_edge(UP)
                    tex_side = Tex("I am moving to the side!")
                    c = Circle().shift(2*DOWN)
                    self.add(tex_top, tex_side, c)
                    tex_side.to_edge(LEFT)
                    c.to_edge(RIGHT, buff=0)

        """
        return self.align_on_border(edge, buff)

    def next_to(
        self,
        mobject_or_point: Mobject | Point3DLike,
        direction: Vector3D = RIGHT,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        aligned_edge: Vector3D = ORIGIN,
        submobject_to_align: Mobject | None = None,
        index_of_submobject_to_align: int | None = None,
        coor_mask: Vector3D = np.array([1, 1, 1]),
    ) -> Self:
        """Move this :class:`~.Mobject` next to another's :class:`~.Mobject` or Point3D.

        Examples
        --------

        .. manim:: GeometricShapes
            :save_last_frame:

            class GeometricShapes(Scene):
                def construct(self):
                    d = Dot()
                    c = Circle()
                    s = Square()
                    t = Triangle()
                    d.next_to(c, RIGHT)
                    s.next_to(c, LEFT)
                    t.next_to(c, DOWN)
                    self.add(d, c, s, t)

        """
        if isinstance(mobject_or_point, Mobject):
            mob = mobject_or_point
            if index_of_submobject_to_align is not None:
                target_aligner = mob[index_of_submobject_to_align]
            else:
                target_aligner = mob
            target_point = target_aligner.get_critical_point(aligned_edge + direction)
        else:
            target_point = mobject_or_point
        if submobject_to_align is not None:
            aligner = submobject_to_align
        elif index_of_submobject_to_align is not None:
            aligner = self[index_of_submobject_to_align]
        else:
            aligner = self
        point_to_align = aligner.get_critical_point(aligned_edge - direction)
        self.shift((target_point - point_to_align + buff * direction) * coor_mask)
        return self

    def shift_onto_screen(self, **kwargs) -> Self:
        space_lengths = [config["frame_x_radius"], config["frame_y_radius"]]
        for vect in UP, DOWN, LEFT, RIGHT:
            dim = np.argmax(np.abs(vect))
            buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_EDGE_BUFFER)
            max_val = space_lengths[dim] - buff
            edge_center = self.get_edge_center(vect)
            if np.dot(edge_center, vect) > max_val:
                self.to_edge(vect, **kwargs)
        return self

    def is_off_screen(self):
        if self.get_left()[0] > config["frame_x_radius"]:
            return True
        if self.get_right()[0] < -config["frame_x_radius"]:
            return True
        if self.get_bottom()[1] > config["frame_y_radius"]:
            return True
        return self.get_top()[1] < -config["frame_y_radius"]

    def stretch_about_point(self, factor: float, dim: int, point: Point3DLike) -> Self:
        return self.stretch(factor, dim, about_point=point)

    def rescale_to_fit(
        self, length: float, dim: int, stretch: bool = False, **kwargs
    ) -> Self:
        old_length = self.length_over_dim(dim)
        if old_length == 0:
            return self
        if stretch:
            self.stretch(length / old_length, dim, **kwargs)
        else:
            self.scale(length / old_length, **kwargs)
        return self

    def scale_to_fit_width(self, width: float, **kwargs) -> Self:
        """Scales the :class:`~.Mobject` to fit a width while keeping height/depth proportional.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.height
            np.float64(2.0)
            >>> sq.scale_to_fit_width(5)
            Square
            >>> sq.width
            np.float64(5.0)
            >>> sq.height
            np.float64(5.0)
        """
        return self.rescale_to_fit(width, 0, stretch=False, **kwargs)

    def stretch_to_fit_width(self, width: float, **kwargs) -> Self:
        """Stretches the :class:`~.Mobject` to fit a width, not keeping height/depth proportional.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.height
            np.float64(2.0)
            >>> sq.stretch_to_fit_width(5)
            Square
            >>> sq.width
            np.float64(5.0)
            >>> sq.height
            np.float64(2.0)
        """
        return self.rescale_to_fit(width, 0, stretch=True, **kwargs)

    def scale_to_fit_height(self, height: float, **kwargs) -> Self:
        """Scales the :class:`~.Mobject` to fit a height while keeping width/depth proportional.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.width
            np.float64(2.0)
            >>> sq.scale_to_fit_height(5)
            Square
            >>> sq.height
            np.float64(5.0)
            >>> sq.width
            np.float64(5.0)
        """
        return self.rescale_to_fit(height, 1, stretch=False, **kwargs)

    def stretch_to_fit_height(self, height: float, **kwargs) -> Self:
        """Stretches the :class:`~.Mobject` to fit a height, not keeping width/depth proportional.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.width
            np.float64(2.0)
            >>> sq.stretch_to_fit_height(5)
            Square
            >>> sq.height
            np.float64(5.0)
            >>> sq.width
            np.float64(2.0)
        """
        return self.rescale_to_fit(height, 1, stretch=True, **kwargs)

    def scale_to_fit_depth(self, depth: float, **kwargs) -> Self:
        """Scales the :class:`~.Mobject` to fit a depth while keeping width/height proportional."""
        return self.rescale_to_fit(depth, 2, stretch=False, **kwargs)

    def stretch_to_fit_depth(self, depth: float, **kwargs) -> Self:
        """Stretches the :class:`~.Mobject` to fit a depth, not keeping width/height proportional."""
        return self.rescale_to_fit(depth, 2, stretch=True, **kwargs)

    def set_coord(self, value, dim: int, direction: Vector3D = ORIGIN) -> Self:
        curr = self.get_coord(dim, direction)
        shift_vect = np.zeros(self.dim)
        shift_vect[dim] = value - curr
        self.shift(shift_vect)
        return self

    def set_x(self, x: float, direction: Vector3D = ORIGIN) -> Self:
        """Set x value of the center of the :class:`~.Mobject` (``int`` or ``float``)"""
        return self.set_coord(x, 0, direction)

    def set_y(self, y: float, direction: Vector3D = ORIGIN) -> Self:
        """Set y value of the center of the :class:`~.Mobject` (``int`` or ``float``)"""
        return self.set_coord(y, 1, direction)

    def set_z(self, z: float, direction: Vector3D = ORIGIN) -> Self:
        """Set z value of the center of the :class:`~.Mobject` (``int`` or ``float``)"""
        return self.set_coord(z, 2, direction)

    def space_out_submobjects(self, factor: float = 1.5, **kwargs) -> Self:
        self.scale(factor, **kwargs)
        for submob in self.submobjects:
            submob.scale(1.0 / factor)
        return self

    def move_to(
        self,
        point_or_mobject: Point3DLike | Mobject,
        aligned_edge: Vector3D = ORIGIN,
        coor_mask: Vector3D = np.array([1, 1, 1]),
    ) -> Self:
        """Move center of the :class:`~.Mobject` to certain Point3D."""
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_critical_point(aligned_edge)
        else:
            target = point_or_mobject
        point_to_align = self.get_critical_point(aligned_edge)
        self.shift((target - point_to_align) * coor_mask)
        return self

    def replace(
        self, mobject: Mobject, dim_to_match: int = 0, stretch: bool = False
    ) -> Self:
        if not mobject.get_num_points() and not mobject.submobjects:
            raise Warning("Attempting to replace mobject with no points")
        if stretch:
            self.stretch_to_fit_width(mobject.width)
            self.stretch_to_fit_height(mobject.height)
        else:
            self.rescale_to_fit(
                mobject.length_over_dim(dim_to_match),
                dim_to_match,
                stretch=False,
            )
        self.shift(mobject.get_center() - self.get_center())
        return self

    def surround(
        self,
        mobject: Mobject,
        dim_to_match: int = 0,
        stretch: bool = False,
        buff: float = MED_SMALL_BUFF,
    ) -> Self:
        self.replace(mobject, dim_to_match, stretch)
        length = mobject.length_over_dim(dim_to_match)
        self.scale((length + buff) / length)
        return self

    def put_start_and_end_on(self, start: Point3DLike, end: Point3DLike) -> Self:
        curr_start, curr_end = self.get_start_and_end()
        curr_vect = curr_end - curr_start
        if np.all(curr_vect == 0):
            # TODO: this looks broken. It makes self.points a Point3D instead
            # of a Point3D_Array. However, modifying this breaks some tests
            # where this is currently expected.
            self.points = np.array(start)
            return self
        target_vect = np.asarray(end) - np.asarray(start)
        axis = (
            normalize(np.cross(curr_vect, target_vect))
            if np.linalg.norm(np.cross(curr_vect, target_vect)) != 0
            else OUT
        )
        self.scale(
            np.linalg.norm(target_vect) / np.linalg.norm(curr_vect),
            about_point=curr_start,
        )
        self.rotate(
            angle_between_vectors(curr_vect, target_vect),
            about_point=curr_start,
            axis=axis,
        )
        self.shift(start - curr_start)
        return self

    # Background rectangle
    def add_background_rectangle(
        self, color: ParsableManimColor | None = None, opacity: float = 0.75, **kwargs
    ) -> Self:
        """Add a BackgroundRectangle as submobject.

        The BackgroundRectangle is added behind other submobjects.

        This can be used to increase the mobjects visibility in front of a noisy background.

        Parameters
        ----------
        color
            The color of the BackgroundRectangle
        opacity
            The opacity of the BackgroundRectangle
        kwargs
            Additional keyword arguments passed to the BackgroundRectangle constructor


        Returns
        -------
        :class:`Mobject`
            ``self``

        See Also
        --------
        :meth:`add_to_back`
        :class:`~.BackgroundRectangle`

        """
        # TODO, this does not behave well when the mobject has points,
        # since it gets displayed on top
        from manim.mobject.geometry.shape_matchers import BackgroundRectangle

        self.background_rectangle = BackgroundRectangle(
            self, color=color, fill_opacity=opacity, **kwargs
        )
        self.add_to_back(self.background_rectangle)
        return self

    def add_background_rectangle_to_submobjects(self, **kwargs) -> Self:
        for submobject in self.submobjects:
            submobject.add_background_rectangle(**kwargs)
        return self

    def add_background_rectangle_to_family_members_with_points(self, **kwargs) -> Self:
        for mob in self.family_members_with_points():
            mob.add_background_rectangle(**kwargs)
        return self

    # Color functions

    def set_color(
        self, color: ParsableManimColor = YELLOW_C, family: bool = True
    ) -> Self:
        """Condition is function which takes in one arguments, (x, y, z).
        Here it just recurses to submobjects, but in subclasses this
        should be further implemented based on the the inner workings
        of color
        """
        if family:
            for submob in self.submobjects:
                submob.set_color(color, family=family)

        self.color = ManimColor.parse(color)
        return self

    def set_color_by_gradient(self, *colors: ParsableManimColor) -> Self:
        """
        Parameters
        ----------
        colors
            The colors to use for the gradient. Use like `set_color_by_gradient(RED, BLUE, GREEN)`.

        self.color = ManimColor.parse(color)
        return self
        """
        self.set_submobject_colors_by_gradient(*colors)
        return self

    def set_colors_by_radial_gradient(
        self,
        center: Point3DLike | None = None,
        radius: float = 1,
        inner_color: ParsableManimColor = WHITE,
        outer_color: ParsableManimColor = BLACK,
    ) -> Self:
        self.set_submobject_colors_by_radial_gradient(
            center,
            radius,
            inner_color,
            outer_color,
        )
        return self

    def set_submobject_colors_by_gradient(self, *colors: Iterable[ParsableManimColor]):
        if len(colors) == 0:
            raise ValueError("Need at least one color")
        elif len(colors) == 1:
            return self.set_color(*colors)

        mobs = self.family_members_with_points()
        new_colors = color_gradient(colors, len(mobs))

        for mob, color in zip(mobs, new_colors):
            mob.set_color(color, family=False)
        return self

    def set_submobject_colors_by_radial_gradient(
        self,
        center: Point3DLike | None = None,
        radius: float = 1,
        inner_color: ParsableManimColor = WHITE,
        outer_color: ParsableManimColor = BLACK,
    ) -> Self:
        if center is None:
            center = self.get_center()

        for mob in self.family_members_with_points():
            t = np.linalg.norm(mob.get_center() - center) / radius
            t = min(t, 1)
            mob_color = interpolate_color(inner_color, outer_color, t)
            mob.set_color(mob_color, family=False)

        return self

    def to_original_color(self) -> Self:
        self.set_color(self.color)
        return self

    def fade_to(
        self, color: ParsableManimColor, alpha: float, family: bool = True
    ) -> Self:
        if self.get_num_points() > 0:
            new_color = interpolate_color(self.get_color(), color, alpha)
            self.set_color(new_color, family=False)
        if family:
            for submob in self.submobjects:
                submob.fade_to(color, alpha)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True) -> Self:
        if family:
            for submob in self.submobjects:
                submob.fade(darkness, family)
        return self

    def get_color(self) -> ManimColor:
        """Returns the color of the :class:`~.Mobject`

        Examples
        --------
        ::

            >>> from manim import Square, RED
            >>> Square(color=RED).get_color() == RED
            True

        """
        return self.color

    ##

    def save_state(self) -> Self:
        """Save the current state (position, color & size). Can be restored with :meth:`~.Mobject.restore`."""
        if hasattr(self, "saved_state"):
            # Prevent exponential growth of data
            self.saved_state = None
        self.saved_state = self.copy()

        return self

    def restore(self) -> Self:
        """Restores the state that was previously saved with :meth:`~.Mobject.save_state`."""
        if not hasattr(self, "saved_state") or self.save_state is None:
            raise Exception("Trying to restore without having saved")
        self.become(self.saved_state)
        return self

    def reduce_across_dimension(self, reduce_func: Callable, dim: int):
        """Find the min or max value from a dimension across all points in this and submobjects."""
        assert dim >= 0
        assert dim <= 2
        if len(self.submobjects) == 0 and len(self.points) == 0:
            # If we have no points and no submobjects, return 0 (e.g. center)
            return 0

        # If we do not have points (but do have submobjects)
        # use only the points from those.
        if len(self.points) == 0:  # noqa: SIM108
            rv = None
        else:
            # Otherwise, be sure to include our own points
            rv = reduce_func(self.points[:, dim])
        # Recursively ask submobjects (if any) for the biggest/
        # smallest dimension they have and compare it to the return value.
        for mobj in self.submobjects:
            value = mobj.reduce_across_dimension(reduce_func, dim)
            rv = value if rv is None else reduce_func([value, rv])
        return rv

    def nonempty_submobjects(self) -> list[Self]:
        return [
            submob
            for submob in self.submobjects
            if len(submob.submobjects) != 0 or len(submob.points) != 0
        ]

    def get_merged_array(self, array_attr: str) -> np.ndarray:
        """Return all of a given attribute from this mobject and all submobjects.

        May contain duplicates; the order is in a depth-first (pre-order)
        traversal of the submobjects.
        """
        result = getattr(self, array_attr)
        for submob in self.submobjects:
            result = np.append(result, submob.get_merged_array(array_attr), axis=0)
        return result

    def get_all_points(self) -> Point3D_Array:
        """Return all points from this mobject and all submobjects.

        May contain duplicates; the order is in a depth-first (pre-order)
        traversal of the submobjects.
        """
        return self.get_merged_array("points")

    # Getters

    def get_points_defining_boundary(self) -> Point3D_Array:
        return self.get_all_points()

    def get_num_points(self) -> int:
        return len(self.points)

    def get_extremum_along_dim(
        self, points: Point3DLike_Array | None = None, dim: int = 0, key: int = 0
    ) -> float:
        np_points: Point3D_Array = (
            self.get_points_defining_boundary()
            if points is None
            else np.asarray(points)
        )
        values = np_points[:, dim]
        if key < 0:
            return np.min(values)
        elif key == 0:
            return (np.min(values) + np.max(values)) / 2
        else:
            return np.max(values)

    def get_critical_point(self, direction: Vector3D) -> Point3D:
        """Picture a box bounding the :class:`~.Mobject`.  Such a box has
        9 'critical points': 4 corners, 4 edge center, the
        center. This returns one of them, along the given direction.

        ::

            sample = Arc(start_angle=PI / 7, angle=PI / 5)

            # These are all equivalent
            max_y_1 = sample.get_top()[1]
            max_y_2 = sample.get_critical_point(UP)[1]
            max_y_3 = sample.get_extremum_along_dim(dim=1, key=1)

        """
        result = np.zeros(self.dim)
        all_points = self.get_points_defining_boundary()
        if len(all_points) == 0:
            return result
        for dim in range(self.dim):
            result[dim] = self.get_extremum_along_dim(
                all_points,
                dim=dim,
                key=direction[dim],
            )
        return result

    # Pseudonyms for more general get_critical_point method

    def get_edge_center(self, direction: Vector3D) -> Point3D:
        """Get edge Point3Ds for certain direction."""
        return self.get_critical_point(direction)

    def get_corner(self, direction: Vector3D) -> Point3D:
        """Get corner Point3Ds for certain direction."""
        return self.get_critical_point(direction)

    def get_center(self) -> Point3D:
        """Get center Point3Ds"""
        return self.get_critical_point(np.zeros(self.dim))

    def get_center_of_mass(self) -> Point3D:
        return np.apply_along_axis(np.mean, 0, self.get_all_points())

    def get_boundary_point(self, direction: Vector3D) -> Point3D:
        all_points = self.get_points_defining_boundary()
        index = np.argmax(np.dot(all_points, np.array(direction).T))
        return all_points[index]

    def get_midpoint(self) -> Point3D:
        """Get Point3Ds of the middle of the path that forms the  :class:`~.Mobject`.

        Examples
        --------

        .. manim:: AngleMidPoint
            :save_last_frame:

            class AngleMidPoint(Scene):
                def construct(self):
                    line1 = Line(ORIGIN, 2*RIGHT)
                    line2 = Line(ORIGIN, 2*RIGHT).rotate_about_origin(80*DEGREES)

                    a = Angle(line1, line2, radius=1.5, other_angle=False)
                    d = Dot(a.get_midpoint()).set_color(RED)

                    self.add(line1, line2, a, d)
                    self.wait()

        """
        return self.point_from_proportion(0.5)

    def get_top(self) -> Point3D:
        """Get top Point3Ds of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(UP)

    def get_bottom(self) -> Point3D:
        """Get bottom Point3Ds of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(DOWN)

    def get_right(self) -> Point3D:
        """Get right Point3Ds of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(RIGHT)

    def get_left(self) -> Point3D:
        """Get left Point3Ds of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(LEFT)

    def get_zenith(self) -> Point3D:
        """Get zenith Point3Ds of a box bounding a 3D :class:`~.Mobject`."""
        return self.get_edge_center(OUT)

    def get_nadir(self) -> Point3D:
        """Get nadir (opposite the zenith) Point3Ds of a box bounding a 3D :class:`~.Mobject`."""
        return self.get_edge_center(IN)

    def length_over_dim(self, dim: int) -> float:
        """Measure the length of an :class:`~.Mobject` in a certain direction."""
        return self.reduce_across_dimension(
            max,
            dim,
        ) - self.reduce_across_dimension(min, dim)

    def get_coord(self, dim: int, direction: Vector3D = ORIGIN):
        """Meant to generalize ``get_x``, ``get_y`` and ``get_z``"""
        return self.get_extremum_along_dim(dim=dim, key=direction[dim])

    def get_x(self, direction: Vector3D = ORIGIN) -> float:
        """Returns x Point3D of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(0, direction)

    def get_y(self, direction: Vector3D = ORIGIN) -> float:
        """Returns y Point3D of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(1, direction)

    def get_z(self, direction: Vector3D = ORIGIN) -> float:
        """Returns z Point3D of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(2, direction)

    def get_start(self) -> Point3D:
        """Returns the point, where the stroke that surrounds the :class:`~.Mobject` starts."""
        self.throw_error_if_no_points()
        return np.array(self.points[0])

    def get_end(self) -> Point3D:
        """Returns the point, where the stroke that surrounds the :class:`~.Mobject` ends."""
        self.throw_error_if_no_points()
        return np.array(self.points[-1])

    def get_start_and_end(self) -> tuple[Point3D, Point3D]:
        """Returns starting and ending point of a stroke as a ``tuple``."""
        return self.get_start(), self.get_end()

    def point_from_proportion(self, alpha: float) -> Point3D:
        raise NotImplementedError("Please override in a child class.")

    def proportion_from_point(self, point: Point3DLike) -> float:
        raise NotImplementedError("Please override in a child class.")

    def get_pieces(self, n_pieces: float) -> Group:
        template = self.copy()
        template.submobjects = []
        alphas = np.linspace(0, 1, n_pieces + 1)
        return Group(
            *(
                template.copy().pointwise_become_partial(self, a1, a2)
                for a1, a2 in zip(alphas[:-1], alphas[1:])
            )
        )

    def get_z_index_reference_point(self) -> Point3D:
        # TODO, better place to define default z_index_group?
        z_index_group = getattr(self, "z_index_group", self)
        return z_index_group.get_center()

    def has_points(self) -> bool:
        """Check if :class:`~.Mobject` contains points."""
        return len(self.points) > 0

    def has_no_points(self) -> bool:
        """Check if :class:`~.Mobject` *does not* contains points."""
        return not self.has_points()

    # Match other mobject properties

    def match_color(self, mobject: Mobject) -> Self:
        """Match the color with the color of another :class:`~.Mobject`."""
        return self.set_color(mobject.get_color())

    def match_dim_size(self, mobject: Mobject, dim: int, **kwargs) -> Self:
        """Match the specified dimension with the dimension of another :class:`~.Mobject`."""
        return self.rescale_to_fit(mobject.length_over_dim(dim), dim, **kwargs)

    def match_width(self, mobject: Mobject, **kwargs) -> Self:
        """Match the width with the width of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 0, **kwargs)

    def match_height(self, mobject: Mobject, **kwargs) -> Self:
        """Match the height with the height of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 1, **kwargs)

    def match_depth(self, mobject: Mobject, **kwargs) -> Self:
        """Match the depth with the depth of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 2, **kwargs)

    def match_coord(
        self, mobject: Mobject, dim: int, direction: Vector3D = ORIGIN
    ) -> Self:
        """Match the Point3Ds with the Point3Ds of another :class:`~.Mobject`."""
        return self.set_coord(
            mobject.get_coord(dim, direction),
            dim=dim,
            direction=direction,
        )

    def match_x(self, mobject: Mobject, direction=ORIGIN) -> Self:
        """Match x coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 0, direction)

    def match_y(self, mobject: Mobject, direction=ORIGIN) -> Self:
        """Match y coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 1, direction)

    def match_z(self, mobject: Mobject, direction=ORIGIN) -> Self:
        """Match z coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 2, direction)

    def align_to(
        self,
        mobject_or_point: Mobject | Point3DLike,
        direction: Vector3D = ORIGIN,
    ) -> Self:
        """Aligns mobject to another :class:`~.Mobject` in a certain direction.

        Examples:
        mob1.align_to(mob2, UP) moves mob1 vertically so that its
        top edge lines ups with mob2's top edge.
        """
        if isinstance(mobject_or_point, Mobject):
            point = mobject_or_point.get_critical_point(direction)
        else:
            point = mobject_or_point

        for dim in range(self.dim):
            if direction[dim] != 0:
                self.set_coord(point[dim], dim, direction)
        return self

    # Family matters

    def __getitem__(self, value):
        self_list = self.split()
        if isinstance(value, slice):
            GroupClass = self.get_group_class()
            return GroupClass(*self_list.__getitem__(value))
        return self_list.__getitem__(value)

    def __iter__(self):
        return iter(self.split())

    def __len__(self):
        return len(self.split())

    def get_group_class(self) -> type[Group]:
        return Group

    @staticmethod
    def get_mobject_type_class() -> type[Mobject]:
        """Return the base class of this mobject type."""
        return Mobject

    def split(self) -> list[Self]:
        result = [self] if len(self.points) > 0 else []
        return result + self.submobjects

    def get_family(self, recurse: bool = True) -> list[Self]:
        sub_families = [x.get_family() for x in self.submobjects]
        all_mobjects = [self] + list(it.chain(*sub_families))
        return remove_list_redundancies(all_mobjects)

    def family_members_with_points(self) -> list[Self]:
        return [m for m in self.get_family() if m.get_num_points() > 0]

    def arrange(
        self,
        direction: Vector3D = RIGHT,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        center: bool = True,
        **kwargs,
    ) -> Self:
        """Sorts :class:`~.Mobject` next to each other on screen.

        Examples
        --------

        .. manim:: Example
            :save_last_frame:

            class Example(Scene):
                def construct(self):
                    s1 = Square()
                    s2 = Square()
                    s3 = Square()
                    s4 = Square()
                    x = VGroup(s1, s2, s3, s4).set_x(0).arrange(buff=1.0)
                    self.add(x)
        """
        for m1, m2 in zip(self.submobjects, self.submobjects[1:]):
            m2.next_to(m1, direction, buff, **kwargs)
        if center:
            self.center()
        return self

    def arrange_in_grid(
        self,
        rows: int | None = None,
        cols: int | None = None,
        buff: float | tuple[float, float] = MED_SMALL_BUFF,
        cell_alignment: Vector3D = ORIGIN,
        row_alignments: str | None = None,  # "ucd"
        col_alignments: str | None = None,  # "lcr"
        row_heights: Iterable[float | None] | None = None,
        col_widths: Iterable[float | None] | None = None,
        flow_order: str = "rd",
        **kwargs,
    ) -> Self:
        """Arrange submobjects in a grid.

        Parameters
        ----------
        rows
            The number of rows in the grid.
        cols
            The number of columns in the grid.
        buff
            The gap between grid cells. To specify a different buffer in the horizontal and
            vertical directions, a tuple of two values can be given - ``(row, col)``.
        cell_alignment
            The way each submobject is aligned in its grid cell.
        row_alignments
            The vertical alignment for each row (top to bottom). Accepts the following characters: ``"u"`` -
            up, ``"c"`` - center, ``"d"`` - down.
        col_alignments
            The horizontal alignment for each column (left to right). Accepts the following characters ``"l"`` - left,
            ``"c"`` - center, ``"r"`` - right.
        row_heights
            Defines a list of heights for certain rows (top to bottom). If the list contains
            ``None``, the corresponding row will fit its height automatically based
            on the highest element in that row.
        col_widths
            Defines a list of widths for certain columns (left to right). If the list contains ``None``, the
            corresponding column will fit its width automatically based on the widest element in that column.
        flow_order
            The order in which submobjects fill the grid. Can be one of the following values:
            "rd", "dr", "ld", "dl", "ru", "ur", "lu", "ul". ("rd" -> fill rightwards then downwards)

        Returns
        -------
        :class:`Mobject`
            ``self``

        Raises
        ------
        ValueError
            If ``rows`` and ``cols`` are too small to fit all submobjects.
        ValueError
            If :code:`cols`, :code:`col_alignments` and :code:`col_widths` or :code:`rows`,
            :code:`row_alignments` and :code:`row_heights` have mismatching sizes.

        Notes
        -----
        If only one of ``cols`` and ``rows`` is set implicitly, the other one will be chosen big
        enough to fit all submobjects. If neither is set, they will be chosen to be about the same,
        tending towards ``cols`` > ``rows`` (simply because videos are wider than they are high).

        If both ``cell_alignment`` and ``row_alignments`` / ``col_alignments`` are
        defined, the latter has higher priority.

        Examples
        --------
        .. manim:: ExampleBoxes
            :save_last_frame:

            class ExampleBoxes(Scene):
                def construct(self):
                    boxes=VGroup(*[Square() for s in range(0,6)])
                    boxes.arrange_in_grid(rows=2, buff=0.1)
                    self.add(boxes)


        .. manim:: ArrangeInGrid
            :save_last_frame:

            class ArrangeInGrid(Scene):
                def construct(self):
                    boxes = VGroup(*[
                        Rectangle(WHITE, 0.5, 0.5).add(Text(str(i+1)).scale(0.5))
                        for i in range(24)
                    ])
                    self.add(boxes)

                    boxes.arrange_in_grid(
                        buff=(0.25,0.5),
                        col_alignments="lccccr",
                        row_alignments="uccd",
                        col_widths=[1, *[None]*4, 1],
                        row_heights=[1, None, None, 1],
                        flow_order="dr"
                    )


        """
        from manim.mobject.geometry.line import Line

        mobs = self.submobjects.copy()
        start_pos = self.get_center()

        # get cols / rows values if given (implicitly)
        def init_size(num, alignments, sizes):
            if num is not None:
                return num
            if alignments is not None:
                return len(alignments)
            if sizes is not None:
                return len(sizes)

        cols = init_size(cols, col_alignments, col_widths)
        rows = init_size(rows, row_alignments, row_heights)

        # calculate rows cols
        if rows is None and cols is None:
            cols = math.ceil(math.sqrt(len(mobs)))
            # make the grid as close to quadratic as possible.
            # choosing cols first can results in cols>rows.
            # This is favored over rows>cols since in general
            # the sceene is wider than high.
        if rows is None:
            rows = math.ceil(len(mobs) / cols)
        if cols is None:
            cols = math.ceil(len(mobs) / rows)
        if rows * cols < len(mobs):
            raise ValueError("Too few rows and columns to fit all submobjetcs.")
        # rows and cols are now finally valid.

        if isinstance(buff, tuple):
            buff_x = buff[0]
            buff_y = buff[1]
        else:
            buff_x = buff_y = buff

        # Initialize alignments correctly
        def init_alignments(alignments, num, mapping, name, dir_):
            if alignments is None:
                # Use cell_alignment as fallback
                return [cell_alignment * dir_] * num
            if len(alignments) != num:
                raise ValueError(f"{name}_alignments has a mismatching size.")
            alignments = list(alignments)
            for i in range(num):
                alignments[i] = mapping[alignments[i]]
            return alignments

        row_alignments = init_alignments(
            row_alignments,
            rows,
            {"u": UP, "c": ORIGIN, "d": DOWN},
            "row",
            RIGHT,
        )
        col_alignments = init_alignments(
            col_alignments,
            cols,
            {"l": LEFT, "c": ORIGIN, "r": RIGHT},
            "col",
            UP,
        )
        # Now row_alignment[r] + col_alignment[c] is the alignment in cell [r][c]

        mapper = {
            "dr": lambda r, c: (rows - r - 1) + c * rows,
            "dl": lambda r, c: (rows - r - 1) + (cols - c - 1) * rows,
            "ur": lambda r, c: r + c * rows,
            "ul": lambda r, c: r + (cols - c - 1) * rows,
            "rd": lambda r, c: (rows - r - 1) * cols + c,
            "ld": lambda r, c: (rows - r - 1) * cols + (cols - c - 1),
            "ru": lambda r, c: r * cols + c,
            "lu": lambda r, c: r * cols + (cols - c - 1),
        }
        if flow_order not in mapper:
            raise ValueError(
                'flow_order must be one of the following values: "dr", "rd", "ld" "dl", "ru", "ur", "lu", "ul".',
            )
        flow_order = mapper[flow_order]

        # Reverse row_alignments and row_heights. Necessary since the
        # grid filling is handled bottom up for simplicity reasons.
        def reverse(maybe_list):
            if maybe_list is not None:
                maybe_list = list(maybe_list)
                maybe_list.reverse()
                return maybe_list

        row_alignments = reverse(row_alignments)
        row_heights = reverse(row_heights)

        placeholder = Mobject()
        # Used to fill up the grid temporarily, doesn't get added to the scene.
        # In this case a Mobject is better than None since it has width and height
        # properties of 0.

        mobs.extend([placeholder] * (rows * cols - len(mobs)))
        grid = [[mobs[flow_order(r, c)] for c in range(cols)] for r in range(rows)]

        measured_heigths = [
            max(grid[r][c].height for c in range(cols)) for r in range(rows)
        ]
        measured_widths = [
            max(grid[r][c].width for r in range(rows)) for c in range(cols)
        ]

        # Initialize row_heights / col_widths correctly using measurements as fallback
        def init_sizes(sizes, num, measures, name):
            if sizes is None:
                sizes = [None] * num
            if len(sizes) != num:
                raise ValueError(f"{name} has a mismatching size.")
            return [
                sizes[i] if sizes[i] is not None else measures[i] for i in range(num)
            ]

        heights = init_sizes(row_heights, rows, measured_heigths, "row_heights")
        widths = init_sizes(col_widths, cols, measured_widths, "col_widths")

        x, y = 0, 0
        for r in range(rows):
            x = 0
            for c in range(cols):
                if grid[r][c] is not placeholder:
                    alignment = row_alignments[r] + col_alignments[c]
                    line = Line(
                        x * RIGHT + y * UP,
                        (x + widths[c]) * RIGHT + (y + heights[r]) * UP,
                    )
                    # Use a mobject to avoid rewriting align inside
                    # box code that Mobject.move_to(Mobject) already
                    # includes.

                    grid[r][c].move_to(line, alignment)
                x += widths[c] + buff_x
            y += heights[r] + buff_y

        self.move_to(start_pos)
        return self

    def sort(
        self,
        point_to_num_func: Callable[[Point3DLike], float] = lambda p: p[0],
        submob_func: Callable[[Mobject], Any] | None = None,
    ) -> Self:
        """Sorts the list of :attr:`submobjects` by a function defined by ``submob_func``."""
        if submob_func is None:

            def submob_func(m: Mobject) -> float:
                return point_to_num_func(m.get_center())

        self.submobjects.sort(key=submob_func)
        return self

    def shuffle(self, recursive: bool = False) -> None:
        """Shuffles the list of :attr:`submobjects`."""
        if recursive:
            for submob in self.submobjects:
                submob.shuffle(recursive=True)
        random.shuffle(self.submobjects)

    def invert(self, recursive: bool = False) -> None:
        """Inverts the list of :attr:`submobjects`.

        Parameters
        ----------
        recursive
            If ``True``, all submobject lists of this mobject's family are inverted.

        Examples
        --------

        .. manim:: InvertSumobjectsExample

            class InvertSumobjectsExample(Scene):
                def construct(self):
                    s = VGroup(*[Dot().shift(i*0.1*RIGHT) for i in range(-20,20)])
                    s2 = s.copy()
                    s2.invert()
                    s2.shift(DOWN)
                    self.play(Write(s), Write(s2))
        """
        if recursive:
            for submob in self.submobjects:
                submob.invert(recursive=True)
        self.submobjects.reverse()

    # Just here to keep from breaking old scenes.
    def arrange_submobjects(self, *args, **kwargs) -> Self:
        """Arrange the position of :attr:`submobjects` with a small buffer.

        Examples
        --------

        .. manim:: ArrangeSumobjectsExample
            :save_last_frame:

            class ArrangeSumobjectsExample(Scene):
                def construct(self):
                    s= VGroup(*[Dot().shift(i*0.1*RIGHT*np.random.uniform(-1,1)+UP*np.random.uniform(-1,1)) for i in range(0,15)])
                    s.shift(UP).set_color(BLUE)
                    s2= s.copy().set_color(RED)
                    s2.arrange_submobjects()
                    s2.shift(DOWN)
                    self.add(s,s2)

        """
        return self.arrange(*args, **kwargs)

    def sort_submobjects(self, *args, **kwargs) -> Self:
        """Sort the :attr:`submobjects`"""
        return self.sort(*args, **kwargs)

    def shuffle_submobjects(self, *args, **kwargs) -> None:
        """Shuffles the order of :attr:`submobjects`

        Examples
        --------

        .. manim:: ShuffleSubmobjectsExample

            class ShuffleSubmobjectsExample(Scene):
                def construct(self):
                    s= VGroup(*[Dot().shift(i*0.1*RIGHT) for i in range(-20,20)])
                    s2= s.copy()
                    s2.shuffle_submobjects()
                    s2.shift(DOWN)
                    self.play(Write(s), Write(s2))
        """
        return self.shuffle(*args, **kwargs)

    # Alignment
    def align_data(self, mobject: Mobject, skip_point_alignment: bool = False) -> None:
        """Aligns the data of this mobject with another mobject.

        Afterwards, the two mobjects will have the same number of submobjects
        (see :meth:`.align_submobjects`), the same parent structure (see
        :meth:`.null_point_align`). If ``skip_point_alignment`` is false,
        they will also have the same number of points (see :meth:`.align_points`).

        Parameters
        ----------
        mobject
            The other mobject this mobject should be aligned to.
        skip_point_alignment
            Controls whether or not the computationally expensive
            point alignment is skipped (default: False).
        """
        self.null_point_align(mobject)
        self.align_submobjects(mobject)
        if not skip_point_alignment:
            self.align_points(mobject)
        # Recurse
        for m1, m2 in zip(self.submobjects, mobject.submobjects):
            m1.align_data(m2)

    def get_point_mobject(self, center=None):
        """The simplest :class:`~.Mobject` to be transformed to or from self.
        Should by a point of the appropriate type
        """
        msg = f"get_point_mobject not implemented for {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def align_points(self, mobject: Mobject) -> Self:
        count1 = self.get_num_points()
        count2 = mobject.get_num_points()
        if count1 < count2:
            self.align_points_with_larger(mobject)
        elif count2 < count1:
            mobject.align_points_with_larger(self)
        return self

    def align_points_with_larger(self, larger_mobject: Mobject):
        raise NotImplementedError("Please override in a child class.")

    def align_submobjects(self, mobject: Mobject) -> Self:
        mob1 = self
        mob2 = mobject
        n1 = len(mob1.submobjects)
        n2 = len(mob2.submobjects)
        mob1.add_n_more_submobjects(max(0, n2 - n1))
        mob2.add_n_more_submobjects(max(0, n1 - n2))
        return self

    def null_point_align(self, mobject: Mobject):
        """If a :class:`~.Mobject` with points is being aligned to
        one without, treat both as groups, and push
        the one with points into its own submobjects
        list.

        Returns
        -------
        :class:`Mobject`
            ``self``
        """
        for m1, m2 in (self, mobject), (mobject, self):
            if m1.has_no_points() and m2.has_points():
                m2.push_self_into_submobjects()
        return self

    def push_self_into_submobjects(self) -> Self:
        copy = self.copy()
        copy.submobjects = []
        self.reset_points()
        self.add(copy)
        return self

    def add_n_more_submobjects(self, n: int) -> Self | None:
        if n == 0:
            return None

        curr = len(self.submobjects)
        if curr == 0:
            # If empty, simply add n point mobjects
            self.submobjects = [self.get_point_mobject() for k in range(n)]
            return None

        target = curr + n
        # TODO, factor this out to utils so as to reuse
        # with VMobject.insert_n_curves
        repeat_indices = (np.arange(target) * curr) // target
        split_factors = [sum(repeat_indices == i) for i in range(curr)]
        new_submobs = []
        for submob, sf in zip(self.submobjects, split_factors):
            new_submobs.append(submob)
            for _ in range(1, sf):
                new_submobs.append(submob.copy().fade(1))
        self.submobjects = new_submobs
        return self

    def repeat_submobject(self, submob: Mobject) -> Self:
        return submob.copy()

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: PathFuncType = straight_path(),
    ) -> Self:
        """Turns this :class:`~.Mobject` into an interpolation between ``mobject1``
        and ``mobject2``.

        Examples
        --------

        .. manim:: DotInterpolation
            :save_last_frame:

            class DotInterpolation(Scene):
                def construct(self):
                    dotR = Dot(color=DARK_GREY)
                    dotR.shift(2 * RIGHT)
                    dotL = Dot(color=WHITE)
                    dotL.shift(2 * LEFT)

                    dotMiddle = VMobject().interpolate(dotL, dotR, alpha=0.3)

                    self.add(dotL, dotR, dotMiddle)
        """
        self.points = path_func(mobject1.points, mobject2.points, alpha)
        self.interpolate_color(mobject1, mobject2, alpha)
        return self

    def interpolate_color(self, mobject1: Mobject, mobject2: Mobject, alpha: float):
        raise NotImplementedError("Please override in a child class.")

    def become(
        self,
        mobject: Mobject,
        match_height: bool = False,
        match_width: bool = False,
        match_depth: bool = False,
        match_center: bool = False,
        stretch: bool = False,
    ) -> Self:
        """Edit points, colors and submobjects to be identical
        to another :class:`~.Mobject`

        .. note::

            If both match_height and match_width are ``True`` then the transformed :class:`~.Mobject`
            will match the height first and then the width.

        Parameters
        ----------
        match_height
            Whether or not to preserve the height of the original
            :class:`~.Mobject`.
        match_width
            Whether or not to preserve the width of the original
            :class:`~.Mobject`.
        match_depth
            Whether or not to preserve the depth of the original
            :class:`~.Mobject`.
        match_center
            Whether or not to preserve the center of the original
            :class:`~.Mobject`.
        stretch
            Whether or not to stretch the target mobject to match the
            the proportions of the original :class:`~.Mobject`.

        Examples
        --------
        .. manim:: BecomeScene

            class BecomeScene(Scene):
                def construct(self):
                    circ = Circle(fill_color=RED, fill_opacity=0.8)
                    square = Square(fill_color=BLUE, fill_opacity=0.2)
                    self.add(circ)
                    self.wait(0.5)
                    circ.become(square)
                    self.wait(0.5)


        The following examples illustrate how mobject measurements
        change when using the ``match_...`` and ``stretch`` arguments.
        We start with a rectangle that is 2 units high and 4 units wide,
        which we want to turn into a circle of radius 3::

            >>> from manim import Rectangle, Circle
            >>> import numpy as np
            >>> rect = Rectangle(height=2, width=4)
            >>> circ = Circle(radius=3)

        With ``stretch=True``, the target circle is deformed to match
        the proportions of the rectangle, which results in the target
        mobject being an ellipse with height 2 and width 4. We can
        check that the resulting points satisfy the ellipse equation
        :math:`x^2/a^2 + y^2/b^2 = 1` with :math:`a = 4/2` and :math:`b = 2/2`
        being the semi-axes::

            >>> result = rect.copy().become(circ, stretch=True)
            >>> result.height, result.width
            (np.float64(2.0), np.float64(4.0))
            >>> ellipse_points = np.array(result.get_anchors())
            >>> ellipse_eq = np.sum(ellipse_points**2 * [1/4, 1, 0], axis=1)
            >>> np.allclose(ellipse_eq, 1)
            True

        With ``match_height=True`` and ``match_width=True`` the circle is
        scaled such that the height or the width of the rectangle will
        be preserved, respectively.
        The points of the resulting mobject satisfy the circle equation
        :math:`x^2 + y^2 = r^2` for the corresponding radius :math:`r`::

            >>> result = rect.copy().become(circ, match_height=True)
            >>> result.height, result.width
            (np.float64(2.0), np.float64(2.0))
            >>> circle_points = np.array(result.get_anchors())
            >>> circle_eq = np.sum(circle_points**2, axis=1)
            >>> np.allclose(circle_eq, 1)
            True
            >>> result = rect.copy().become(circ, match_width=True)
            >>> result.height, result.width
            (np.float64(4.0), np.float64(4.0))
            >>> circle_points = np.array(result.get_anchors())
            >>> circle_eq = np.sum(circle_points**2, axis=1)
            >>> np.allclose(circle_eq, 2**2)
            True

        With ``match_center=True``, the resulting mobject is moved such that
        its center is the same as the center of the original mobject::

            >>> rect = rect.shift(np.array([0, 1, 0]))
            >>> np.allclose(rect.get_center(), circ.get_center())
            False
            >>> result = rect.copy().become(circ, match_center=True)
            >>> np.allclose(rect.get_center(), result.get_center())
            True
        """
        mobject = mobject.copy()
        if stretch:
            mobject.stretch_to_fit_height(self.height)
            mobject.stretch_to_fit_width(self.width)
            mobject.stretch_to_fit_depth(self.depth)
        else:
            if match_height:
                mobject.match_height(self)
            if match_width:
                mobject.match_width(self)
            if match_depth:
                mobject.match_depth(self)

        if match_center:
            mobject.move_to(self.get_center())

        self.align_data(mobject, skip_point_alignment=True)
        for sm1, sm2 in zip(self.get_family(), mobject.get_family()):
            sm1.points = np.array(sm2.points)
            sm1.interpolate_color(sm1, sm2, 1)
        return self

    def match_points(self, mobject: Mobject, copy_submobjects: bool = True) -> Self:
        """Edit points, positions, and submobjects to be identical
        to another :class:`~.Mobject`, while keeping the style unchanged.

        Examples
        --------
        .. manim:: MatchPointsScene

            class MatchPointsScene(Scene):
                def construct(self):
                    circ = Circle(fill_color=RED, fill_opacity=0.8)
                    square = Square(fill_color=BLUE, fill_opacity=0.2)
                    self.add(circ)
                    self.wait(0.5)
                    self.play(circ.animate.match_points(square))
                    self.wait(0.5)
        """
        for sm1, sm2 in zip(self.get_family(), mobject.get_family()):
            sm1.points = np.array(sm2.points)
        return self

    # Errors
    def throw_error_if_no_points(self) -> None:
        if self.has_no_points():
            caller_name = sys._getframe(1).f_code.co_name
            raise Exception(
                f"Cannot call Mobject.{caller_name} for a Mobject with no points",
            )

    # About z-index
    def set_z_index(
        self,
        z_index_value: float,
        family: bool = True,
    ) -> Self:
        """Sets the :class:`~.Mobject`'s :attr:`z_index` to the value specified in `z_index_value`.

        Parameters
        ----------
        z_index_value
            The new value of :attr:`z_index` set.
        family
            If ``True``, the :attr:`z_index` value of all submobjects is also set.

        Returns
        -------
        :class:`Mobject`
            The Mobject itself, after :attr:`z_index` is set. For chaining purposes. (Returns `self`.)

        Examples
        --------
        .. manim:: SetZIndex
            :save_last_frame:

            class SetZIndex(Scene):
                def construct(self):
                    text = Text('z_index = 3', color = PURE_RED).shift(UP).set_z_index(3)
                    square = Square(2, fill_opacity=1).set_z_index(2)
                    tex = Tex(r'zIndex = 1', color = PURE_BLUE).shift(DOWN).set_z_index(1)
                    circle = Circle(radius = 1.7, color = GREEN, fill_opacity = 1) # z_index = 0

                    # Displaying order is now defined by z_index values
                    self.add(text)
                    self.add(square)
                    self.add(tex)
                    self.add(circle)
        """
        if family:
            for submob in self.submobjects:
                submob.set_z_index(z_index_value, family=family)
        self.z_index = z_index_value
        return self

    def set_z_index_by_z_Point3D(self) -> Self:
        """Sets the :class:`~.Mobject`'s z Point3D to the value of :attr:`z_index`.

        Returns
        -------
        :class:`Mobject`
            The Mobject itself, after :attr:`z_index` is set. (Returns `self`.)
        """
        z_coord = self.get_center()[-1]
        self.set_z_index(z_coord)
        return self


class Group(Mobject, metaclass=ConvertToOpenGL):
    """Groups together multiple :class:`Mobjects <.Mobject>`.

    Notes
    -----
    When adding the same mobject more than once, repetitions are ignored.
    Use :meth:`.Mobject.copy` to create a separate copy which can then
    be added to the group.
    """

    def __init__(self, *mobjects, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add(*mobjects)


class _AnimationBuilder:
    def __init__(self, mobject) -> None:
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

    def __getattr__(self, method_name) -> types.MethodType:
        method = getattr(self.mobject.target, method_name)
        has_overridden_animation = hasattr(method, "_override_animate")

        if (self.is_chaining and has_overridden_animation) or self.overridden_animation:
            raise NotImplementedError(
                "Method chaining is currently not supported for overridden animations",
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
        from ..animation.transform import (  # is this to prevent circular import?
            _MethodAnimation,
        )

        if self.overridden_animation:
            anim = self.overridden_animation
        else:
            anim = _MethodAnimation(self.mobject, self.methods)

        for attr, value in self.anim_args.items():
            setattr(anim, attr, value)

        return anim


def override_animate(method) -> types.FunctionType:
    r"""Decorator for overriding method animations.

    This allows to specify a method (returning an :class:`~.Animation`)
    which is called when the decorated method is used with the ``.animate`` syntax
    for animating the application of a method.

    .. seealso::

        :attr:`Mobject.animate`

    .. note::

        Overridden methods cannot be combined with normal or other overridden
        methods using method chaining with the ``.animate`` syntax.


    Examples
    --------

    .. manim:: AnimationOverrideExample

        class CircleWithContent(VGroup):
            def __init__(self, content):
                super().__init__()
                self.circle = Circle()
                self.content = content
                self.add(self.circle, content)
                content.move_to(self.circle.get_center())

            def clear_content(self):
                self.remove(self.content)
                self.content = None

            @override_animate(clear_content)
            def _clear_content_animation(self, anim_args=None):
                if anim_args is None:
                    anim_args = {}
                anim = Uncreate(self.content, **anim_args)
                self.clear_content()
                return anim

        class AnimationOverrideExample(Scene):
            def construct(self):
                t = Text("hello!")
                my_mobject = CircleWithContent(t)
                self.play(Create(my_mobject))
                self.play(my_mobject.animate.clear_content())
                self.wait()

    """

    def decorator(animation_method):
        method._override_animate = animation_method
        return animation_method

    return decorator
