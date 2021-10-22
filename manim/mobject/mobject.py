"""Base classes for objects that can be displayed."""


__all__ = ["Mobject", "Group", "override_animate"]


import copy
import itertools as it
import operator as op
import random
import sys
import types
import warnings
from functools import partialmethod, reduce
from math import ceil
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from colour import Color

from .. import config
from ..constants import *
from ..utils.color import (
    BLACK,
    WHITE,
    YELLOW_C,
    Colors,
    color_gradient,
    interpolate_color,
)
from ..utils.deprecation import deprecated
from ..utils.exceptions import MultiAnimationOverrideException
from ..utils.iterables import list_update, remove_list_redundancies
from ..utils.paths import straight_path
from ..utils.simple_functions import get_parameters
from ..utils.space_ops import angle_between_vectors, normalize, rotation_matrix
from .opengl_compatibility import ConvertToOpenGL

# TODO: Explain array_attrs

Updater = Union[Callable[["Mobject"], None], Callable[["Mobject", float], None]]
T = TypeVar("T", bound="Mobject")

if TYPE_CHECKING:
    from ..animation.animation import Animation


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
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.animation_overrides: Dict[
            Type["Animation"],
            Callable[["Mobject"], "Animation"],
        ] = {}
        cls._add_intrinsic_animation_overrides()
        cls._original__init__ = cls.__init__

    def __init__(self, color=WHITE, name=None, dim=3, target=None, z_index=0):
        self.color = Color(color) if color else None
        self.name = self.__class__.__name__ if name is None else name
        self.dim = dim
        self.target = target
        self.z_index = z_index
        self.point_hash = None
        self.submobjects = []
        self.updaters = []
        self.updating_suspended = False
        self.reset_points()
        self.generate_points()
        self.init_colors()

    @classmethod
    def animation_override_for(
        cls,
        animation_class: Type["Animation"],
    ) -> "Optional[Callable[[Mobject, ...], Animation]]":
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
    def _add_intrinsic_animation_overrides(cls):
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
        animation_class: Type["Animation"],
        override_func: "Callable[[Mobject, ...], Animation]",
    ):
        """Add an animation override.

        This does not apply to subclasses.

        Parameters
        ----------
        animation_class
            The animation type to be overridden
        override_func
            The function returning an animation replacing the default animation. It gets
            passed the parameters given to the animnation constructor.

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
    def set_default(cls, **kwargs):
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
            (<Color #83c167>, 0.25)
            >>> Square.set_default()
            >>> s = Square(); s.color, s.fill_opacity
            (<Color white>, 0.0)

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
    def animate(self):
        """Used to animate the application of a method.

        .. warning::

            Passing multiple animations for the same :class:`Mobject` in one
            call to :meth:`~.Scene.play` is discouraged and will most likely
            not work properly. Instead of writing an animation like

            ::

                self.play(my_mobject.animate.shift(RIGHT), my_mobject.animate.rotate(PI))

            make use of method chaining for ``animate``, meaning::

                self.play(my_mobject.animate.shift(RIGHT).rotate(PI))

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

    def __deepcopy__(self, clone_from_id):
        cls = self.__class__
        result = cls.__new__(cls)
        clone_from_id[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, clone_from_id))
        result.original_id = str(id(self))
        return result

    def __repr__(self):
        if config["renderer"] == "opengl":
            return super().__repr__()
        else:
            return str(self.name)

    def reset_points(self):
        """Sets :attr:`points` to be an empty array."""
        self.points = np.zeros((0, self.dim))

    def init_colors(self):
        """Initializes the colors.

        Gets called upon creation. This is an empty method that can be implemented by
        subclasses.
        """
        pass

    def generate_points(self):
        """Initializes :attr:`points` and therefore the shape.

        Gets called upon creation. This is an empty method that can be implemented by
        subclasses.
        """
        pass

    def add(self, *mobjects: "Mobject") -> "Mobject":
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

        Adding an object to itself raises an error::

            >>> outer.add(outer)
            Traceback (most recent call last):
            ...
            ValueError: Mobject cannot contain self

        """
        for m in mobjects:
            if not isinstance(m, Mobject):
                raise TypeError("All submobjects must be of type Mobject")
            if m is self:
                raise ValueError("Mobject cannot contain self")
        self.submobjects = list_update(self.submobjects, mobjects)
        return self

    def __add__(self, mobject):
        raise NotImplementedError

    def __iadd__(self, mobject):
        raise NotImplementedError

    def add_to_back(self, *mobjects: "Mobject") -> "Mobject":
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
        if self in mobjects:
            raise ValueError("A mobject shouldn't contain itself")

        for mobject in mobjects:
            if not isinstance(mobject, Mobject):
                raise TypeError("All submobjects must be of type Mobject")

        self.remove(*mobjects)
        # dict.fromkeys() removes duplicates while maintaining order
        self.submobjects = list(dict.fromkeys(mobjects)) + self.submobjects
        return self

    def remove(self, *mobjects: "Mobject") -> "Mobject":
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

    def set(self, **kwargs) -> "Mobject":
        """Sets attributes.

        Mainly to be used along with :attr:`animate` to
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

    def __getattr__(self, attr):
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
    def width(self):
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
    def width(self, value):
        self.scale_to_fit_width(value)

    @property
    def height(self):
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
    def height(self, value):
        self.scale_to_fit_height(value)

    @property
    def depth(self):
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
    def depth(self, value):
        self.scale_to_fit_depth(value)

    def get_array_attrs(self):
        return ["points"]

    def apply_over_attr_arrays(self, func):
        for attr in self.get_array_attrs():
            setattr(self, attr, func(getattr(self, attr)))
        return self

    # Displaying

    def get_image(self, camera=None):
        if camera is None:
            from ..camera.camera import Camera

            camera = Camera()
        camera.capture_mobject(self)
        return camera.get_image()

    def show(self, camera=None):
        self.get_image(camera=camera).show()

    def save_image(self, name=None):
        """Saves an image of only this :class:`Mobject` at its position to a png
        file."""
        self.get_image().save(
            Path(config.get_dir("video_dir")).joinpath((name or str(self)) + ".png"),
        )

    def copy(self: T) -> T:
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

    def generate_target(self, use_deepcopy=False):
        self.target = None  # Prevent unbounded linear recursion
        if use_deepcopy:
            self.target = copy.deepcopy(self)
        else:
            self.target = self.copy()
        return self.target

    # Updating

    def update(self, dt: float = 0, recursive: bool = True) -> "Mobject":
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
            parameters = get_parameters(updater)
            if "dt" in parameters:
                updater(self, dt)
            else:
                updater(self)
        if recursive:
            for submob in self.submobjects:
                submob.update(dt, recursive)
        return self

    def get_time_based_updaters(self) -> List[Updater]:
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
        return [updater for updater in self.updaters if "dt" in get_parameters(updater)]

    def has_time_based_updater(self) -> bool:
        """Test if ``self`` has a time based updater.

        Returns
        -------
        class:`bool`
            ``True`` if at least one updater uses the ``dt`` parameter, ``False``
            otherwise.

        See Also
        --------
        :meth:`get_time_based_updaters`

        """
        return any("dt" in get_parameters(updater) for updater in self.updaters)

    def get_updaters(self) -> List[Updater]:
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

    def get_family_updaters(self):
        return list(it.chain(*(sm.get_updaters() for sm in self.get_family())))

    def add_updater(
        self,
        update_function: Updater,
        index: Optional[int] = None,
        call_updater: bool = False,
    ) -> "Mobject":
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
                    line = Square()

                    #Let the line rotate 90° per second
                    line.add_updater(lambda mobject, dt: mobject.rotate(dt*90*DEGREES))
                    self.add(line)
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
            update_function(self, 0)
        return self

    def remove_updater(self, update_function: Updater) -> "Mobject":
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

    def clear_updaters(self, recursive: bool = True) -> "Mobject":
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

    def match_updaters(self, mobject: "Mobject") -> "Mobject":
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

    def suspend_updating(self, recursive: bool = True) -> "Mobject":
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

    def resume_updating(self, recursive: bool = True) -> "Mobject":
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

    def apply_to_family(self, func: Callable[["Mobject"], None]) -> "Mobject":
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

    def shift(self, *vectors: np.ndarray) -> "Mobject":
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

    def scale(self, scale_factor: float, **kwargs) -> "Mobject":
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
        Mobject
            The scaled mobject.

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

    def rotate_about_origin(self, angle, axis=OUT, axes=[]):
        """Rotates the :class:`~.Mobject` about the ORIGIN, which is at [0,0,0]."""
        return self.rotate(angle, axis, about_point=ORIGIN)

    def rotate(
        self,
        angle,
        axis=OUT,
        about_point: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        """Rotates the :class:`~.Mobject` about a certain point."""
        rot_matrix = rotation_matrix(angle, axis)
        self.apply_points_function_about_point(
            lambda points: np.dot(points, rot_matrix.T), about_point, **kwargs
        )
        return self

    def flip(self, axis=UP, **kwargs):
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

    def stretch(self, factor, dim, **kwargs):
        def func(points):
            points[:, dim] *= factor
            return points

        self.apply_points_function_about_point(func, **kwargs)
        return self

    def apply_function(self, function, **kwargs):
        # Default to applying matrix about the origin, not mobjects center
        if len(kwargs) == 0:
            kwargs["about_point"] = ORIGIN
        self.apply_points_function_about_point(
            lambda points: np.apply_along_axis(function, 1, points), **kwargs
        )
        return self

    def apply_function_to_position(self, function):
        self.move_to(function(self.get_center()))
        return self

    def apply_function_to_submobject_positions(self, function):
        for submob in self.submobjects:
            submob.apply_function_to_position(function)
        return self

    def apply_matrix(self, matrix, **kwargs):
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

    def apply_complex_function(self, function, **kwargs):
        """Applies a complex function to a :class:`Mobject`.
        The x and y coordinates correspond to the real and imaginary parts respectively.

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

    def wag(self, direction=RIGHT, axis=DOWN, wag_factor=1.0):
        for mob in self.family_members_with_points():
            alphas = np.dot(mob.points, np.transpose(axis))
            alphas -= min(alphas)
            alphas /= max(alphas)
            alphas = alphas ** wag_factor
            mob.points += np.dot(
                alphas.reshape((len(alphas), 1)),
                np.array(direction).reshape((1, mob.dim)),
            )
        return self

    def reverse_points(self):
        for mob in self.family_members_with_points():
            mob.apply_over_attr_arrays(lambda arr: np.array(list(reversed(arr))))
        return self

    def repeat(self, count: int):
        """This can make transition animations nicer"""

        def repeat_array(array):
            return reduce(lambda a1, a2: np.append(a1, a2, axis=0), [array] * count)

        for mob in self.family_members_with_points():
            mob.apply_over_attr_arrays(repeat_array)
        return self

    # In place operations.
    # Note, much of these are now redundant with default behavior of
    # above methods

    def apply_points_function_about_point(
        self,
        func,
        about_point=None,
        about_edge=None,
    ):
        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
            about_point = self.get_critical_point(about_edge)
        for mob in self.family_members_with_points():
            mob.points -= about_point
            mob.points = func(mob.points)
            mob.points += about_point
        return self

    @deprecated(
        since="v0.11.0",
        until="v0.12.0",
        replacement="rotate",
    )
    def rotate_in_place(self, angle, axis=OUT):
        # redundant with default behavior of rotate now.
        return self.rotate(angle, axis=axis)

    @deprecated(
        since="v0.11.0",
        until="v0.12.0",
        replacement="scale",
    )
    def scale_in_place(self, scale_factor, **kwargs):
        # Redundant with default behavior of scale now.
        return self.scale(scale_factor, **kwargs)

    @deprecated(
        since="v0.11.0",
        until="v0.12.0",
        replacement="scale",
    )
    def scale_about_point(self, scale_factor, point):
        # Redundant with default behavior of scale now.
        return self.scale(scale_factor, about_point=point)

    def pose_at_angle(self, **kwargs):
        self.rotate(TAU / 14, RIGHT + UP, **kwargs)
        return self

    # Positioning methods

    def center(self):
        self.shift(-self.get_center())
        return self

    def align_on_border(self, direction, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER):
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

    def to_corner(self, corner=LEFT + DOWN, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER):
        return self.align_on_border(corner, buff)

    def to_edge(self, edge=LEFT, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER):
        return self.align_on_border(edge, buff)

    def next_to(
        self,
        mobject_or_point,
        direction=RIGHT,
        buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        aligned_edge=ORIGIN,
        submobject_to_align=None,
        index_of_submobject_to_align=None,
        coor_mask=np.array([1, 1, 1]),
    ):
        """Move this :class:`~.Mobject` next to another's :class:`~.Mobject` or coordinate.

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

    def shift_onto_screen(self, **kwargs):
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
        if self.get_top()[1] < -config["frame_y_radius"]:
            return True
        return False

    def stretch_about_point(self, factor, dim, point):
        return self.stretch(factor, dim, about_point=point)

    @deprecated(
        since="v0.11.0",
        until="v0.12.0",
        replacement="stretch",
    )
    def stretch_in_place(self, factor, dim):
        # Now redundant with stretch
        return self.stretch(factor, dim)

    def rescale_to_fit(self, length, dim, stretch=False, **kwargs):
        old_length = self.length_over_dim(dim)
        if old_length == 0:
            return self
        if stretch:
            self.stretch(length / old_length, dim, **kwargs)
        else:
            self.scale(length / old_length, **kwargs)
        return self

    def scale_to_fit_width(self, width, **kwargs):
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
            2.0
            >>> sq.scale_to_fit_width(5)
            Square
            >>> sq.width
            5.0
            >>> sq.height
            5.0
        """

        return self.rescale_to_fit(width, 0, stretch=False, **kwargs)

    def stretch_to_fit_width(self, width, **kwargs):
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
            2.0
            >>> sq.stretch_to_fit_width(5)
            Square
            >>> sq.width
            5.0
            >>> sq.height
            2.0
        """

        return self.rescale_to_fit(width, 0, stretch=True, **kwargs)

    def scale_to_fit_height(self, height, **kwargs):
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
            2.0
            >>> sq.scale_to_fit_height(5)
            Square
            >>> sq.height
            5.0
            >>> sq.width
            5.0
        """

        return self.rescale_to_fit(height, 1, stretch=False, **kwargs)

    def stretch_to_fit_height(self, height, **kwargs):
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
            2.0
            >>> sq.stretch_to_fit_height(5)
            Square
            >>> sq.height
            5.0
            >>> sq.width
            2.0
        """

        return self.rescale_to_fit(height, 1, stretch=True, **kwargs)

    def scale_to_fit_depth(self, depth, **kwargs):
        """Scales the :class:`~.Mobject` to fit a depth while keeping width/height proportional."""

        return self.rescale_to_fit(depth, 2, stretch=False, **kwargs)

    def stretch_to_fit_depth(self, depth, **kwargs):
        """Stretches the :class:`~.Mobject` to fit a depth, not keeping width/height proportional."""

        return self.rescale_to_fit(depth, 2, stretch=True, **kwargs)

    def set_coord(self, value, dim, direction=ORIGIN):
        curr = self.get_coord(dim, direction)
        shift_vect = np.zeros(self.dim)
        shift_vect[dim] = value - curr
        self.shift(shift_vect)
        return self

    def set_x(self, x, direction=ORIGIN):
        """Set x value of the center of the :class:`~.Mobject` (``int`` or ``float``)"""
        return self.set_coord(x, 0, direction)

    def set_y(self, y, direction=ORIGIN):
        """Set y value of the center of the :class:`~.Mobject` (``int`` or ``float``)"""
        return self.set_coord(y, 1, direction)

    def set_z(self, z, direction=ORIGIN):
        """Set z value of the center of the :class:`~.Mobject` (``int`` or ``float``)"""
        return self.set_coord(z, 2, direction)

    def space_out_submobjects(self, factor=1.5, **kwargs):
        self.scale(factor, **kwargs)
        for submob in self.submobjects:
            submob.scale(1.0 / factor)
        return self

    def move_to(
        self,
        point_or_mobject,
        aligned_edge=ORIGIN,
        coor_mask=np.array([1, 1, 1]),
    ):
        """Move center of the :class:`~.Mobject` to certain coordinate."""
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_critical_point(aligned_edge)
        else:
            target = point_or_mobject
        point_to_align = self.get_critical_point(aligned_edge)
        self.shift((target - point_to_align) * coor_mask)
        return self

    def replace(self, mobject, dim_to_match=0, stretch=False):
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
        mobject: "Mobject",
        dim_to_match=0,
        stretch=False,
        buff=MED_SMALL_BUFF,
    ):
        self.replace(mobject, dim_to_match, stretch)
        length = mobject.length_over_dim(dim_to_match)
        self.scale((length + buff) / length)
        return self

    def put_start_and_end_on(self, start, end):
        curr_start, curr_end = self.get_start_and_end()
        curr_vect = curr_end - curr_start
        if np.all(curr_vect == 0):
            raise Exception("Cannot position endpoints of closed loop")
        target_vect = np.array(end) - np.array(start)
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
        self, color: Colors = BLACK, opacity: float = 0.75, **kwargs
    ):
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
        from ..mobject.shape_matchers import BackgroundRectangle

        self.background_rectangle = BackgroundRectangle(
            self, color=color, fill_opacity=opacity, **kwargs
        )
        self.add_to_back(self.background_rectangle)
        return self

    def add_background_rectangle_to_submobjects(self, **kwargs):
        for submobject in self.submobjects:
            submobject.add_background_rectangle(**kwargs)
        return self

    def add_background_rectangle_to_family_members_with_points(self, **kwargs):
        for mob in self.family_members_with_points():
            mob.add_background_rectangle(**kwargs)
        return self

    # Color functions

    def set_color(self, color: Color = YELLOW_C, family: bool = True):
        """Condition is function which takes in one arguments, (x, y, z).
        Here it just recurses to submobjects, but in subclasses this
        should be further implemented based on the the inner workings
        of color
        """
        if family:
            for submob in self.submobjects:
                submob.set_color(color, family=family)
        self.color = Color(color)
        return self

    def set_color_by_gradient(self, *colors):
        self.set_submobject_colors_by_gradient(*colors)
        return self

    def set_colors_by_radial_gradient(
        self,
        center=None,
        radius=1,
        inner_color=WHITE,
        outer_color=BLACK,
    ):
        self.set_submobject_colors_by_radial_gradient(
            center,
            radius,
            inner_color,
            outer_color,
        )
        return self

    def set_submobject_colors_by_gradient(self, *colors):
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
        center=None,
        radius=1,
        inner_color=WHITE,
        outer_color=BLACK,
    ):
        if center is None:
            center = self.get_center()

        for mob in self.family_members_with_points():
            t = np.linalg.norm(mob.get_center() - center) / radius
            t = min(t, 1)
            mob_color = interpolate_color(inner_color, outer_color, t)
            mob.set_color(mob_color, family=False)

        return self

    def to_original_color(self):
        self.set_color(self.color)
        return self

    def fade_to(self, color, alpha, family=True):
        if self.get_num_points() > 0:
            new_color = interpolate_color(self.get_color(), color, alpha)
            self.set_color(new_color, family=False)
        if family:
            for submob in self.submobjects:
                submob.fade_to(color, alpha)
        return self

    def fade(self, darkness=0.5, family=True):
        if family:
            for submob in self.submobjects:
                submob.fade(darkness, family)
        return self

    def get_color(self):
        """Returns the color of the :class:`~.Mobject`"""
        return self.color

    ##

    def save_state(self):
        """Save the current state (position, color & size). Can be restored with :meth:`~.Mobject.restore`."""
        if hasattr(self, "saved_state"):
            # Prevent exponential growth of data
            self.saved_state = None
        self.saved_state = self.copy()

        return self

    def restore(self):
        """Restores the state that was previously saved with :meth:`~.Mobject.save_state`."""
        if not hasattr(self, "saved_state") or self.save_state is None:
            raise Exception("Trying to restore without having saved")
        self.become(self.saved_state)
        return self

    ##

    def reduce_across_dimension(self, points_func, reduce_func, dim):
        points = self.get_all_points()
        if points is None or len(points) == 0:
            # Note, this default means things like empty VGroups
            # will appear to have a center at [0, 0, 0]
            return 0
        values = points_func(points[:, dim])
        return reduce_func(values)

    def nonempty_submobjects(self):
        return [
            submob
            for submob in self.submobjects
            if len(submob.submobjects) != 0 or len(submob.points) != 0
        ]

    def get_merged_array(self, array_attr):
        result = getattr(self, array_attr)
        for submob in self.submobjects:
            result = np.append(result, submob.get_merged_array(array_attr), axis=0)
            submob.get_merged_array(array_attr)
        return result

    def get_all_points(self):
        return self.get_merged_array("points")

    # Getters

    def get_points_defining_boundary(self):
        return self.get_all_points()

    def get_num_points(self):
        return len(self.points)

    def get_extremum_along_dim(self, points=None, dim=0, key=0):
        if points is None:
            points = self.get_points_defining_boundary()
        values = points[:, dim]
        if key < 0:
            return np.min(values)
        elif key == 0:
            return (np.min(values) + np.max(values)) / 2
        else:
            return np.max(values)

    def get_critical_point(self, direction):
        """Picture a box bounding the :class:`~.Mobject`.  Such a box has
        9 'critical points': 4 corners, 4 edge center, the
        center. This returns one of them, along the given direction.

        ::

            sample = Arc(start_angle=PI/7, angle = PI/5)

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

    def get_edge_center(self, direction) -> np.ndarray:
        """Get edge coordinates for certain direction."""
        return self.get_critical_point(direction)

    def get_corner(self, direction) -> np.ndarray:
        """Get corner coordinates for certain direction."""
        return self.get_critical_point(direction)

    def get_center(self) -> np.ndarray:
        """Get center coordinates"""
        return self.get_critical_point(np.zeros(self.dim))

    def get_center_of_mass(self):
        return np.apply_along_axis(np.mean, 0, self.get_all_points())

    def get_boundary_point(self, direction):
        all_points = self.get_points_defining_boundary()
        index = np.argmax(np.dot(all_points, np.array(direction).T))
        return all_points[index]

    def get_midpoint(self) -> np.ndarray:
        """Get coordinates of the middle of the path that forms the  :class:`~.Mobject`.

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

    def get_top(self) -> np.ndarray:
        """Get top coordinates of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(UP)

    def get_bottom(self) -> np.ndarray:
        """Get bottom coordinates of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(DOWN)

    def get_right(self) -> np.ndarray:
        """Get right coordinates of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(RIGHT)

    def get_left(self) -> np.ndarray:
        """Get left coordinates of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(LEFT)

    def get_zenith(self) -> np.ndarray:
        """Get zenith coordinates of a box bounding a 3D :class:`~.Mobject`."""
        return self.get_edge_center(OUT)

    def get_nadir(self) -> np.ndarray:
        """Get nadir (opposite the zenith) coordinates of a box bounding a 3D :class:`~.Mobject`."""
        return self.get_edge_center(IN)

    def length_over_dim(self, dim):
        """Measure the length of an :class:`~.Mobject` in a certain direction."""
        return (
            self.reduce_across_dimension(
                np.max,
                np.max,
                dim,
            )
            - self.reduce_across_dimension(np.min, np.min, dim)
        )

    def get_coord(self, dim, direction=ORIGIN):
        """Meant to generalize ``get_x``, ``get_y`` and ``get_z``"""
        return self.get_extremum_along_dim(dim=dim, key=direction[dim])

    def get_x(self, direction=ORIGIN) -> np.float64:
        """Returns x coordinate of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(0, direction)

    def get_y(self, direction=ORIGIN) -> np.float64:
        """Returns y coordinate of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(1, direction)

    def get_z(self, direction=ORIGIN) -> np.float64:
        """Returns z coordinate of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(2, direction)

    def get_start(self):
        """Returns the point, where the stroke that surrounds the :class:`~.Mobject` starts."""
        self.throw_error_if_no_points()
        return np.array(self.points[0])

    def get_end(self):
        """Returns the point, where the stroke that surrounds the :class:`~.Mobject` ends."""
        self.throw_error_if_no_points()
        return np.array(self.points[-1])

    def get_start_and_end(self):
        """Returns starting and ending point of a stroke as a ``tuple``."""
        return self.get_start(), self.get_end()

    def point_from_proportion(self, alpha):
        raise NotImplementedError("Please override in a child class.")

    def proportion_from_point(self, point):
        raise NotImplementedError("Please override in a child class.")

    def get_pieces(self, n_pieces):
        template = self.copy()
        template.submobjects = []
        alphas = np.linspace(0, 1, n_pieces + 1)
        return Group(
            *(
                template.copy().pointwise_become_partial(self, a1, a2)
                for a1, a2 in zip(alphas[:-1], alphas[1:])
            )
        )

    def get_z_index_reference_point(self):
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

    def match_color(self, mobject: "Mobject"):
        """Match the color with the color of another :class:`~.Mobject`."""
        return self.set_color(mobject.get_color())

    def match_dim_size(self, mobject: "Mobject", dim, **kwargs):
        """Match the specified dimension with the dimension of another :class:`~.Mobject`."""
        return self.rescale_to_fit(mobject.length_over_dim(dim), dim, **kwargs)

    def match_width(self, mobject: "Mobject", **kwargs):
        """Match the width with the width of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 0, **kwargs)

    def match_height(self, mobject: "Mobject", **kwargs):
        """Match the height with the height of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 1, **kwargs)

    def match_depth(self, mobject: "Mobject", **kwargs):
        """Match the depth with the depth of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 2, **kwargs)

    def match_coord(self, mobject: "Mobject", dim, direction=ORIGIN):
        """Match the coordinates with the coordinates of another :class:`~.Mobject`."""
        return self.set_coord(
            mobject.get_coord(dim, direction),
            dim=dim,
            direction=direction,
        )

    def match_x(self, mobject: "Mobject", direction=ORIGIN):
        """Match x coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 0, direction)

    def match_y(self, mobject: "Mobject", direction=ORIGIN):
        """Match y coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 1, direction)

    def match_z(self, mobject: "Mobject", direction=ORIGIN):
        """Match z coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 2, direction)

    def align_to(
        self,
        mobject_or_point: Union["Mobject", np.ndarray, List],
        direction=ORIGIN,
        alignment_vect=UP,
    ):
        """Aligns mobject to another :class:`~.Mobject` in a certain direction.

        Examples:
        mob1.align_to(mob2, UP) moves mob1 vertically so that its
        top edge lines ups with mob2's top edge.

        mob1.align_to(mob2, alignment_vect = RIGHT) moves mob1
        horizontally so that it's center is directly above/below
        the center of mob2
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

    def get_group_class(self):
        return Group

    def split(self):
        result = [self] if len(self.points) > 0 else []
        return result + self.submobjects

    def get_family(self, recurse=True):
        sub_families = list(map(Mobject.get_family, self.submobjects))
        all_mobjects = [self] + list(it.chain(*sub_families))
        return remove_list_redundancies(all_mobjects)

    def family_members_with_points(self):
        return [m for m in self.get_family() if m.get_num_points() > 0]

    def arrange(
        self,
        direction: Sequence[float] = RIGHT,
        buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        center=True,
        **kwargs,
    ):
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
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        buff: Union[float, Tuple[float, float]] = MED_SMALL_BUFF,
        cell_alignment: np.ndarray = ORIGIN,
        row_alignments: Optional[str] = None,  # "ucd"
        col_alignments: Optional[str] = None,  # "lcr"
        row_heights: Optional[Iterable[Optional[float]]] = None,
        col_widths: Optional[Iterable[Optional[float]]] = None,
        flow_order: str = "rd",
        **kwargs,
    ) -> "Mobject":
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
        Mobject
            The mobject.

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
                    #Add some numbered boxes:
                    np.random.seed(3)
                    boxes = VGroup(*[
                        Rectangle(WHITE, np.random.random()+.5, np.random.random()+.5).add(Text(str(i+1)).scale(0.5))
                        for i in range(22)
                    ])
                    self.add(boxes)

                    boxes.arrange_in_grid(
                        buff=(0.25,0.5),
                        col_alignments="lccccr",
                        row_alignments="uccd",
                        col_widths=[2, *[None]*4, 2],
                        flow_order="dr"
                    )


        """
        from .geometry import Line

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
            cols = ceil(np.sqrt(len(mobs)))
            # make the grid as close to quadratic as possible.
            # choosing cols first can results in cols>rows.
            # This is favored over rows>cols since in general
            # the sceene is wider than high.
        if rows is None:
            rows = ceil(len(mobs) / cols)
        if cols is None:
            cols = ceil(len(mobs) / rows)
        if rows * cols < len(mobs):
            raise ValueError("Too few rows and columns to fit all submobjetcs.")
        # rows and cols are now finally valid.

        if isinstance(buff, tuple):
            buff_x = buff[0]
            buff_y = buff[1]
        else:
            buff_x = buff_y = buff

        # Initialize alignments correctly
        def init_alignments(alignments, num, mapping, name, dir):
            if alignments is None:
                # Use cell_alignment as fallback
                return [cell_alignment * dir] * num
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

    def sort(self, point_to_num_func=lambda p: p[0], submob_func=None):
        """Sorts the list of :attr:`submobjects` by a function defined by ``submob_func``."""
        if submob_func is None:

            def submob_func(m):
                return point_to_num_func(m.get_center())

        self.submobjects.sort(key=submob_func)
        return self

    def shuffle(self, recursive=False):
        """Shuffles the list of :attr:`submobjects`."""
        if recursive:
            for submob in self.submobjects:
                submob.shuffle(recursive=True)
        random.shuffle(self.submobjects)

    def invert(self, recursive=False):
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
        list.reverse(self.submobjects)

    # Just here to keep from breaking old scenes.
    def arrange_submobjects(self, *args, **kwargs):
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

    def sort_submobjects(self, *args, **kwargs):
        """Sort the :attr:`submobjects`"""
        return self.sort(*args, **kwargs)

    def shuffle_submobjects(self, *args, **kwargs):
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
    def align_data(self, mobject: "Mobject"):
        self.null_point_align(mobject)
        self.align_submobjects(mobject)
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

    def align_points(self, mobject):
        count1 = self.get_num_points()
        count2 = mobject.get_num_points()
        if count1 < count2:
            self.align_points_with_larger(mobject)
        elif count2 < count1:
            mobject.align_points_with_larger(self)
        return self

    def align_points_with_larger(self, larger_mobject):
        raise NotImplementedError("Please override in a child class.")

    def align_submobjects(self, mobject):
        mob1 = self
        mob2 = mobject
        n1 = len(mob1.submobjects)
        n2 = len(mob2.submobjects)
        mob1.add_n_more_submobjects(max(0, n2 - n1))
        mob2.add_n_more_submobjects(max(0, n1 - n2))
        return self

    def null_point_align(self, mobject: "Mobject") -> "Mobject":
        """If a :class:`~.Mobject` with points is being aligned to
        one without, treat both as groups, and push
        the one with points into its own submobjects
        list.
        """
        for m1, m2 in (self, mobject), (mobject, self):
            if m1.has_no_points() and m2.has_points():
                m2.push_self_into_submobjects()
        return self

    def push_self_into_submobjects(self):
        copy = self.copy()
        copy.submobjects = []
        self.reset_points()
        self.add(copy)
        return self

    def add_n_more_submobjects(self, n):
        if n == 0:
            return

        curr = len(self.submobjects)
        if curr == 0:
            # If empty, simply add n point mobjects
            self.submobjects = [self.get_point_mobject() for k in range(n)]
            return

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

    def repeat_submobject(self, submob):
        return submob.copy()

    def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
        """Turns this :class:`~.Mobject` into an interpolation between ``mobject1``
        and ``mobject2``.

        Examples
        --------

        .. manim:: DotInterpolation
            :save_last_frame:

            class DotInterpolation(Scene):
                def construct(self):
                    dotL = Dot(color=DARK_GREY)
                    dotL.shift(2 * RIGHT)
                    dotR = Dot(color=WHITE)
                    dotR.shift(2 * LEFT)

                    dotMiddle = VMobject().interpolate(dotL, dotR, alpha=0.3)

                    self.add(dotL, dotR, dotMiddle)
        """
        self.points = path_func(mobject1.points, mobject2.points, alpha)
        self.interpolate_color(mobject1, mobject2, alpha)
        return self

    def interpolate_color(self, mobject1, mobject2, alpha):
        raise NotImplementedError("Please override in a child class.")

    def become(self, mobject: "Mobject", copy_submobjects: bool = True):
        """Edit points, colors and submobjects to be identical
        to another :class:`~.Mobject`

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
        """
        self.align_data(mobject)
        for sm1, sm2 in zip(self.get_family(), mobject.get_family()):
            sm1.points = np.array(sm2.points)
            sm1.interpolate_color(sm1, sm2, 1)
        return self

    def match_points(self, mobject: "Mobject", copy_submobjects: bool = True):
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
        self.align_data(mobject)
        for sm1, sm2 in zip(self.get_family(), mobject.get_family()):
            sm1.points = np.array(sm2.points)
        return self

    # Errors
    def throw_error_if_no_points(self):
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
    ) -> "VMobject":
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

    def set_z_index_by_z_coordinate(self):
        """Sets the :class:`~.Mobject`'s z coordinate to the value of :attr:`z_index`.

        Returns
        -------
        :class:`Mobject`
            The Mobject itself, after :attr:`z_index` is set. (Returns `self`.)
        """
        z_coord = self.get_center()[-1]
        self.set_z_index(z_coord)
        return self


class Group(Mobject, metaclass=ConvertToOpenGL):
    """Groups together multiple :class:`Mobjects <.Mobject>`."""

    def __init__(self, *mobjects, **kwargs):
        super().__init__(**kwargs)
        self.add(*mobjects)


class _AnimationBuilder:
    def __init__(self, mobject):
        self.mobject = mobject
        self.mobject.generate_target()

        self.overridden_animation = None
        self.is_chaining = False
        self.methods = []

        # Whether animation args can be passed
        self.cannot_pass_args = False
        self.anim_args = {}

    def __call__(self, **kwargs):
        if self.cannot_pass_args:
            raise ValueError(
                "Animation arguments must be passed before accessing methods and can only be passed once",
            )

        self.anim_args = kwargs
        self.cannot_pass_args = True

        return self

    def __getattr__(self, method_name):
        method = getattr(self.mobject.target, method_name)
        self.methods.append(method)
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
                method(*method_args, **method_kwargs)
            return self

        self.is_chaining = True
        self.cannot_pass_args = True

        return update_target

    def build(self):
        from ..animation.transform import _MethodAnimation

        if self.overridden_animation:
            anim = self.overridden_animation
        else:
            anim = _MethodAnimation(self.mobject, self.methods)

        for attr, value in self.anim_args.items():
            setattr(anim, attr, value)

        return anim


def override_animate(method):
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
