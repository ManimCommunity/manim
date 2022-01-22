from __future__ import annotations

import copy
from functools import partialmethod
from typing import Callable, Dict, Iterable, Optional, Sequence, Type, TypeVar

import numpy as np

from manim.constants import ORIGIN, OUT, TAU, UP
from manim.utils.space_ops import rotation_matrix, rotation_matrix_transpose

from ..animation.animation_utils import _AnimationBuilder
from ..utils.exceptions import MultiAnimationOverrideException

T = TypeVar("T", bound="MobjectBase")


class MobjectBase:
    """Base class for mobjects.

    This class is agnostic of the rendering backend.

    TODO: documentation.
    """

    animation_overrides = {}

    ### Mobject initialization ###

    def __init__(self, name=None):
        """Initializes a mobject.

        TODO: documentation.
        """
        if name is None:
            name = self.__class__.__name__
        self.name = name

        self._bounding_box = None

        self.parents = []
        self._submobjects = []
        self._points = []

    ### Generic class methods ###

    def __repr__(self):
        """Returns a string representation of this mobject.

        Returns the class name by default.
        """
        return self.name

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Code that is executed upon (sub)class initialization.

        Required for the ``set_default`` mechanism.
        """
        super().__init_subclass__(**kwargs)
        cls.animation_overrides: dict[
            type[Animation],
            Callable[[MobjectBase], Animation],
        ] = {}
        cls._add_intrinsic_animation_overrides()
        cls._original__init__ = cls.__init__

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

    @classmethod
    def _add_intrinsic_animation_overrides(cls):
        """Initializes animation overrides marked with the :func:`~.override_animation`
        decorator.
        """
        for method_name in dir(cls):
            # Ignore dunder methods
            if method_name.startswith("__"):
                continue

            try:
                method = getattr(cls, method_name)
            except AttributeError:
                continue
            if hasattr(method, "_override_animation"):
                animation_class = method._override_animation
                cls.add_animation_override(animation_class, method)

    @classmethod
    def animation_override_for(
        cls,
        animation_class: type[Animation],
    ) -> Optional[Callable[[Mobject, ...], Animation]]:
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
    def add_animation_override(
        cls,
        animation_class: type[Animation],
        override_func: Callable[[Mobject, ...], Animation],
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

    def __sub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __iadd__(self, other):
        raise NotImplementedError

    ### Animation related methods ###

    def set(self, **kwargs):
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

    @property
    def animate(self):
        """Used to animate the application of any method of :code:`self`.

        Any method called on :code:`animate` is converted to an animation of applying
        that method on the mobject itself.

        For example, :code:`square.set_fill(WHITE)` sets the fill color of a square,
        while :code:`sqaure.animate.set_fill(WHITE)` animates this action.

        Multiple methods can be put in a single animation once via chaining:

        ::

            self.play(my_mobject.animate.shift(RIGHT).rotate(PI))

        .. warning::

            Passing multiple animations for the same :class:`Mobject` in one
            call to :meth:`~.Scene.play` is discouraged and will most likely
            not work properly. Instead of writing an animation like

            ::

                self.play(my_mobject.animate.shift(RIGHT), my_mobject.animate.rotate(PI))

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

    ### Mobject family related methods ###

    def get_group_class(self):
        raise NotImplementedError

    @property
    def submobjects(self):
        return self._submobjects

    @submobjects.setter
    def submobjects(self, submobject_list):
        self._submobjects = submobject_list
        self.invalidate_bounding_box()

    def __getitem__(self, value):
        if isinstance(value, slice):
            Group = self.get_group_class()
            return Group(*self.submobjects.__getitem__(value))
        return self.submobjects.__getitem__(value)

    def __iter__(self):
        return iter(self.submobjects)

    def __len__(self):
        return len(self.submobjects)

    def has_submobjects(self):
        return len(self.submobjects) > 0

    def get_family(self):
        """Return the collection of all mobjects this mobject consists
        of. Basically self plus (recursively) all submobjects."""
        pass

    def copy(self: T) -> T:
        """Creates a copy of this mobject.

        The created copy is almost a full deep copy, with the exception
        of the mobject's parents: the copied mobject is considered to
        be a separate and new mobject; in particular the copy is not considered
        to be included in the same parent mobjects as the original mobject.
        """
        parents = self.parents
        self.parents = []
        result = copy.deepcopy(self)
        self.parents = parents
        return result

    ### Mobject points and related methods ###

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, pts):
        self._points = pts
        self.invalidate_bounding_box()

    def has_points(self):
        """Checks whether this mobject has any points set."""
        return len(self.points) > 0

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            self.update_bounding_box()
        return self._bounding_box

    def update_bounding_box(self):
        if self.has_submobjects():
            points = np.concatenate(
                [
                    self.points,
                    *(
                        mob.bounding_box
                        for mob in self.get_family()[1:]
                        if mob.has_points()
                    ),
                ]
            )

        else:
            points = self.points

        if len(points) == 0:
            points = np.zeros((1, 3))

        min_point = points.min(0)
        max_point = points.max(0)
        mid_point = (min_point + max_point) / 2
        self._bounding_box = np.array([min_point, mid_point, max_point])

    def invalidate_bounding_box(self, recurse_down=False, recurse_up=True):
        for family_member in self.get_family(recurse=recurse_down):
            family_member._bounding_box = None
        if recurse_up:
            for parent in self.parents:
                parent.invalidate_bounding_box(recurse_up=True)

    def get_bounding_box_point(self, direction):
        """Returns the special boundary point of the bounding box
        along the specified direction.

        The (three-dimensional) bounding box has 27 special points:
        8 corners, 12 edge midpoints, 6 face midpoints, and one center.
        This method determines the point along the specified direction
        by considering the sign of the coordinates of the direction
        vector.
        """
        indices = (np.sign(direction) + 1).astype(int)
        return np.array([self.bounding_box[indices[i]][i] for i in range(3)])

    def get_critical_point(self, direction):
        """Alias of :meth:`get_bounding_box_point`."""
        return self.get_bounding_box_point(direction=direction)

    def get_corner(self, direction):
        """Alias of :meth:`get_bounding_box_point`."""
        return self.get_bounding_box_point(direction=direction)

    def get_edge_center(self, direction):
        """Alias of :meth:`get_bounding_box_point`."""
        return self.get_bounding_box_point(direction=direction)

    def get_center(self):
        """Returns the coordinates of the center of the bounding
        box of this mobject.
        """
        return self.bounding_box[1].copy()

    def get_pieces(self, n_pieces):
        template = self.copy()
        template.submobjects = []
        alphas = np.linspace(0, 1, n_pieces + 1)
        Group = self.get_group_class()
        return Group(
            *(
                template.copy().pointwise_become_partial(self, a1, a2)
                for a1, a2 in zip(alphas[:-1], alphas[1:])
            )
        )

    def length_over_dim(self, dimension):
        """Return the length of the mobject over the given dimension.

        Parameters
        ----------
        dimension
            The dimension as the index of the points, i.e., 0 for
            the :math:`x`-axis, 1 for the :math:`y`-axis, 2 for the
            :math:`z`-axis.
        """
        return (self.bounding_box[2] - self.bounding_box[0])[dimension]

    @property
    def width(self):
        """The width of the mobject.

        Returns
        -------
        :class:`float`

        See also
        --------
        :meth:`length_over_dim`

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

        """
        return self.length_over_dim(0)

    @width.setter
    def width(self, value):
        self.rescale_to_fit(value, 0, stretch=False)

    @property
    def height(self):
        """The height of the mobject.

        Returns
        -------
        :class:`float`

        See also
        --------
        :meth:`length_over_dim`

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

        """
        return self.length_over_dim(1)

    @height.setter
    def height(self, value):
        self.rescale_to_fit(value, 1, stretch=False)

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
        return self.length_over_dim(2)

    @depth.setter
    def depth(self, value):
        self.rescale_to_fit(value, 2, stretch=False)

    ### Mobject transformations ###

    def apply_points_function(
        self,
        func,
        about_point=None,
        about_edge=ORIGIN,
        works_on_bounding_box=False,
    ):
        """Apply a function to the points of this mobject.

        Parameters
        ----------

        func
            The function being applied to this mobject's points.
        about_point
            Specifies where the origin of the (shifted) coordinate
            system used to compute the images of the points under the
            function should be located. If ``None`` (the default), then
            the origin is determined via the ``about_edge`` keyword
            argument.
        about_edge
            If ``about_point`` is ``None``, this parameter allows to
            determine the origin of the (shifted) coordinate system
            of the transformation relative to the mobject by returning
            the corresponding bounding box point. Defaults to ``ORIGIN``,
            which results in the mobject's center. If set to ``None``,
            the coordinate system is not shifted before applying the
            transformation.
        works_on_bounding_box
            If set to ``True``, the function will also be applied to the
            currently cached bounding box points. The bounding box then
            does not need to be invalidated and recomputed.
        """
        if about_point is None:
            if about_edge is not None:
                about_point = self.get_bounding_box_point(about_edge)
            else:
                about_point = ORIGIN

        for mob in self.get_family():
            if not mob.has_points():
                if works_on_bounding_box and mob._bounding_box is not None:
                    mob._bounding_box[:] = (
                        func(mob._bounding_box - about_point) + about_point
                    )
                continue
            point_lists = (
                [mob._bounding_box, mob._points]
                if works_on_bounding_box and mob._bounding_box is not None
                else [mob.points]
            )
            for points in point_lists:
                points[:] = func(points - about_point) + about_point
            # TODO: check that the bounding box is still valid, i.e.,
            # first point has to be component-wise <= last point. if not,
            # reverse entries.

        if works_on_bounding_box:
            for parent in self.parents:
                parent.invalidate_bounding_box()

        return self

    def rotate(
        self,
        angle,
        axis=OUT,
        about_point: Sequence[float] | None = None,
        **kwargs,
    ):
        """Rotates this mobject about a certain point.

        Parameters
        ----------
        angle
            The rotation angle.
        axis
            The direction of the rotation axis. Defaults to ``OUT``,
            which corresponds to rotations in the :math:`(x,y)`-plane.
        about_point
            The rotation center.
        """
        rot_matrix = rotation_matrix(angle, axis)
        rot_matrix_T = rotation_matrix_transpose(angle, axis)
        self.apply_points_function(
            lambda points: np.dot(points, rot_matrix.T),
            about_point=about_point,
            **kwargs,
        )
        return self

    def rotate_about_origin(self, angle, axis=OUT):
        """Rotates this mobject about the origin of the scene.

        .. seealso::

            :meth:`rotate`
        """
        return self.rotate(angle=angle, axis=axis, about_point=ORIGIN)

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

    def shift(self, vector):
        """Shifts this mobject along a given vector.

        Parameters
        ----------

        vector
            The vector along which the mobject is shifted.

        """
        self.apply_points_function(
            lambda points: points + vector,
            about_edge=None,
            works_on_bounding_box=True,
        )
        return self

    def stretch(self, factor, dimension, **kwargs):
        """Stretches this mobject along the specified dimension.

        Parameters
        ----------
        factor
            The stretch factor.
        dimension
            The dimension along which the mobject is stretched. This
            is the index of the point arrays, i.e., 0 corresponds
            to stretching along the :math:`x`-axis, etc.
        """

        def func(points):
            points[:, dimension] *= factor
            return points

        self.apply_points_function(func, works_on_bounding_box=True, **kwargs)
        return self

    def scale(
        self,
        scale_factor: float,
        about_point: Sequence[float] | None = None,
        about_edge: Sequence[float] = ORIGIN,
        **kwargs,
    ):
        r"""Scale the size by a factor.

        Default behavior is to scale about the center of the mobject.
        The argument about_edge can be a vector, indicating which side of
        the mobject to scale about, e.g., mob.scale(about_edge = RIGHT)
        scales about mob.get_right().

        Otherwise, if about_point is given a value, scaling is done with
        respect to that point.

        Parameters
        ----------
        scale_factor
            The scaling factor :math:`\alpha`. If :math:`0 < |\alpha| < 1`, the mobject
            will shrink, and for :math:`|\alpha| > 1` it will grow. Furthermore,
            if :math:`\alpha < 0`, the mobject is also flipped.
        kwargs
            Additional keyword arguments passed to
            :meth:`apply_points_function_about_point`.

        Returns
        -------
        MobjectBase
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
        self.apply_points_function(
            lambda points: scale_factor * points,
            about_point=about_point,
            about_edge=about_edge,
            works_on_bounding_box=True,
            **kwargs,
        )
        return self

    def rescale_to_fit(self, value, dimension, stretch=False, **kwargs):
        """Rescale this mobject to have the specified measurements across
        the given dimension.

        Parameters
        ----------
        value
            The target measurement after rescaling.
        dimension
            The dimension across which the mobjects measurement is adjusted.
        stretch
            If set to ``True``, the mobject is only stretched along the given
            dimension. Otherwise (the default behavior), the mobject is scaled
            along all dimensions.
        kwargs
            Further keyword arguments passed to :meth:`scale` or :meth:`stretch`.
        """
        old_value = self.length_over_dim(dimension)
        if old_value == 0:
            # TODO: logging that mobject could not be stretched / scaled?
            return self
        if stretch:
            self.stretch(value / old_value, dimension=dimension, **kwargs)
        else:
            self.scale(value / old_value, **kwargs)
        return self
