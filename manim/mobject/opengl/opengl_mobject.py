from __future__ import annotations

import copy
import inspect
import itertools as it
import random
import sys
import types
from collections.abc import Iterable, Iterator, Sequence
from functools import partialmethod, wraps
from math import ceil
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import moderngl
import numpy as np

from manim import config, logger
from manim.constants import *
from manim.renderer.shader_wrapper import get_colormap_code
from manim.utils.bezier import integer_interpolate, interpolate
from manim.utils.color import (
    WHITE,
    ManimColor,
    ParsableManimColor,
    color_gradient,
    color_to_rgb,
    rgb_to_hex,
)
from manim.utils.config_ops import _Data, _Uniforms

# from ..utils.iterables import batch_by_property
from manim.utils.iterables import (
    batch_by_property,
    list_update,
    listify,
    make_even,
    resize_array,
    resize_preserving_order,
    resize_with_interpolation,
    uniq_chain,
)
from manim.utils.paths import straight_path
from manim.utils.space_ops import (
    angle_between_vectors,
    normalize,
    rotation_matrix_transpose,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import Self, TypeAlias

    from manim.renderer.shader_wrapper import ShaderWrapper
    from manim.typing import (
        ManimFloat,
        MappingFunction,
        MatrixMN,
        MultiMappingFunction,
        PathFuncType,
        Point3D,
        Point3D_Array,
        Point3DLike,
        Point3DLike_Array,
        Vector3D,
    )

    TimeBasedUpdater: TypeAlias = Callable[["Mobject", float], object]
    NonTimeBasedUpdater: TypeAlias = Callable[["Mobject"], object]
    Updater: TypeAlias = NonTimeBasedUpdater | TimeBasedUpdater

    T = TypeVar("T")


def affects_shader_info_id(
    func: Callable[[OpenGLMobject], OpenGLMobject],
) -> Callable[[OpenGLMobject], OpenGLMobject]:
    @wraps(func)
    def wrapper(self: OpenGLMobject) -> OpenGLMobject:
        for mob in self.get_family():
            func(mob)
            mob.refresh_shader_wrapper_id()
        return self

    return wrapper


__all__ = ["OpenGLMobject", "OpenGLGroup", "OpenGLPoint", "_AnimationBuilder"]


class OpenGLMobject:
    """Mathematical Object: base class for objects that can be displayed on screen.

    Attributes
    ----------
    submobjects : List[:class:`OpenGLMobject`]
        The contained objects.
    points : :class:`numpy.ndarray`
        The points of the objects.

        .. seealso::

            :class:`~.OpenGLVMobject`

    """

    shader_dtype = [
        ("point", np.float32, (3,)),
    ]
    shader_folder = ""

    # _Data and _Uniforms are set as class variables to tell manim how to handle setting/getting these attributes later.
    points = _Data()
    bounding_box = _Data()
    rgbas = _Data()

    is_fixed_in_frame = _Uniforms()
    is_fixed_orientation = _Uniforms()
    fixed_orientation_center = _Uniforms()  # for fixed orientation reference
    gloss = _Uniforms()
    shadow = _Uniforms()

    def __init__(
        self,
        color: ParsableManimColor | Iterable[ParsableManimColor] = WHITE,
        opacity: float = 1,
        dim: int = 3,  # TODO, get rid of this
        # Lighting parameters
        # Positive gloss up to 1 makes it reflect the light.
        gloss: float = 0.0,
        # Positive shadow up to 1 makes a side opposite the light darker
        shadow: float = 0.0,
        # For shaders
        render_primitive: int = moderngl.TRIANGLES,
        texture_paths: dict[str, str] | None = None,
        depth_test: bool = False,
        # If true, the mobject will not get rotated according to camera position
        is_fixed_in_frame: bool = False,
        is_fixed_orientation: bool = False,
        # Must match in attributes of vert shader
        # Event listener
        listen_to_events: bool = False,
        model_matrix: MatrixMN | None = None,
        should_render: bool = True,
        name: str | None = None,
        **kwargs,
    ):
        self.name = self.__class__.__name__ if name is None else name
        # getattr in case data/uniforms are already defined in parent classes.
        self.data = getattr(self, "data", {})
        self.uniforms = getattr(self, "uniforms", {})

        self.opacity = opacity
        self.dim = dim  # TODO, get rid of this
        # Lighting parameters
        # Positive gloss up to 1 makes it reflect the light.
        self.gloss = gloss
        # Positive shadow up to 1 makes a side opposite the light darker
        self.shadow = shadow
        # For shaders
        self.render_primitive = render_primitive
        self.texture_paths = texture_paths
        self.depth_test = depth_test
        # If true, the mobject will not get rotated according to camera position
        self.is_fixed_in_frame = float(is_fixed_in_frame)
        self.is_fixed_orientation = float(is_fixed_orientation)
        self.fixed_orientation_center = (0, 0, 0)
        # Must match in attributes of vert shader
        # Event listener
        self.listen_to_events = listen_to_events

        self._submobjects = []
        self.parents = []
        self.parent = None
        self.family = [self]
        self.locked_data_keys = set()
        self.needs_new_bounding_box = True
        if model_matrix is None:
            self.model_matrix = np.eye(4)
        else:
            self.model_matrix = model_matrix

        self.init_data()
        self.init_updaters()
        # self.init_event_listners()
        self.init_points()
        self.color = ManimColor.parse(color)
        self.init_colors()

        self.shader_indices = None

        if self.depth_test:
            self.apply_depth_test()

        self.should_render = should_render

    def _assert_valid_submobjects(self, submobjects: Iterable[OpenGLMobject]) -> Self:
        """Check that all submobjects are actually instances of
        :class:`OpenGLMobject`, and that none of them is
        ``self`` (an :class:`OpenGLMobject` cannot contain itself).

        This is an auxiliary function called when adding OpenGLMobjects to the
        :attr:`submobjects` list.

        This function is intended to be overridden by subclasses such as
        :class:`OpenGLVMobject`, which should assert that only other
        OpenGLVMobjects may be added into it.

        Parameters
        ----------
        submobjects
            The list containing values to validate.

        Returns
        -------
        :class:`OpenGLMobject`
            The OpenGLMobject itself.

        Raises
        ------
        TypeError
            If any of the values in `submobjects` is not an
            :class:`OpenGLMobject`.
        ValueError
            If there was an attempt to add an :class:`OpenGLMobject` as its own
            submobject.
        """
        return self._assert_valid_submobjects_internal(submobjects, OpenGLMobject)

    def _assert_valid_submobjects_internal(
        self, submobjects: Iterable[OpenGLMobject], mob_class: type[OpenGLMobject]
    ) -> Self:
        for i, submob in enumerate(submobjects):
            if not isinstance(submob, mob_class):
                error_message = (
                    f"Only values of type {mob_class.__name__} can be added "
                    f"as submobjects of {type(self).__name__}, but the value "
                    f"{submob} (at index {i}) is of type "
                    f"{type(submob).__name__}."
                )
                # Intended for subclasses such as OpenGLVMobject, which
                # cannot have regular OpenGLMobjects as submobjects
                if isinstance(submob, OpenGLMobject):
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
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._original__init__ = cls.__init__

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self.name)

    def __sub__(self, other):
        return NotImplemented

    def __isub__(self, other):
        return NotImplemented

    def __add__(self, mobject):
        return NotImplemented

    def __iadd__(self, mobject):
        return NotImplemented

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

    def init_data(self) -> None:
        """Initializes the ``points``, ``bounding_box`` and ``rgbas`` attributes and groups them into self.data.
        Subclasses can inherit and overwrite this method to extend `self.data`.
        """
        self.points = np.zeros((0, 3))
        self.bounding_box = np.zeros((3, 3))
        self.rgbas = np.zeros((1, 4))

    def init_colors(self) -> object:
        """Initializes the colors.

        Gets called upon creation
        """
        self.set_color(self.color, self.opacity)

    def init_points(self) -> object:
        """Initializes :attr:`points` and therefore the shape.

        Gets called upon creation. This is an empty method that can be implemented by
        subclasses.
        """
        # Typically implemented in subclass, unless purposefully left blank
        pass

    def set(self, **kwargs) -> Self:
        """Sets attributes.

        Mainly to be used along with :attr:`animate` to
        animate setting attributes.

        Examples
        --------
        ::

            >>> mob = OpenGLMobject()
            >>> mob.set(foo=0)
            OpenGLMobject
            >>> mob.foo
            0

        Parameters
        ----------
        **kwargs
            The attributes and corresponding values to set.

        Returns
        -------
        :class:`OpenGLMobject`
            ``self``


        """
        for attr, value in kwargs.items():
            setattr(self, attr, value)

        return self

    def set_data(self, data: dict[str, Any]) -> Self:
        for key in data:
            self.data[key] = data[key].copy()
        return self

    def set_uniforms(self, uniforms: dict[str, Any]) -> Self:
        for key in uniforms:
            self.uniforms[key] = uniforms[key]  # Copy?
        return self

    @property
    def animate(self) -> _AnimationBuilder | Self:
        """Used to animate the application of a method.

        .. warning::

            Passing multiple animations for the same :class:`OpenGLMobject` in one
            call to :meth:`~.Scene.play` is discouraged and will most likely
            not work properly. Instead of writing an animation like

            ::

                self.play(
                    my_mobject.animate.shift(RIGHT), my_mobject.animate.rotate(PI)
                )

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
             will interpolate the :class:`~.OpenGLMobject` between its points prior to
             ``.animate`` and its points after applying ``.animate`` to it. This may
             result in unexpected behavior when attempting to interpolate along paths,
             or rotations.
             If you want animations to consider the points between, consider using
             :class:`~.ValueTracker` with updaters instead.

        """
        return _AnimationBuilder(self)

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

    # Only these methods should directly affect points
    @width.setter
    def width(self, value: float) -> None:
        self.rescale_to_fit(value, 0, stretch=False)

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
    def height(self, value: float) -> None:
        self.rescale_to_fit(value, 1, stretch=False)

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
    def depth(self, value: float) -> None:
        self.rescale_to_fit(value, 2, stretch=False)

    def resize_points(self, new_length, resize_func=resize_array):
        if new_length != len(self.points):
            self.points = resize_func(self.points, new_length)
        self.refresh_bounding_box()
        return self

    def set_points(self, points: Point3DLike_Array) -> Self:
        if len(points) == len(self.points):
            self.points[:] = points
        elif isinstance(points, np.ndarray):
            self.points = points.copy()
        else:
            self.points = np.array(points)
        self.refresh_bounding_box()
        return self

    def apply_over_attr_arrays(
        self, func: Callable[[npt.NDArray[T]], npt.NDArray[T]]
    ) -> Self:
        # TODO: OpenGLMobject.get_array_attrs() doesn't even exist!
        for attr in self.get_array_attrs():
            setattr(self, attr, func(getattr(self, attr)))
        return self

    def append_points(self, new_points: Point3DLike_Array) -> Self:
        self.points = np.vstack([self.points, new_points])
        self.refresh_bounding_box()
        return self

    def reverse_points(self) -> Self:
        for mob in self.get_family():
            for key in mob.data:
                mob.data[key] = mob.data[key][::-1]
        return self

    def get_midpoint(self) -> Point3D:
        """Get coordinates of the middle of the path that forms the  :class:`~.OpenGLMobject`.

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

    # TODO: name is inconsistent with Mobject.apply_points_function_about_point()
    def apply_points_function(
        self,
        func: MultiMappingFunction,
        about_point: Point3DLike | None = None,
        about_edge: Vector3D | None = ORIGIN,
        works_on_bounding_box: bool = False,
    ) -> Self:
        if about_point is None and about_edge is not None:
            about_point = self.get_bounding_box_point(about_edge)

        for mob in self.get_family():
            arrs = []
            if mob.has_points():
                arrs.append(mob.points)
            if works_on_bounding_box:
                arrs.append(mob.get_bounding_box())

            for arr in arrs:
                if about_point is None:
                    arr[:] = func(arr)
                else:
                    arr[:] = func(arr - about_point) + about_point

        if not works_on_bounding_box:
            self.refresh_bounding_box(recurse_down=True)
        else:
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    # Others related to points

    def match_points(self, mobject: OpenGLMobject) -> Self:
        """Edit points, positions, and submobjects to be identical
        to another :class:`~.OpenGLMobject`, while keeping the style unchanged.

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
        self.set_points(mobject.points)
        return self

    def clear_points(self) -> Self:
        self.points = np.empty((0, 3))
        return self

    def get_num_points(self) -> int:
        return len(self.points)

    def get_all_points(self) -> Point3D_Array:
        if self.submobjects:
            return np.vstack([sm.points for sm in self.get_family()])
        else:
            return self.points

    def has_points(self) -> bool:
        return self.get_num_points() > 0

    def get_bounding_box(self) -> npt.NDArray[float]:
        if self.needs_new_bounding_box:
            self.bounding_box = self.compute_bounding_box()
            self.needs_new_bounding_box = False
        return self.bounding_box

    def compute_bounding_box(self) -> npt.NDArray[float]:
        all_points = np.vstack(
            [
                self.points,
                *(
                    mob.get_bounding_box()
                    for mob in self.get_family()[1:]
                    if mob.has_points()
                ),
            ],
        )
        if len(all_points) == 0:
            return np.zeros((3, self.dim))
        else:
            # Lower left and upper right corners
            mins = all_points.min(0)
            maxs = all_points.max(0)
            mids = (mins + maxs) / 2
            return np.array([mins, mids, maxs])

    def refresh_bounding_box(
        self, recurse_down: bool = False, recurse_up: bool = True
    ) -> Self:
        for mob in self.get_family(recurse_down):
            mob.needs_new_bounding_box = True
        if recurse_up:
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    def is_point_touching(
        self, point: Point3DLike, buff: float = MED_SMALL_BUFF
    ) -> bool:
        bb = self.get_bounding_box()
        mins = bb[0] - buff
        maxs = bb[2] + buff
        return (point >= mins).all() and (point <= maxs).all()

    # Family matters

    def __getitem__(self, value: int | slice) -> OpenGLMobject:
        if isinstance(value, slice):
            GroupClass = self.get_group_class()
            return GroupClass(*self.split().__getitem__(value))
        return self.split().__getitem__(value)

    def __iter__(self) -> Iterator[OpenGLMobject]:
        return iter(self.split())

    def __len__(self) -> int:
        return len(self.split())

    def split(self) -> Sequence[OpenGLMobject]:
        return self.submobjects

    def assemble_family(self) -> Self:
        sub_families = (sm.get_family() for sm in self.submobjects)
        self.family = [self, *uniq_chain(*sub_families)]
        self.refresh_has_updater_status()
        self.refresh_bounding_box()
        for parent in self.parents:
            parent.assemble_family()
        return self

    def get_family(self, recurse: bool = True) -> Sequence[OpenGLMobject]:
        if recurse and hasattr(self, "family"):
            return self.family
        else:
            return [self]

    def family_members_with_points(self) -> Sequence[OpenGLMobject]:
        return [m for m in self.get_family() if m.has_points()]

    def add(self, *mobjects: OpenGLMobject, update_parent: bool = False) -> Self:
        """Add mobjects as submobjects.

        The mobjects are added to :attr:`submobjects`.

        Subclasses of mobject may implement ``+`` and ``+=`` dunder methods.

        Parameters
        ----------
        mobjects
            The mobjects to add.

        Returns
        -------
        :class:`OpenGLMobject`
            ``self``

        Raises
        ------
        :class:`ValueError`
            When a mobject tries to add itself.
        :class:`TypeError`
            When trying to add an object that is not an instance of :class:`OpenGLMobject`.


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

            >>> outer = OpenGLMobject()
            >>> inner = OpenGLMobject()
            >>> outer = outer.add(inner)

        Duplicates are not added again::

            >>> outer = outer.add(inner)
            >>> len(outer.submobjects)
            1

        Only OpenGLMobjects can be added::

            >>> outer.add(3)
            Traceback (most recent call last):
            ...
            TypeError: Only values of type OpenGLMobject can be added as submobjects of OpenGLMobject, but the value 3 (at index 0) is of type int.

        Adding an object to itself raises an error::

            >>> outer.add(outer)
            Traceback (most recent call last):
            ...
            ValueError: Cannot add OpenGLMobject as a submobject of itself (at index 0).

        """
        if update_parent:
            assert len(mobjects) == 1, "Can't set multiple parents."
            mobjects[0].parent = self

        self._assert_valid_submobjects(mobjects)

        if any(mobjects.count(elem) > 1 for elem in mobjects):
            logger.warning(
                "Attempted adding some Mobject as a child more than once, "
                "this is not possible. Repetitions are ignored.",
            )
        for mobject in mobjects:
            if mobject not in self.submobjects:
                self.submobjects.append(mobject)
            if self not in mobject.parents:
                mobject.parents.append(self)
        self.assemble_family()
        return self

    def insert(
        self, index: int, mobject: OpenGLMobject, update_parent: bool = False
    ) -> Self:
        """Inserts a mobject at a specific position into self.submobjects

        Effectively just calls  ``self.submobjects.insert(index, mobject)``,
        where ``self.submobjects`` is a list.

        Highly adapted from ``OpenGLMobject.add``.

        Parameters
        ----------
        index
            The index at which
        mobject
            The mobject to be inserted.
        update_parent
            Whether or not to set ``mobject.parent`` to ``self``.
        """
        if update_parent:
            mobject.parent = self

        self._assert_valid_submobjects([mobject])

        if mobject not in self.submobjects:
            self.submobjects.insert(index, mobject)

        if self not in mobject.parents:
            mobject.parents.append(self)

        self.assemble_family()
        return self

    def remove(self, *mobjects: OpenGLMobject, update_parent: bool = False) -> Self:
        """Remove :attr:`submobjects`.

        The mobjects are removed from :attr:`submobjects`, if they exist.

        Subclasses of mobject may implement ``-`` and ``-=`` dunder methods.

        Parameters
        ----------
        mobjects
            The mobjects to remove.

        Returns
        -------
        :class:`OpenGLMobject`
            ``self``

        See Also
        --------
        :meth:`add`

        """
        if update_parent:
            assert len(mobjects) == 1, "Can't remove multiple parents."
            mobjects[0].parent = None

        for mobject in mobjects:
            if mobject in self.submobjects:
                self.submobjects.remove(mobject)
            if self in mobject.parents:
                mobject.parents.remove(self)
        self.assemble_family()
        return self

    def add_to_back(self, *mobjects: OpenGLMobject) -> Self:
        # NOTE: is the note true OpenGLMobjects?
        """Add all passed mobjects to the back of the submobjects.

        If :attr:`submobjects` already contains the given mobjects, they just get moved
        to the back instead.

        Parameters
        ----------
        mobjects
            The mobjects to add.

        Returns
        -------
        :class:`OpenGLMobject`
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
            When trying to add an object that is not an instance of :class:`OpenGLMobject`.

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
        self.submobjects = list_update(mobjects, self.submobjects)
        return self

    def replace_submobject(self, index: int, new_submob: OpenGLMobject) -> Self:
        self._assert_valid_submobjects([new_submob])
        old_submob = self.submobjects[index]
        if self in old_submob.parents:
            old_submob.parents.remove(self)
        self.submobjects[index] = new_submob
        self.assemble_family()
        return self

    # Submobject organization

    def arrange(
        self, direction: Vector3D = RIGHT, center: bool = True, **kwargs
    ) -> Self:
        """Sorts :class:`~.OpenGLMobject` next to each other on screen.

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
                    x = OpenGLVGroup(s1, s2, s3, s4).set_x(0).arrange(buff=1.0)
                    self.add(x)
        """
        for m1, m2 in zip(self.submobjects, self.submobjects[1:]):
            m2.next_to(m1, direction, **kwargs)
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
        row_heights: Sequence[float | None] | None = None,
        col_widths: Sequence[float | None] | None = None,
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
        OpenGLMobject
            The mobject.

        NOTES
        -----

        If only one of ``cols`` and ``rows`` is set implicitly, the other one will be chosen big
        enough to fit all submobjects. If neither is set, they will be chosen to be about the same,
        tending towards ``cols`` > ``rows`` (simply because videos are wider than they are high).

        If both ``cell_alignment`` and ``row_alignments`` / ``col_alignments`` are
        defined, the latter has higher priority.


        Raises
        ------
        ValueError
            If ``rows`` and ``cols`` are too small to fit all submobjects.
        ValueError
            If :code:`cols`, :code:`col_alignments` and :code:`col_widths` or :code:`rows`,
            :code:`row_alignments` and :code:`row_heights` have mismatching sizes.

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
        from manim.mobject.geometry.line import Line

        mobs = self.submobjects.copy()
        start_pos = self.get_center()

        # get cols / rows values if given (implicitly)
        def init_size(
            num: int | None,
            alignments: str | None,
            sizes: Sequence[float | None] | None,
            name: str,
        ) -> int:
            if num is not None:
                return num
            if alignments is not None:
                return len(alignments)
            if sizes is not None:
                return len(sizes)
            raise ValueError(
                f"At least one of the following parameters: '{name}s', "
                f"'{name}_alignments' or "
                f"'{name}_{'widths' if name == 'col' else 'heights'}', "
                "must not be None"
            )

        cols = init_size(cols, col_alignments, col_widths, "col")
        rows = init_size(rows, row_alignments, row_heights, "row")

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
        def init_alignments(
            str_alignments: str | None,
            num: int,
            mapping: dict[str, Vector3D],
            name: str,
            direction: Vector3D,
        ) -> Sequence[Vector3D]:
            if str_alignments is None:
                # Use cell_alignment as fallback
                return [cell_alignment * direction] * num
            if len(str_alignments) != num:
                raise ValueError(f"{name}_alignments has a mismatching size.")
            return [mapping[letter] for letter in str_alignments]

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
        def reverse(maybe_list: Sequence[Any] | None) -> Sequence[Any] | None:
            if maybe_list is not None:
                maybe_list = list(maybe_list)
                maybe_list.reverse()
                return maybe_list
            return None

        row_alignments = reverse(row_alignments)
        row_heights = reverse(row_heights)

        placeholder = OpenGLMobject()
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
        def init_sizes(
            sizes: Sequence[float | None] | None,
            num: int,
            measures: Sequence[float],
            name: str,
        ) -> Sequence[float]:
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

    def get_grid(
        self, n_rows: int, n_cols: int, height: float | None = None, **kwargs
    ) -> OpenGLGroup:
        """
        Returns a new mobject containing multiple copies of this one
        arranged in a grid
        """
        grid = self.duplicate(n_rows * n_cols)
        grid.arrange_in_grid(n_rows, n_cols, **kwargs)
        if height is not None:
            grid.set_height(height)
        return grid

    def duplicate(self, n: int) -> OpenGLGroup:
        """Returns an :class:`~.OpenGLGroup` containing ``n`` copies of the mobject."""
        return self.get_group_class()(*[self.copy() for _ in range(n)])

    def sort(
        self,
        point_to_num_func: Callable[[Point3DLike], float] = lambda p: p[0],
        submob_func: Callable[[OpenGLMobject], Any] | None = None,
    ) -> Self:
        """Sorts the list of :attr:`submobjects` by a function defined by ``submob_func``."""
        if submob_func is not None:
            self.submobjects.sort(key=submob_func)
        else:
            self.submobjects.sort(key=lambda m: point_to_num_func(m.get_center()))
        return self

    def shuffle(self, recurse: bool = False) -> Self:
        """Shuffles the order of :attr:`submobjects`

        Examples
        --------

        .. manim:: ShuffleSubmobjectsExample

            class ShuffleSubmobjectsExample(Scene):
                def construct(self):
                    s= OpenGLVGroup(*[Dot().shift(i*0.1*RIGHT) for i in range(-20,20)])
                    s2= s.copy()
                    s2.shuffle()
                    s2.shift(DOWN)
                    self.play(Write(s), Write(s2))
        """
        if recurse:
            for submob in self.submobjects:
                submob.shuffle(recurse=True)
        random.shuffle(self.submobjects)
        self.assemble_family()
        return self

    def invert(self, recursive: bool = False) -> Self:
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
        self.assemble_family()
        return self

    # Copying

    def copy(self, shallow: bool = False) -> OpenGLMobject:
        """Create and return an identical copy of the :class:`OpenGLMobject` including all
        :attr:`submobjects`.

        Returns
        -------
        :class:`OpenGLMobject`
            The copy.

        Parameters
        ----------
        shallow
            Controls whether a shallow copy is returned.

        Note
        ----
        The clone is initially not visible in the Scene, even if the original was.
        """
        if not shallow:
            return self.deepcopy()

        # TODO, either justify reason for shallow copy, or
        # remove this redundancy everywhere
        # return self.deepcopy()

        parents = self.parents
        self.parents = []
        copy_mobject = copy.copy(self)
        self.parents = parents

        copy_mobject.data = dict(self.data)
        for key in self.data:
            copy_mobject.data[key] = self.data[key].copy()

        # TODO, are uniforms ever numpy arrays?
        copy_mobject.uniforms = dict(self.uniforms)

        copy_mobject.submobjects = []
        copy_mobject.add(*(sm.copy() for sm in self.submobjects))
        copy_mobject.match_updaters(self)

        copy_mobject.needs_new_bounding_box = self.needs_new_bounding_box

        # Make sure any mobject or numpy array attributes are copied
        family = self.get_family()
        for attr, value in list(self.__dict__.items()):
            if (
                isinstance(value, OpenGLMobject)
                and value in family
                and value is not self
            ):
                setattr(copy_mobject, attr, value.copy())
            if isinstance(value, np.ndarray):
                setattr(copy_mobject, attr, value.copy())
            # if isinstance(value, ShaderWrapper):
            #     setattr(copy_mobject, attr, value.copy())
        return copy_mobject

    def deepcopy(self) -> OpenGLMobject:
        parents = self.parents
        self.parents = []
        result = copy.deepcopy(self)
        self.parents = parents
        return result

    def generate_target(self, use_deepcopy: bool = False) -> OpenGLMobject:
        self.target = None  # Prevent exponential explosion
        if use_deepcopy:
            self.target = self.deepcopy()
        else:
            self.target = self.copy()
        return self.target

    def save_state(self, use_deepcopy: bool = False) -> Self:
        """Save the current state (position, color & size). Can be restored with :meth:`~.OpenGLMobject.restore`."""
        if hasattr(self, "saved_state"):
            # Prevent exponential growth of data
            self.saved_state = None
        if use_deepcopy:
            self.saved_state = self.deepcopy()
        else:
            self.saved_state = self.copy()
        return self

    def restore(self) -> Self:
        """Restores the state that was previously saved with :meth:`~.OpenGLMobject.save_state`."""
        if not hasattr(self, "saved_state") or self.save_state is None:
            raise Exception("Trying to restore without having saved")
        self.become(self.saved_state)
        return self

    # Updating

    def init_updaters(self) -> None:
        self.time_based_updaters = []
        self.non_time_updaters = []
        self.has_updaters = False
        self.updating_suspended = False

    def update(self, dt: float = 0, recurse: bool = True) -> Self:
        if not self.has_updaters or self.updating_suspended:
            return self
        for updater in self.time_based_updaters:
            updater(self, dt)
        for updater in self.non_time_updaters:
            updater(self)
        if recurse:
            for submob in self.submobjects:
                submob.update(dt, recurse)
        return self

    def get_time_based_updaters(self) -> Sequence[TimeBasedUpdater]:
        return self.time_based_updaters

    def has_time_based_updater(self) -> bool:
        return len(self.time_based_updaters) > 0

    def get_updaters(self) -> Sequence[Updater]:
        return self.time_based_updaters + self.non_time_updaters

    def get_family_updaters(self) -> Sequence[Updater]:
        return list(it.chain(*(sm.get_updaters() for sm in self.get_family())))

    def add_updater(
        self,
        update_function: Updater,
        index: int | None = None,
        call_updater: bool = False,
    ) -> Self:
        if "dt" in inspect.signature(update_function).parameters:
            updater_list = self.time_based_updaters
        else:
            updater_list = self.non_time_updaters

        if index is None:
            updater_list.append(update_function)
        else:
            updater_list.insert(index, update_function)

        self.refresh_has_updater_status()
        if call_updater:
            self.update()
        return self

    def remove_updater(self, update_function: Updater) -> Self:
        for updater_list in [self.time_based_updaters, self.non_time_updaters]:
            while update_function in updater_list:
                updater_list.remove(update_function)
        self.refresh_has_updater_status()
        return self

    def clear_updaters(self, recurse: bool = True) -> Self:
        self.time_based_updaters = []
        self.non_time_updaters = []
        self.refresh_has_updater_status()
        if recurse:
            for submob in self.submobjects:
                submob.clear_updaters()
        return self

    def match_updaters(self, mobject: OpenGLMobject) -> Self:
        self.clear_updaters()
        for updater in mobject.get_updaters():
            self.add_updater(updater)
        return self

    def suspend_updating(self, recurse: bool = True) -> Self:
        self.updating_suspended = True
        if recurse:
            for submob in self.submobjects:
                submob.suspend_updating(recurse)
        return self

    def resume_updating(self, recurse: bool = True, call_updater: bool = True) -> Self:
        self.updating_suspended = False
        if recurse:
            for submob in self.submobjects:
                submob.resume_updating(recurse)
        for parent in self.parents:
            parent.resume_updating(recurse=False, call_updater=False)
        if call_updater:
            self.update(dt=0, recurse=recurse)
        return self

    def refresh_has_updater_status(self) -> Self:
        self.has_updaters = any(mob.get_updaters() for mob in self.get_family())
        return self

    # Transforming operations

    def shift(self, vector: Vector3D) -> Self:
        self.apply_points_function(
            lambda points: points + vector,
            about_edge=None,
            works_on_bounding_box=True,
        )
        return self

    def scale(
        self,
        scale_factor: float,
        about_point: Sequence[float] | None = None,
        about_edge: Sequence[float] = ORIGIN,
        **kwargs,
    ) -> Self:
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
            The scaling factor :math:`\alpha`. If :math:`0 < |\alpha| < 1`, the mobject
            will shrink, and for :math:`|\alpha| > 1` it will grow. Furthermore,
            if :math:`\alpha < 0`, the mobject is also flipped.
        kwargs
            Additional keyword arguments passed to
            :meth:`apply_points_function`.

        Returns
        -------
        OpenGLMobject
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

    def stretch(self, factor: float, dim: int, **kwargs) -> Self:
        def func(points: Point3D_Array) -> Point3D_Array:
            points[:, dim] *= factor
            return points

        self.apply_points_function(func, works_on_bounding_box=True, **kwargs)
        return self

    def rotate_about_origin(self, angle: float, axis: Vector3D = OUT) -> Self:
        return self.rotate(angle, axis, about_point=ORIGIN)

    def rotate(
        self,
        angle: float,
        axis: Vector3D = OUT,
        about_point: Sequence[float] | None = None,
        **kwargs,
    ) -> Self:
        """Rotates the :class:`~.OpenGLMobject` about a certain point."""
        rot_matrix_T = rotation_matrix_transpose(angle, axis)
        self.apply_points_function(
            lambda points: np.dot(points, rot_matrix_T),
            about_point=about_point,
            **kwargs,
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

    def apply_function(self, function: MappingFunction, **kwargs) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if len(kwargs) == 0:
            kwargs["about_point"] = ORIGIN

        def multi_mapping_function(points: Point3D_Array) -> Point3D_Array:
            result: Point3D_Array = np.apply_along_axis(function, 1, points)
            return result

        self.apply_points_function(multi_mapping_function, **kwargs)
        return self

    def apply_function_to_position(self, function: MappingFunction) -> Self:
        self.move_to(function(self.get_center()))
        return self

    def apply_function_to_submobject_positions(self, function: MappingFunction) -> Self:
        for submob in self.submobjects:
            submob.apply_function_to_position(function)
        return self

    def apply_matrix(self, matrix: MatrixMN, **kwargs) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if ("about_point" not in kwargs) and ("about_edge" not in kwargs):
            kwargs["about_point"] = ORIGIN
        full_matrix = np.identity(self.dim)
        matrix = np.array(matrix)
        full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        self.apply_points_function(
            lambda points: np.dot(points, full_matrix.T), **kwargs
        )
        return self

    def apply_complex_function(
        self, function: Callable[[complex], complex], **kwargs
    ) -> Self:
        """Applies a complex function to a :class:`OpenGLMobject`.
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

    def hierarchical_model_matrix(self) -> MatrixMN:
        if self.parent is None:
            return self.model_matrix

        model_matrices = [self.model_matrix]
        current_object = self
        while current_object.parent is not None:
            model_matrices.append(current_object.parent.model_matrix)
            current_object = current_object.parent
        return np.linalg.multi_dot(list(reversed(model_matrices)))

    def wag(
        self,
        direction: Vector3D = RIGHT,
        axis: Vector3D = DOWN,
        wag_factor: float = 1.0,
    ) -> Self:
        for mob in self.family_members_with_points():
            alphas = np.dot(mob.points, np.transpose(axis))
            alphas -= min(alphas)
            alphas /= max(alphas)
            alphas = alphas**wag_factor
            mob.set_points(
                mob.points
                + np.dot(
                    alphas.reshape((len(alphas), 1)),
                    np.array(direction).reshape((1, mob.dim)),
                ),
            )
        return self

    # Positioning methods

    def center(self) -> Self:
        """Moves the mobject to the center of the Scene."""
        self.shift(-self.get_center())
        return self

    def align_on_border(
        self,
        direction: Vector3D,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """
        Direction just needs to be a vector pointing towards side or
        corner in the 2d plane.
        """
        target_point = np.sign(direction) * (
            config["frame_x_radius"],
            config["frame_y_radius"],
            0,
        )
        point_to_align = self.get_bounding_box_point(direction)
        shift_val = target_point - point_to_align - buff * np.array(direction)
        shift_val = shift_val * abs(np.sign(direction))
        self.shift(shift_val)
        return self

    def to_corner(
        self,
        corner: Vector3D = LEFT + DOWN,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        return self.align_on_border(corner, buff)

    def to_edge(
        self,
        edge: Vector3D = LEFT,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        return self.align_on_border(edge, buff)

    def next_to(
        self,
        mobject_or_point: OpenGLMobject | Point3DLike,
        direction: Vector3D = RIGHT,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        aligned_edge: Vector3D = ORIGIN,
        submobject_to_align: OpenGLMobject | None = None,
        index_of_submobject_to_align: int | None = None,
        coor_mask: Point3DLike = np.array([1, 1, 1]),
    ) -> Self:
        """Move this :class:`~.OpenGLMobject` next to another's :class:`~.OpenGLMobject` or coordinate.

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
        if isinstance(mobject_or_point, OpenGLMobject):
            mob = mobject_or_point
            if index_of_submobject_to_align is not None:
                target_aligner = mob[index_of_submobject_to_align]
            else:
                target_aligner = mob
            target_point = target_aligner.get_bounding_box_point(
                aligned_edge + direction,
            )
        else:
            target_point = mobject_or_point
        if submobject_to_align is not None:
            aligner = submobject_to_align
        elif index_of_submobject_to_align is not None:
            aligner = self[index_of_submobject_to_align]
        else:
            aligner = self
        point_to_align = aligner.get_bounding_box_point(aligned_edge - direction)
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

    def is_off_screen(self) -> bool:
        if self.get_left()[0] > config.frame_x_radius:
            return True
        if self.get_right()[0] < config.frame_x_radius:
            return True
        if self.get_bottom()[1] > config.frame_y_radius:
            return True
        return self.get_top()[1] < -config.frame_y_radius

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

    def stretch_to_fit_width(self, width: float, **kwargs) -> Self:
        """Stretches the :class:`~.OpenGLMobject` to fit a width, not keeping height/depth proportional.

        Returns
        -------
        :class:`OpenGLMobject`
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

    def stretch_to_fit_height(self, height: float, **kwargs) -> Self:
        """Stretches the :class:`~.OpenGLMobject` to fit a height, not keeping width/height proportional."""
        return self.rescale_to_fit(height, 1, stretch=True, **kwargs)

    def stretch_to_fit_depth(self, depth: float, **kwargs) -> Self:
        """Stretches the :class:`~.OpenGLMobject` to fit a depth, not keeping width/height proportional."""
        return self.rescale_to_fit(depth, 1, stretch=True, **kwargs)

    def set_width(self, width: float, stretch: bool = False, **kwargs) -> Self:
        """Scales the :class:`~.OpenGLMobject` to fit a width while keeping height/depth proportional.

        Returns
        -------
        :class:`OpenGLMobject`
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
        return self.rescale_to_fit(width, 0, stretch=stretch, **kwargs)

    scale_to_fit_width = set_width

    def set_height(self, height: float, stretch: bool = False, **kwargs) -> Self:
        """Scales the :class:`~.OpenGLMobject` to fit a height while keeping width/depth proportional."""
        return self.rescale_to_fit(height, 1, stretch=stretch, **kwargs)

    scale_to_fit_height = set_height

    def set_depth(self, depth: float, stretch: bool = False, **kwargs):
        """Scales the :class:`~.OpenGLMobject` to fit a depth while keeping width/height proportional."""
        return self.rescale_to_fit(depth, 2, stretch=stretch, **kwargs)

    scale_to_fit_depth = set_depth

    def set_coord(self, value: float, dim: int, direction: Vector3D = ORIGIN) -> Self:
        curr = self.get_coord(dim, direction)
        shift_vect = np.zeros(self.dim)
        shift_vect[dim] = value - curr
        self.shift(shift_vect)
        return self

    def set_x(self, x: float, direction: Vector3D = ORIGIN) -> Self:
        """Set x value of the center of the :class:`~.OpenGLMobject` (``int`` or ``float``)"""
        return self.set_coord(x, 0, direction)

    def set_y(self, y: float, direction: Vector3D = ORIGIN) -> Self:
        """Set y value of the center of the :class:`~.OpenGLMobject` (``int`` or ``float``)"""
        return self.set_coord(y, 1, direction)

    def set_z(self, z: float, direction: Vector3D = ORIGIN) -> Self:
        """Set z value of the center of the :class:`~.OpenGLMobject` (``int`` or ``float``)"""
        return self.set_coord(z, 2, direction)

    def space_out_submobjects(self, factor: float = 1.5, **kwargs) -> Self:
        self.scale(factor, **kwargs)
        for submob in self.submobjects:
            submob.scale(1.0 / factor)
        return self

    def move_to(
        self,
        point_or_mobject: Point3DLike | OpenGLMobject,
        aligned_edge: Vector3D = ORIGIN,
        coor_mask: Point3DLike = np.array([1, 1, 1]),
    ) -> Self:
        """Move center of the :class:`~.OpenGLMobject` to certain coordinate."""
        if isinstance(point_or_mobject, OpenGLMobject):
            target = point_or_mobject.get_bounding_box_point(aligned_edge)
        else:
            target = point_or_mobject
        point_to_align = self.get_bounding_box_point(aligned_edge)
        self.shift((target - point_to_align) * coor_mask)
        return self

    def replace(
        self,
        mobject: OpenGLMobject,
        dim_to_match: int = 0,
        stretch: bool = False,
    ) -> Self:
        if not mobject.get_num_points() and not mobject.submobjects:
            self.scale(0)
            return self
        if stretch:
            for i in range(self.dim):
                self.rescale_to_fit(mobject.length_over_dim(i), i, stretch=True)
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
        mobject: OpenGLMobject,
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

    # Color functions

    def set_rgba_array(
        self,
        color: ParsableManimColor | Iterable[ParsableManimColor] | None = None,
        opacity: float | Iterable[float] | None = None,
        name: str = "rgbas",
        recurse: bool = True,
    ) -> Self:
        if color is not None:
            rgbs = np.array([color_to_rgb(c) for c in listify(color)])
        if opacity is not None:
            opacities = listify(opacity)

        # Color only
        if color is not None and opacity is None:
            for mob in self.get_family(recurse):
                mob.data[name] = resize_array(
                    mob.data[name] if name in mob.data else np.empty((1, 3)), len(rgbs)
                )
                mob.data[name][:, :3] = rgbs

        # Opacity only
        if color is None and opacity is not None:
            for mob in self.get_family(recurse):
                mob.data[name] = resize_array(
                    mob.data[name] if name in mob.data else np.empty((1, 3)),
                    len(opacities),
                )
                mob.data[name][:, 3] = opacities

        # Color and opacity
        if color is not None and opacity is not None:
            rgbas = np.array([[*rgb, o] for rgb, o in zip(*make_even(rgbs, opacities))])
            for mob in self.get_family(recurse):
                mob.data[name] = rgbas.copy()
        return self

    def set_rgba_array_direct(
        self,
        rgbas: npt.NDArray[RGBA_Array_Float],
        name: str = "rgbas",
        recurse: bool = True,
    ) -> Self:
        """Directly set rgba data from `rgbas` and optionally do the same recursively
        with submobjects. This can be used if the `rgbas` have already been generated
        with the correct shape and simply need to be set.

        Parameters
        ----------
        rgbas
            the rgba to be set as data
        name
            the name of the data attribute to be set
        recurse
            set to true to recursively apply this method to submobjects
        """
        for mob in self.get_family(recurse):
            mob.data[name] = rgbas.copy()

    def set_color(
        self,
        color: ParsableManimColor | Iterable[ParsableManimColor] | None,
        opacity: float | Iterable[float] | None = None,
        recurse: bool = True,
    ) -> Self:
        self.set_rgba_array(color, opacity, recurse=False)
        # Recurse to submobjects differently from how set_rgba_array
        # in case they implement set_color differently
        if color is not None:
            self.color: ManimColor = ManimColor.parse(color)
        if opacity is not None:
            self.opacity = opacity
        if recurse:
            for submob in self.submobjects:
                submob.set_color(color, recurse=True)
        return self

    def set_opacity(
        self, opacity: float | Iterable[float] | None, recurse: bool = True
    ) -> Self:
        self.set_rgba_array(color=None, opacity=opacity, recurse=False)
        if recurse:
            for submob in self.submobjects:
                submob.set_opacity(opacity, recurse=True)
        return self

    def get_color(self) -> str:
        return rgb_to_hex(self.rgbas[0, :3])

    def get_opacity(self) -> float:
        return self.rgbas[0, 3]

    def set_color_by_gradient(self, *colors: ParsableManimColor) -> Self:
        return self.set_submobject_colors_by_gradient(*colors)

    def set_submobject_colors_by_gradient(self, *colors: ParsableManimColor) -> Self:
        if len(colors) == 0:
            raise Exception("Need at least one color")
        elif len(colors) == 1:
            return self.set_color(*colors)

        # mobs = self.family_members_with_points()
        mobs = self.submobjects
        new_colors = color_gradient(colors, len(mobs))

        for mob, color in zip(mobs, new_colors):
            mob.set_color(color)
        return self

    def fade(self, darkness: float = 0.5, recurse: bool = True) -> Self:
        return self.set_opacity(1.0 - darkness, recurse=recurse)

    def get_gloss(self) -> float:
        return self.gloss

    def set_gloss(self, gloss: float, recurse: bool = True) -> Self:
        for mob in self.get_family(recurse):
            mob.gloss = gloss
        return self

    def get_shadow(self) -> float:
        return self.shadow

    def set_shadow(self, shadow: float, recurse: bool = True) -> Self:
        for mob in self.get_family(recurse):
            mob.shadow = shadow
        return self

    # Background rectangle

    def add_background_rectangle(
        self,
        color: ParsableManimColor | None = None,
        opacity: float = 0.75,
        **kwargs,
    ) -> Self:
        # TODO, this does not behave well when the mobject has points,
        # since it gets displayed on top
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
        :class:`OpenGLMobject`
            ``self``

        See Also
        --------
        :meth:`add_to_back`
        :class:`~.BackgroundRectangle`

        """
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

    # Getters

    def get_bounding_box_point(self, direction: Vector3D) -> Point3D:
        bb = self.get_bounding_box()
        indices = (np.sign(direction) + 1).astype(int)
        return np.array([bb[indices[i]][i] for i in range(3)])

    def get_edge_center(self, direction: Vector3D) -> Point3D:
        """Get edge coordinates for certain direction."""
        return self.get_bounding_box_point(direction)

    def get_corner(self, direction: Vector3D) -> Point3D:
        """Get corner coordinates for certain direction."""
        return self.get_bounding_box_point(direction)

    def get_center(self) -> Point3D:
        """Get center coordinates."""
        return self.get_bounding_box()[1]

    def get_center_of_mass(self) -> Point3D:
        return self.get_all_points().mean(0)

    def get_boundary_point(self, direction: Vector3D) -> Point3D:
        all_points = self.get_all_points()
        boundary_directions = all_points - self.get_center()
        norms = np.linalg.norm(boundary_directions, axis=1)
        boundary_directions /= np.repeat(norms, 3).reshape((len(norms), 3))
        index = np.argmax(np.dot(boundary_directions, np.array(direction).T))
        return all_points[index]

    def get_continuous_bounding_box_point(self, direction: Vector3D) -> Point3D:
        dl, center, ur = self.get_bounding_box()
        corner_vect = ur - center
        return center + direction / np.max(
            np.abs(
                np.true_divide(
                    direction,
                    corner_vect,
                    out=np.zeros(len(direction)),
                    where=((corner_vect) != 0),
                ),
            ),
        )

    def get_top(self) -> Point3D:
        """Get top coordinates of a box bounding the :class:`~.OpenGLMobject`"""
        return self.get_edge_center(UP)

    def get_bottom(self) -> Point3D:
        """Get bottom coordinates of a box bounding the :class:`~.OpenGLMobject`"""
        return self.get_edge_center(DOWN)

    def get_right(self) -> Point3D:
        """Get right coordinates of a box bounding the :class:`~.OpenGLMobject`"""
        return self.get_edge_center(RIGHT)

    def get_left(self) -> Point3D:
        """Get left coordinates of a box bounding the :class:`~.OpenGLMobject`"""
        return self.get_edge_center(LEFT)

    def get_zenith(self) -> Point3D:
        """Get zenith coordinates of a box bounding a 3D :class:`~.OpenGLMobject`."""
        return self.get_edge_center(OUT)

    def get_nadir(self) -> Point3D:
        """Get nadir (opposite the zenith) coordinates of a box bounding a 3D :class:`~.OpenGLMobject`."""
        return self.get_edge_center(IN)

    def length_over_dim(self, dim: int) -> float:
        bb = self.get_bounding_box()
        return abs((bb[2] - bb[0])[dim])

    def get_width(self) -> float:
        """Returns the width of the mobject."""
        return self.length_over_dim(0)

    def get_height(self) -> float:
        """Returns the height of the mobject."""
        return self.length_over_dim(1)

    def get_depth(self) -> float:
        """Returns the depth of the mobject."""
        return self.length_over_dim(2)

    def get_coord(self, dim: int, direction: Vector3D = ORIGIN) -> ManimFloat:
        """Meant to generalize ``get_x``, ``get_y`` and ``get_z``"""
        return self.get_bounding_box_point(direction)[dim]

    def get_x(self, direction: Vector3D = ORIGIN) -> ManimFloat:
        """Returns x coordinate of the center of the :class:`~.OpenGLMobject` as ``float``"""
        return self.get_coord(0, direction)

    def get_y(self, direction: Vector3D = ORIGIN) -> ManimFloat:
        """Returns y coordinate of the center of the :class:`~.OpenGLMobject` as ``float``"""
        return self.get_coord(1, direction)

    def get_z(self, direction: Vector3D = ORIGIN) -> ManimFloat:
        """Returns z coordinate of the center of the :class:`~.OpenGLMobject` as ``float``"""
        return self.get_coord(2, direction)

    def get_start(self) -> Point3D:
        """Returns the point, where the stroke that surrounds the :class:`~.OpenGLMobject` starts."""
        self.throw_error_if_no_points()
        return np.array(self.points[0])

    def get_end(self) -> Point3D:
        """Returns the point, where the stroke that surrounds the :class:`~.OpenGLMobject` ends."""
        self.throw_error_if_no_points()
        return np.array(self.points[-1])

    def get_start_and_end(self) -> tuple[Point3D, Point3D]:
        """Returns starting and ending point of a stroke as a ``tuple``."""
        return self.get_start(), self.get_end()

    def point_from_proportion(self, alpha: float) -> Point3D:
        points = self.points
        i, subalpha = integer_interpolate(0, len(points) - 1, alpha)
        return interpolate(points[i], points[i + 1], subalpha)

    def pfp(self, alpha: float) -> Point3D:
        """Abbreviation for point_from_proportion"""
        return self.point_from_proportion(alpha)

    def get_pieces(self, n_pieces: int) -> OpenGLMobject:
        template = self.copy()
        template.submobjects = []
        alphas = np.linspace(0, 1, n_pieces + 1)
        return OpenGLGroup(
            *(
                template.copy().pointwise_become_partial(self, a1, a2)
                for a1, a2 in zip(alphas[:-1], alphas[1:])
            )
        )

    def get_z_index_reference_point(self) -> Point3D:
        # TODO, better place to define default z_index_group?
        z_index_group = getattr(self, "z_index_group", self)
        return z_index_group.get_center()

    # Match other mobject properties

    def match_color(self, mobject: OpenGLMobject) -> Self:
        """Match the color with the color of another :class:`~.OpenGLMobject`."""
        return self.set_color(mobject.get_color())

    def match_dim_size(self, mobject: OpenGLMobject, dim: int, **kwargs) -> Self:
        """Match the specified dimension with the dimension of another :class:`~.OpenGLMobject`."""
        return self.rescale_to_fit(mobject.length_over_dim(dim), dim, **kwargs)

    def match_width(self, mobject: OpenGLMobject, **kwargs) -> Self:
        """Match the width with the width of another :class:`~.OpenGLMobject`."""
        return self.match_dim_size(mobject, 0, **kwargs)

    def match_height(self, mobject: OpenGLMobject, **kwargs) -> Self:
        """Match the height with the height of another :class:`~.OpenGLMobject`."""
        return self.match_dim_size(mobject, 1, **kwargs)

    def match_depth(self, mobject: OpenGLMobject, **kwargs) -> Self:
        """Match the depth with the depth of another :class:`~.OpenGLMobject`."""
        return self.match_dim_size(mobject, 2, **kwargs)

    def match_coord(
        self, mobject: OpenGLMobject, dim: int, direction: Vector3D = ORIGIN
    ) -> Self:
        """Match the coordinates with the coordinates of another :class:`~.OpenGLMobject`."""
        return self.set_coord(
            mobject.get_coord(dim, direction),
            dim=dim,
            direction=direction,
        )

    def match_x(self, mobject: OpenGLMobject, direction: Vector3D = ORIGIN) -> Self:
        """Match x coord. to the x coord. of another :class:`~.OpenGLMobject`."""
        return self.match_coord(mobject, 0, direction)

    def match_y(self, mobject: OpenGLMobject, direction: Vector3D = ORIGIN) -> Self:
        """Match y coord. to the x coord. of another :class:`~.OpenGLMobject`."""
        return self.match_coord(mobject, 1, direction)

    def match_z(self, mobject: OpenGLMobject, direction: Vector3D = ORIGIN) -> Self:
        """Match z coord. to the x coord. of another :class:`~.OpenGLMobject`."""
        return self.match_coord(mobject, 2, direction)

    def align_to(
        self,
        mobject_or_point: OpenGLMobject | Point3DLike,
        direction: Vector3D = ORIGIN,
    ) -> Self:
        """
        Examples:
        mob1.align_to(mob2, UP) moves mob1 vertically so that its
        top edge lines ups with mob2's top edge.

        mob1.align_to(mob2, alignment_vect = RIGHT) moves mob1
        horizontally so that it's center is directly above/below
        the center of mob2
        """
        if isinstance(mobject_or_point, OpenGLMobject):
            point = mobject_or_point.get_bounding_box_point(direction)
        else:
            point = mobject_or_point

        for dim in range(self.dim):
            if direction[dim] != 0:
                self.set_coord(point[dim], dim, direction)
        return self

    def get_group_class(self) -> type[OpenGLGroup]:
        return OpenGLGroup

    @staticmethod
    def get_mobject_type_class() -> type[OpenGLMobject]:
        """Return the base class of this mobject type."""
        return OpenGLMobject

    # Alignment

    def align_data_and_family(self, mobject: OpenGLMobject) -> Self:
        self.align_family(mobject)
        self.align_data(mobject)
        return self

    def align_data(self, mobject: OpenGLMobject) -> Self:
        # In case any data arrays get resized when aligned to shader data
        # self.refresh_shader_data()
        for mob1, mob2 in zip(self.get_family(), mobject.get_family()):
            # Separate out how points are treated so that subclasses
            # can handle that case differently if they choose
            mob1.align_points(mob2)
            for key in mob1.data.keys() & mob2.data.keys():
                if key == "points":
                    continue
                arr1 = mob1.data[key]
                arr2 = mob2.data[key]
                if len(arr2) > len(arr1):
                    mob1.data[key] = resize_preserving_order(arr1, len(arr2))
                elif len(arr1) > len(arr2):
                    mob2.data[key] = resize_preserving_order(arr2, len(arr1))
        return self

    def align_points(self, mobject: OpenGLMobject) -> Self:
        max_len = max(self.get_num_points(), mobject.get_num_points())
        for mob in (self, mobject):
            mob.resize_points(max_len, resize_func=resize_preserving_order)
        return self

    def align_family(self, mobject: OpenGLMobject) -> Self:
        mob1 = self
        mob2 = mobject
        n1 = len(mob1)
        n2 = len(mob2)
        if n1 != n2:
            mob1.add_n_more_submobjects(max(0, n2 - n1))
            mob2.add_n_more_submobjects(max(0, n1 - n2))
        # Recurse
        for sm1, sm2 in zip(mob1.submobjects, mob2.submobjects):
            sm1.align_family(sm2)
        return self

    def push_self_into_submobjects(self) -> Self:
        copy = self.deepcopy()
        copy.submobjects = []
        self.resize_points(0)
        self.add(copy)
        return self

    def add_n_more_submobjects(self, n: int) -> Self:
        if n == 0:
            return self

        curr = len(self.submobjects)
        if curr == 0:
            # If empty, simply add n point mobjects
            null_mob = self.copy()
            null_mob.set_points([self.get_center()])
            self.submobjects = [null_mob.copy() for k in range(n)]
            return self
        target = curr + n
        repeat_indices = (np.arange(target) * curr) // target
        split_factors = [(repeat_indices == i).sum() for i in range(curr)]
        new_submobs = []
        for submob, sf in zip(self.submobjects, split_factors):
            new_submobs.append(submob)
            for _ in range(1, sf):
                new_submob = submob.copy()
                # If the submobject is at all transparent, then
                # make the copy completely transparent
                if submob.get_opacity() < 1:
                    new_submob.set_opacity(0)
                new_submobs.append(new_submob)
        self.submobjects = new_submobs
        return self

    # Interpolate

    def interpolate(
        self,
        mobject1: OpenGLMobject,
        mobject2: OpenGLMobject,
        alpha: float,
        path_func: PathFuncType = straight_path(),
    ) -> Self:
        """Turns this :class:`~.OpenGLMobject` into an interpolation between ``mobject1``
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

                    dotMiddle = OpenGLVMobject().interpolate(dotL, dotR, alpha=0.3)

                    self.add(dotL, dotR, dotMiddle)
        """
        for key in self.data:
            if key in self.locked_data_keys:
                continue
            if len(self.data[key]) == 0:
                continue
            if key not in mobject1.data or key not in mobject2.data:
                continue

            func = path_func if key in ("points", "bounding_box") else interpolate

            self.data[key][:] = func(mobject1.data[key], mobject2.data[key], alpha)

        for key in self.uniforms:
            if key != "fixed_orientation_center":
                self.uniforms[key] = interpolate(
                    mobject1.uniforms[key],
                    mobject2.uniforms[key],
                    alpha,
                )
            else:
                self.uniforms["fixed_orientation_center"] = tuple(
                    interpolate(
                        np.array(mobject1.uniforms["fixed_orientation_center"]),
                        np.array(mobject2.uniforms["fixed_orientation_center"]),
                        alpha,
                    )
                )
        return self

    def pointwise_become_partial(
        self, mobject: OpenGLMobject, a: float, b: float
    ) -> None:
        """
        Set points in such a way as to become only
        part of mobject.
        Inputs 0 <= a < b <= 1 determine what portion
        of mobject to become.
        """
        pass  # To implement in subclass

    def become(
        self,
        mobject: OpenGLMobject,
        match_height: bool = False,
        match_width: bool = False,
        match_depth: bool = False,
        match_center: bool = False,
        stretch: bool = False,
    ) -> Self:
        """Edit all data and submobjects to be identical
        to another :class:`~.OpenGLMobject`

        .. note::

            If both match_height and match_width are ``True`` then the transformed :class:`~.OpenGLMobject`
            will match the height first and then the width

        Parameters
        ----------
        match_height
            If ``True``, then the transformed :class:`~.OpenGLMobject` will match the height of the original
        match_width
            If ``True``, then the transformed :class:`~.OpenGLMobject` will match the width of the original
        match_depth
            If ``True``, then the transformed :class:`~.OpenGLMobject` will match the depth of the original
        match_center
            If ``True``, then the transformed :class:`~.OpenGLMobject` will match the center of the original
        stretch
            If ``True``, then the transformed :class:`~.OpenGLMobject` will stretch to fit the proportions of the original

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

        self.align_family(mobject)
        for sm1, sm2 in zip(self.get_family(), mobject.get_family()):
            sm1.set_data(sm2.data)
            sm1.set_uniforms(sm2.uniforms)
        self.refresh_bounding_box(recurse_down=True)
        return self

    # Locking data

    def lock_data(self, keys: Iterable[str]) -> None:
        """
        To speed up some animations, particularly transformations,
        it can be handy to acknowledge which pieces of data
        won't change during the animation so that calls to
        interpolate can skip this, and so that it's not
        read into the shader_wrapper objects needlessly
        """
        if self.has_updaters:
            return
        # Be sure shader data has most up to date information
        self.refresh_shader_data()
        self.locked_data_keys = set(keys)

    def lock_matching_data(
        self, mobject1: OpenGLMobject, mobject2: OpenGLMobject
    ) -> Self:
        for sm, sm1, sm2 in zip(
            self.get_family(),
            mobject1.get_family(),
            mobject2.get_family(),
        ):
            keys = sm.data.keys() & sm1.data.keys() & sm2.data.keys()
            sm.lock_data(
                list(
                    filter(
                        lambda key: np.all(sm1.data[key] == sm2.data[key]),
                        keys,
                    ),
                ),
            )
        return self

    def unlock_data(self) -> None:
        for mob in self.get_family():
            mob.locked_data_keys = set()

    # Operations touching shader uniforms

    @affects_shader_info_id
    def fix_in_frame(self) -> Self:
        self.is_fixed_in_frame = 1.0
        return self

    @affects_shader_info_id
    def fix_orientation(self) -> Self:
        self.is_fixed_orientation = 1.0
        self.fixed_orientation_center = tuple(self.get_center())
        self.depth_test = True
        return self

    @affects_shader_info_id
    def unfix_from_frame(self) -> Self:
        self.is_fixed_in_frame = 0.0
        return self

    @affects_shader_info_id
    def unfix_orientation(self) -> Self:
        self.is_fixed_orientation = 0.0
        self.fixed_orientation_center = (0, 0, 0)
        self.depth_test = False
        return self

    @affects_shader_info_id
    def apply_depth_test(self) -> Self:
        self.depth_test = True
        return self

    @affects_shader_info_id
    def deactivate_depth_test(self) -> Self:
        self.depth_test = False
        return self

    # Shader code manipulation

    def replace_shader_code(self, old_code: str, new_code: str) -> Self:
        # TODO, will this work with VMobject structure, given
        # that it does not simpler return shader_wrappers of
        # family?
        for wrapper in self.get_shader_wrapper_list():
            wrapper.replace_code(old_code, new_code)
        return self

    def set_color_by_code(self, glsl_code: str) -> Self:
        """
        Takes a snippet of code and inserts it into a
        context which has the following variables:
        vec4 color, vec3 point, vec3 unit_normal.
        The code should change the color variable
        """
        self.replace_shader_code("///// INSERT COLOR FUNCTION HERE /////", glsl_code)
        return self

    def set_color_by_xyz_func(
        self,
        glsl_snippet: str,
        min_value: float = -5.0,
        max_value: float = 5.0,
        colormap: str = "viridis",
    ) -> Self:
        """
        Pass in a glsl expression in terms of x, y and z which returns
        a float.
        """
        # TODO, add a version of this which changes the point data instead
        # of the shader code
        for char in "xyz":
            glsl_snippet = glsl_snippet.replace(char, "point." + char)
        rgb_list = get_colormap_list(colormap)
        self.set_color_by_code(
            f"color.rgb = float_to_color({glsl_snippet}, {float(min_value)}, {float(max_value)}, {get_colormap_code(rgb_list)});",
        )
        return self

    # For shader data

    def refresh_shader_wrapper_id(self) -> Self:
        self.get_shader_wrapper().refresh_id()
        return self

    def get_shader_wrapper(self) -> ShaderWrapper:
        from manim.renderer.shader_wrapper import ShaderWrapper

        # if hasattr(self, "__shader_wrapper"):
        # return self.__shader_wrapper

        self.shader_wrapper = ShaderWrapper(
            vert_data=self.get_shader_data(),
            vert_indices=self.get_shader_vert_indices(),
            uniforms=self.get_shader_uniforms(),
            depth_test=self.depth_test,
            texture_paths=self.texture_paths,
            render_primitive=self.render_primitive,
            shader_folder=self.__class__.shader_folder,
        )
        return self.shader_wrapper

    def get_shader_wrapper_list(self) -> Sequence[ShaderWrapper]:
        shader_wrappers = it.chain(
            [self.get_shader_wrapper()],
            *(sm.get_shader_wrapper_list() for sm in self.submobjects),
        )
        batches = batch_by_property(shader_wrappers, lambda sw: sw.get_id())

        result = []
        for wrapper_group, _ in batches:
            shader_wrapper = wrapper_group[0]
            if not shader_wrapper.is_valid():
                continue
            shader_wrapper.combine_with(*wrapper_group[1:])
            if len(shader_wrapper.vert_data) > 0:
                result.append(shader_wrapper)
        return result

    def check_data_alignment(self, array: npt.NDArray, data_key: str) -> Self:
        # Makes sure that self.data[key] can be broadcast into
        # the given array, meaning its length has to be either 1
        # or the length of the array
        d_len = len(self.data[data_key])
        if d_len != 1 and d_len != len(array):
            self.data[data_key] = resize_with_interpolation(
                self.data[data_key],
                len(array),
            )
        return self

    def get_resized_shader_data_array(self, length: float) -> npt.NDArray:
        # If possible, try to populate an existing array, rather
        # than recreating it each frame
        points = self.points
        shader_data = np.zeros(len(points), dtype=self.shader_dtype)
        return shader_data

    def read_data_to_shader(
        self,
        shader_data: npt.NDArray,  # has structured data type, ex. ("point", np.float32, (3,))
        shader_data_key: str,
        data_key: str,
    ) -> None:
        if data_key in self.locked_data_keys:
            return
        self.check_data_alignment(shader_data, data_key)
        shader_data[shader_data_key] = self.data[data_key]

    def get_shader_data(self) -> npt.NDArray:
        shader_data = self.get_resized_shader_data_array(self.get_num_points())
        self.read_data_to_shader(shader_data, "point", "points")
        return shader_data

    def refresh_shader_data(self) -> None:
        self.get_shader_data()

    def get_shader_uniforms(self) -> dict[str, Any]:
        return self.uniforms

    def get_shader_vert_indices(self) -> Sequence[int]:
        return self.shader_indices

    @property
    def submobjects(self) -> Sequence[OpenGLMobject]:
        return self._submobjects if hasattr(self, "_submobjects") else []

    @submobjects.setter
    def submobjects(self, submobject_list: Iterable[OpenGLMobject]) -> None:
        self.remove(*self.submobjects)
        self.add(*submobject_list)

    # Errors

    def throw_error_if_no_points(self) -> None:
        if not self.has_points():
            message = (
                "Cannot call OpenGLMobject.{} " + "for a OpenGLMobject with no points"
            )
            caller_name = sys._getframe(1).f_code.co_name
            raise Exception(message.format(caller_name))


class OpenGLGroup(OpenGLMobject):
    def __init__(self, *mobjects: OpenGLMobject, **kwargs):
        super().__init__(**kwargs)
        self.add(*mobjects)


class OpenGLPoint(OpenGLMobject):
    def __init__(
        self,
        location: Point3DLike = ORIGIN,
        artificial_width: float = 1e-6,
        artificial_height: float = 1e-6,
        **kwargs,
    ):
        self.artificial_width = artificial_width
        self.artificial_height = artificial_height
        super().__init__(**kwargs)
        self.set_location(location)

    def get_width(self) -> float:
        return self.artificial_width

    def get_height(self) -> float:
        return self.artificial_height

    def get_location(self) -> Point3D:
        return self.points[0].copy()

    def get_bounding_box_point(self, *args, **kwargs) -> Point3D:
        return self.get_location()

    def set_location(self, new_loc: Point3D) -> None:
        self.set_points(np.array(new_loc, ndmin=2, dtype=float))


class _AnimationBuilder:
    def __init__(self, mobject: OpenGLMobject):
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

    def __getattr__(self, method_name: str) -> Callable[..., Self]:
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

    def build(self) -> _MethodAnimation:
        from manim.animation.transform import _MethodAnimation

        if self.overridden_animation:
            anim = self.overridden_animation
        else:
            anim = _MethodAnimation(self.mobject, self.methods)

        for attr, value in self.anim_args.items():
            setattr(anim, attr, value)

        return anim


def override_animate(method: types.FunctionType) -> types.FunctionType:
    r"""Decorator for overriding method animations.

    This allows to specify a method (returning an :class:`~.Animation`)
    which is called when the decorated method is used with the ``.animate`` syntax
    for animating the application of a method.

    .. seealso::

        :attr:`OpenGLMobject.animate`

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
