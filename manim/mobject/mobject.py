"""Base classes for objects that can be displayed."""


__all__ = ["Mobject", "Group", "override_animate"]


from functools import reduce, wraps
import copy
import itertools as it
import operator as op
import random
import sys
import types
from typing import Callable, List, Optional, Union
import warnings

from pathlib import Path
from colour import Color
from manim.utils.bezier import integer_interpolate
import moderngl
import numpy as np

from .. import config
from ..constants import *
from ..container import Container
from ..utils.color import (
    Colors,
    color_gradient,
    WHITE,
    BLACK,
    YELLOW_C,
    color_to_rgb,
    rgb_to_hex,
)
from ..utils.color import interpolate_color
from ..utils.iterables import (
    batch_by_property,
    list_update,
    listify,
    make_even,
    resize_preserving_order,
    resize_with_interpolation,
)
from ..utils.iterables import remove_list_redundancies
from ..utils.iterables import resize_array
from ..utils.paths import straight_path
from ..utils.simple_functions import get_parameters
from ..utils.space_ops import angle_of_vector
from ..utils.space_ops import get_norm
from ..utils.space_ops import rotation_matrix
from ..utils.space_ops import rotation_matrix_transpose

# TODO: Explain array_attrs

Updater = Union[Callable[["Mobject"], None], Callable[["Mobject", float], None]]


def affects_shader_info_id(func):
    @wraps(func)
    def wrapper(self):
        for mob in self.get_family():
            func(mob)
            # mob.refresh_shader_wrapper_id()
        return self

    return wrapper


def interpolate(start: int, end: int, alpha: float) -> float:
    return (1 - alpha) * start + alpha * end


class Mobject(Container):
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

    """

    shader_dtype = [
        ("point", np.float32, (3,)),
    ]
    shader_folder = ""

    def __init__(
        self,
        color=WHITE,
        opacity=1,
        name=None,
        dim=3,
        target=None,
        z_index=0,
        gloss=0.0,
        # Positive shadow up to 1 makes a side opposite the light darker
        shadow=0.0,
        # For shaders
        render_primitive=moderngl.TRIANGLE_STRIP,
        texture_paths=None,
        depth_test=False,
        # If true, the mobject will not get rotated according to camera position
        is_fixed_in_frame=False,
        # Must match in attributes of vert shader
        # Event listener
        listen_to_events=False,
        **kwargs,
    ):
        # OpenGL data.

        # If true, the mobject will not get rotated according to camera position
        self.is_fixed_in_frame = is_fixed_in_frame
        self.gloss = gloss
        self.shadow = shadow
        self.needs_new_bounding_box = True
        self.locked_data_keys = set()

        self.opengl = config["use_opengl_renderer"]
        self.opacity = opacity
        # For shaders
        self.render_primitive = render_primitive
        self.texture_paths = texture_paths
        self.depth_test = depth_test

        # Must match in attributes of vert shader
        # Event listener
        self.listen_to_events = listen_to_events
        self.dim = dim
        self.submobjects = []
        self.parents = []
        self.updaters = []
        self.updating_suspended = False
        self.family = [self]
        self.name = self.__class__.__name__ if name is None else name
        self.target = target
        self.z_index = z_index
        self.point_hash = None

        self.init_data()
        self.init_uniforms()
        self.reset_points()

        self.color = Color(color)

        # self.init_event_listners()
        self.generate_points()
        self.init_colors()

        self.shader_indices = None
        if self.depth_test:
            self.apply_depth_test()

        Container.__init__(self, **kwargs)

    def get_points(self):
        return self.points

    def init_data(self):
        self.data = {
            "points": np.zeros((0, 3)),
            "bounding_box": np.zeros((3, 3)),
            "rgbas": np.zeros((1, 4)),
        }

    def init_uniforms(self):
        self.uniforms = {
            "is_fixed_in_frame": float(self.is_fixed_in_frame),
            "gloss": self.gloss,
            "shadow": self.shadow,
        }

    def resize_points(self, new_length, resize_func=resize_array):
        if new_length != len(self.points):
            self.points = resize_func(self.points, new_length)
        self.refresh_bounding_box()
        return self

    def get_bounding_box(self):
        if self.needs_new_bounding_box:
            self.data["bounding_box"] = self.compute_bounding_box()
            self.needs_new_bounding_box = False
        return self.data["bounding_box"]

    def compute_bounding_box(self):
        all_points = np.vstack(
            [
                self.get_points(),
                *(
                    mob.get_bounding_box()
                    for mob in self.get_family()[1:]
                    if mob.has_points()
                ),
            ]
        )
        if len(all_points) == 0:
            return np.zeros((3, self.dim))
        else:
            # Lower left and upper right corners
            mins = all_points.min(0)
            maxs = all_points.max(0)
            mids = (mins + maxs) / 2
            return np.array([mins, mids, maxs])

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
        return str(self.name)

    def reset_points(self):
        """Sets :attr:`points` to be an empty array."""
        self.points = np.zeros((0, self.dim))

    def init_colors(self):
        """Initializes the colors.

        Gets called upon creation. This is an empty method that can be implemented by subclasses.
        """
        if self.opengl:
            self.set_color(self.color, self.opacity)

    def generate_points(self):
        """Initializes :attr:`points` and therefore the shape.

        Gets called upon creation. This is an empty method that can be implemented by subclasses.
        """
        pass

    @property
    def points(self):
        return self.data["points"]

    @points.setter
    def points(self, value):
        self.data["points"] = value

    def match_points(self, mobject):
        self.set_points(mobject.get_points())

    def set_points(self, points):
        if len(points) == len(self.points):
            self.points[:] = points
        elif isinstance(points, np.ndarray):
            self.points = points.copy()
        else:
            self.points = np.array(points)
        self.refresh_bounding_box()

        return self

    def set_data(self, data):
        for key in data:
            self.data[key] = data[key].copy()
        return self

    def refresh_bounding_box(self, recurse_down=False, recurse_up=True):
        for mob in self.get_family(recurse_down):
            mob.needs_new_bounding_box = True
        if recurse_up:
            for parent in self.parents:
                parent.refresh_bounding_box()
        return self

    def is_point_touching(self, point, buff=MED_SMALL_BUFF):
        bb = self.get_bounding_box()
        mins = bb[0] - buff
        maxs = bb[2] + buff
        return (point >= mins).all() and (point <= maxs).all()

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
        for mobject in mobjects:
            if self in mobjects:
                raise ValueError("Mobject cannot contain self")
            if not isinstance(mobject, Mobject):
                raise TypeError("All submobjects must be of type Mobject")
            if mobject not in self.submobjects:
                self.submobjects = list_update(self.submobjects, mobjects)
            # if self not in mobject.parents:
            #     mobject.parents = list_update(self.submobjects, mobject.parents)
        self.assemble_family()
        return self

    def __add__(self, mobject):
        raise NotImplementedError

    def __iadd__(self, mobject):
        raise NotImplementedError

    def set_submobjects(self, submobject_list):
        self.remove(*self.submobjects)
        self.add(*submobject_list)
        return self

    def replace_submobject(self, index, new_submob):
        old_submob = self.submobjects[index]
        if self in old_submob.parents:
            old_submob.parents.remove(self)
        self.submobjects[index] = new_submob
        self.assemble_family()
        return self

    def digest_mobject_attrs(self):
        """
        Ensures all attributes which are mobjects are included
        in the submobjects list.
        """
        mobject_attrs = [
            x for x in list(self.__dict__.values()) if isinstance(x, Mobject)
        ]
        self.set_submobjects(list_update(self.submobjects, mobject_attrs))
        return self

    def add_to_back(self, *mobjects: "Mobject") -> "Mobject":
        """Add all passed mobjects to the back of the submobjects.

        If :attr:`submobjects` already contains the given mobjects, they just get moved to the back instead.

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
        """
        self.set_submobjects(list_update(mobjects, self.submobjects))
        return self

    def remove(self, *mobjects: "Mobject") -> "Mobject":
        """Remove submobjects.

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
            # if self in mobject.parents:
            #     mobject.parents.remove(self)
        self.assemble_family()
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
                    "This method is not guaranteed to stay around. Please prefer getting the attribute normally.",
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
                    "This method is not guaranteed to stay around. Please prefer setting the attribute normally or with Mobject.set().",
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
        self.get_image().save(
            Path(config.get_dir("video_dir")).joinpath((name or str(self)) + ".png")
        )

    def copy(self) -> "Mobject":
        """Create and return an identical copy of the Mobject including all submobjects.

        Returns
        -------
        :class:`Mobject`
            The copy.

        Note
        ----
        The clone is initially not visible in the Scene, even if the original was.
        """
        return copy.deepcopy(self)

    def generate_target(self):
        self.target = self.copy()
        return self.target

    # Updating

    def update(self, dt: float = 0, recursive: bool = True) -> "Mobject":
        """Apply all updaters.

        Does nothing if updating is suspended.

        Parameters
        ----------
        dt
            The parameter ``dt`` to pass to the update functions. Usually this is the time in seconds since the last call of ``update``.
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
            ``True`` if at least one updater uses the ``dt`` parameter, ``False`` otherwise.

        See Also
        --------
        :meth:`get_time_based_updaters`

        """
        for updater in self.updaters:
            if "dt" in get_parameters(updater):
                return True
        return False

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
        return list(it.chain(*[sm.get_updaters() for sm in self.get_family()]))

    def add_updater(
        self,
        update_function: Updater,
        index: Optional[int] = None,
        call_updater: bool = False,
    ) -> "Mobject":
        """Add an update function to this mobject.

        Update functions, or updaters in short, are functions that are applied to the Mobject in every frame.

        Parameters
        ----------
        update_function
            The update function to be added.
            Whenever :meth:`update` is called, this update function gets called using ``self`` as the first parameter.
            The updater can have a second parameter ``dt``. If it uses this parameter, it gets called using a second value ``dt``, usually representing the time in seconds since the last call of :meth:`update`.
        index
            The index at which the new updater should be added in ``self.updaters``. In case ``index`` is ``None`` the updater will be added at the end.
        call_updater
            Wheather or not to call the updater initially. If ``True``, the updater will be called using ``dt=0``.

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
            # update_function(self, 0)
            self.update()
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
        All updaters from submobjects are removed, but only updaters of the given mobject are matched, not those of it's submobjects.

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
        for parent in self.parents:
            parent.resume_updating(recurse=False, call_updater=False)
        self.update(dt=0, recursive=recursive)
        return self

    # Transforming operations

    def apply_to_family(self, func: Callable[["Mobject"], None]) -> "Mobject":
        """Apply a function to ``self`` and every submobject with points recursively.

        Parameters
        ----------
        func
            The function to apply to each mobject. ``func`` gets passed the respective (sub)mobject as parameter.

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
            Vectors to shift by. If multiple vectors are given, they are added together.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See also
        --------
        :meth:`move_to`
        """

        if self.opengl:
            self.apply_points_function(
                lambda points: points + vectors[0],
                about_edge=None,
                works_on_bounding_box=True,
            )
        else:
            total_vector = reduce(op.add, vectors)
            for mob in self.family_members_with_points():
                mob.points = mob.points.astype("float")
                mob.points += total_vector
        return self

    def scale(self, scale_factor: float, **kwargs) -> "Mobject":
        """Scale the size by a factor.

        Default behavior is to scale about the center of the mobject.

        Parameters
        ----------
        scale_factor
            The scaling factor. Values 0 < |`scale_factor`| < 1 will shrink the mobject, 1 < |`scale_factor`| will increase it's size. A `scale_factor`<0 resuls in  additionally flipping by 180°.
        kwargs :
            Additional keyword arguments passed to :meth:`apply_points_function_about_point`.

        Returns
        -------
        :class:`Mobject`
            ``self``

        See also
        --------
        :meth:`move_to`

        """
        if self.opengl:
            self.apply_points_function(
                lambda points: scale_factor * points,
                works_on_bounding_box=True,
                **kwargs,
            )
        else:
            self.apply_points_function_about_point(
                lambda points: scale_factor * points, **kwargs
            )
        return self

    def rotate_about_origin(self, angle, axis=OUT):
        return self.rotate(angle, axis, about_point=ORIGIN)

    def rotate(self, angle, axis=OUT, **kwargs):
        if self.opengl:
            rot_matrix_T = rotation_matrix_transpose(angle, axis)
            self.apply_points_function(
                lambda points: np.dot(points, rot_matrix_T), **kwargs
            )
        else:
            rot_matrix = rotation_matrix(angle, axis)
            self.apply_points_function_about_point(
                lambda points: np.dot(points, rot_matrix.T), **kwargs
            )
        return self

    def flip(self, axis=UP, **kwargs):
        return self.rotate(TAU / 2, axis, **kwargs)

    def stretch(self, factor, dim, **kwargs):
        def func(points):
            points[:, dim] *= factor
            return points

        self.apply_function_to_points(func, works_on_bounding_box=True, **kwargs)
        return self

    def apply_function(self, function, **kwargs):
        # Default to applying matrix about the origin, not mobjects center
        if len(kwargs) == 0:
            kwargs["about_point"] = ORIGIN
        self.apply_function_to_points(
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

    def apply_function_to_points(self, function, *args, **kwargs):
        if self.opengl:
            self.apply_points_function(function, *args, **kwargs)
        else:
            self.apply_points_function_about_point(function, *args, **kwargs)

    def apply_matrix(self, matrix, **kwargs):
        # Default to applying matrix about the origin, not mobjects center
        if ("about_point" not in kwargs) and ("about_edge" not in kwargs):
            kwargs["about_point"] = ORIGIN
        full_matrix = np.identity(self.dim)
        matrix = np.array(matrix)
        full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        self.apply_function_to_points(
            lambda points: np.dot(points, full_matrix.T), **kwargs
        )
        return self

    def apply_complex_function(self, function, **kwargs):
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
        if self.opengl:
            for mob in self.get_family():
                for key in mob.data:
                    mob.data[key] = mob.data[key][::-1]
        else:
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
    def apply_points_function(
        self,
        func,
        about_point=None,
        about_edge=ORIGIN,
        works_on_bounding_box=False,
        **kwargs,
    ):
        if about_point is None and about_edge is not None:
            about_point = self.get_critical_point(about_edge)

        for mob in self.get_family():
            arrs = []
            if mob.has_points():
                arrs.append(mob.get_points())
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

    def apply_points_function_about_point(
        self, func, about_point=None, about_edge=None, **kwargs
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

    def rotate_in_place(self, angle, axis=OUT):
        # redundant with default behavior of rotate now.
        return self.rotate(angle, axis=axis)

    def scale_in_place(self, scale_factor, **kwargs):
        # Redundant with default behavior of scale now.
        return self.scale(scale_factor, **kwargs)

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
        """Move this mobject next to another mobject or coordinate.

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
        """Scales the mobject to fit a width while keeping height/depth proportional.

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
        """Stretches the mobject to fit a width, not keeping height/depth proportional.

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
        """Scales the mobject to fit a height while keeping width/depth proportional.

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
        """Stretches the mobject to fit a height, not keeping width/depth proportional.

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
        """Scales the mobject to fit a depth while keeping width/height proportional."""

        return self.rescale_to_fit(depth, 2, stretch=False, **kwargs)

    def stretch_to_fit_depth(self, depth, **kwargs):
        """Stretches the mobject to fit a depth, not keeping width/height proportional."""

        return self.rescale_to_fit(depth, 2, stretch=True, **kwargs)

    def set_coord(self, value, dim, direction=ORIGIN):
        curr = self.get_coord(dim, direction)
        shift_vect = np.zeros(self.dim)
        shift_vect[dim] = value - curr
        self.shift(shift_vect)
        return self

    def set_width(self, width, stretch=False, **kwargs):
        return self.rescale_to_fit(width, 0, stretch=stretch, **kwargs)

    def set_height(self, height, stretch=False, **kwargs):
        return self.rescale_to_fit(height, 1, stretch=stretch, **kwargs)

    def set_depth(self, depth, stretch=False, **kwargs):
        return self.rescale_to_fit(depth, 2, stretch=stretch, **kwargs)

    def set_x(self, x, direction=ORIGIN):
        return self.set_coord(x, 0, direction)

    def set_y(self, y, direction=ORIGIN):
        return self.set_coord(y, 1, direction)

    def set_z(self, z, direction=ORIGIN):
        return self.set_coord(z, 2, direction)

    def space_out_submobjects(self, factor=1.5, **kwargs):
        self.scale(factor, **kwargs)
        for submob in self.submobjects:
            submob.scale(1.0 / factor)
        return self

    def move_to(
        self, point_or_mobject, aligned_edge=ORIGIN, coor_mask=np.array([1, 1, 1])
    ):
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
                mobject.length_over_dim(dim_to_match), dim_to_match, stretch=False
            )
        self.shift(mobject.get_center() - self.get_center())
        return self

    def surround(self, mobject, dim_to_match=0, stretch=False, buff=MED_SMALL_BUFF):
        self.replace(mobject, dim_to_match, stretch)
        length = mobject.length_over_dim(dim_to_match)
        self.scale_in_place((length + buff) / length)
        return self

    def put_start_and_end_on(self, start, end):
        curr_start, curr_end = self.get_start_and_end()
        curr_vect = curr_end - curr_start
        if np.all(curr_vect == 0):
            raise Exception("Cannot position endpoints of closed loop")
        target_vect = np.array(end) - np.array(start)
        self.scale(
            get_norm(target_vect) / get_norm(curr_vect),
            about_point=curr_start,
        )
        self.rotate(
            angle_of_vector(target_vect) - angle_of_vector(curr_vect),
            about_point=curr_start,
        )
        self.shift(start - curr_start)
        return self

    # TODO: This should be moved to VM object for consistency
    def set_rgba_array(self, color=None, opacity=None, name="rgbas", recurse=True):
        if color is not None:
            rgbs = np.array([color_to_rgb(c) for c in listify(color)])
        if opacity is not None:
            opacities = listify(opacity)

        # Color only
        if color is not None and opacity is None:
            for mob in self.get_family(recurse):
                mob.data[name] = resize_array(mob.data[name], len(rgbs))
                mob.data[name][:, :3] = rgbs

        # Opacity only
        if color is None and opacity is not None:
            for mob in self.get_family(recurse):
                mob.data[name] = resize_array(mob.data[name], len(opacities))
                mob.data[name][:, 3] = opacities

        # Color and opacity
        if color is not None and opacity is not None:
            rgbas = np.array([[*rgb, o] for rgb, o in zip(*make_even(rgbs, opacities))])
            for mob in self.get_family(recurse):
                mob.data[name] = rgbas.copy()
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

    def set_color(
        self, color: Color = YELLOW_C, opacity: float = None, family: bool = True
    ):
        """Condition is function which takes in one arguments, (x, y, z).
        Here it just recurses to submobjects, but in subclasses this
        should be further implemented based on the the inner workings
        of color
        """
        self.set_rgba_array(color, opacity, recurse=False)
        if family:
            for submob in self.submobjects:
                submob.set_color(color, opacity, family=family)
        self.color = Color(color)
        return self

    def set_color_by_gradient(self, *colors):
        self.set_submobject_colors_by_gradient(*colors)
        return self

    def set_colors_by_radial_gradient(
        self, center=None, radius=1, inner_color=WHITE, outer_color=BLACK
    ):
        self.set_submobject_colors_by_radial_gradient(
            center, radius, inner_color, outer_color
        )
        return self

    def set_submobject_colors_by_gradient(self, *colors):
        if len(colors) == 0:
            raise ValueError("Need at least one color")
        elif len(colors) == 1:
            return self.set_color(*colors)

        mobs = self.family_members_with_points()
        # mobs = self.submobjects
        new_colors = color_gradient(colors, len(mobs))

        for mob, color in zip(mobs, new_colors):
            mob.set_color(color, family=False)
        return self

    def set_submobject_colors_by_radial_gradient(
        self, center=None, radius=1, inner_color=WHITE, outer_color=BLACK
    ):
        if center is None:
            center = self.get_center()

        for mob in self.family_members_with_points():
            t = get_norm(mob.get_center() - center) / radius
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
        if self.opengl:
            self.set_opacity(1.0 - darkness, recurse=family)
            return
        if family:
            for submob in self.submobjects:
                submob.fade(darkness, family)
        return self

    def get_color(self):
        if self.opengl:
            return rgb_to_hex(self.data["rgbas"][0, :3])
        return self.color

    def get_gloss(self):
        return self.gloss

    def set_gloss(self, gloss, recurse=True):
        for mob in self.get_family(recurse):
            mob.gloss = gloss
        return self

    def get_shadow(self):
        return self.shadow

    def set_shadow(self, shadow, recurse=True):
        for mob in self.get_family(recurse):
            mob.shadow = shadow
        return self

    ##

    def save_state(self):
        if hasattr(self, "saved_state"):
            # Prevent exponential growth of data
            self.saved_state = None
        self.saved_state = self.copy()

        return self

    def restore(self):
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
        """Picture a box bounding the mobject.  Such a box has
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
                all_points, dim=dim, key=direction[dim]
            )
        return result

    # Pseudonyms for more general get_critical_point method

    def get_edge_center(self, direction):
        return self.get_critical_point(direction)

    def get_corner(self, direction):
        return self.get_critical_point(direction)

    def get_center(self):
        return self.get_critical_point(np.zeros(self.dim))

    def get_center_of_mass(self):
        return np.apply_along_axis(np.mean, 0, self.get_all_points())

    def get_boundary_point(self, direction):
        if self.opengl:
            all_points = self.get_all_points()
            boundary_directions = all_points - self.get_center()
            norms = np.linalg.norm(boundary_directions, axis=1)
            boundary_directions /= np.repeat(norms, 3).reshape((len(norms), 3))
            index = np.argmax(np.dot(boundary_directions, np.array(direction).T))
            return all_points[index]

        all_points = self.get_points_defining_boundary()
        index = np.argmax(np.dot(all_points, np.array(direction).T))
        return all_points[index]

    def get_continuous_bounding_box_point(self, direction):
        dl, center, ur = self.get_bounding_box()
        corner_vect = ur - center
        return center + direction / np.max(
            np.abs(
                np.true_divide(
                    direction,
                    corner_vect,
                    out=np.zeros(len(direction)),
                    where=((corner_vect) != 0),
                )
            )
        )

    def get_top(self):
        return self.get_edge_center(UP)

    def get_bottom(self):
        return self.get_edge_center(DOWN)

    def get_right(self):
        return self.get_edge_center(RIGHT)

    def get_left(self):
        return self.get_edge_center(LEFT)

    def get_zenith(self):
        return self.get_edge_center(OUT)

    def get_nadir(self):
        return self.get_edge_center(IN)

    def length_over_dim(self, dim):
        if self.opengl:
            bb = self.get_bounding_box()
            return abs((bb[2] - bb[0])[dim])

        return self.reduce_across_dimension(
            np.max, np.max, dim
        ) - self.reduce_across_dimension(np.min, np.min, dim)

    def get_coord(self, dim, direction=ORIGIN):
        """Meant to generalize get_x, get_y, get_z"""
        return self.get_extremum_along_dim(dim=dim, key=direction[dim])

    def get_width(self):
        return self.length_over_dim(0)

    def get_height(self):
        return self.length_over_dim(1)

    def get_depth(self):
        return self.length_over_dim(2)

    def get_x(self, direction=ORIGIN):
        return self.get_coord(0, direction)

    def get_y(self, direction=ORIGIN):
        return self.get_coord(1, direction)

    def get_z(self, direction=ORIGIN):
        return self.get_coord(2, direction)

    def get_start(self):
        self.throw_error_if_no_points()
        return np.array(self.points[0])

    def get_end(self):
        self.throw_error_if_no_points()
        return np.array(self.points[-1])

    def get_start_and_end(self):
        return self.get_start(), self.get_end()

    def point_from_proportion(self, alpha):
        points = self.get_points()
        i, subalpha = integer_interpolate(0, len(points) - 1, alpha)
        return interpolate(points[i], points[i + 1], subalpha)

    def pfp(self, alpha):
        """Abbreviation fo point_from_proportion"""
        return self.point_from_proportion(alpha)

    def get_pieces(self, n_pieces):
        template = self.copy()
        template.set_submobjects([])
        alphas = np.linspace(0, 1, n_pieces + 1)
        return Group(
            *[
                template.copy().pointwise_become_partial(self, a1, a2)
                for a1, a2 in zip(alphas[:-1], alphas[1:])
            ]
        )

    def get_z_index_reference_point(self):
        # TODO, better place to define default z_index_group?
        z_index_group = getattr(self, "z_index_group", self)
        return z_index_group.get_center()

    def has_points(self):
        return len(self.points) > 0

    def has_no_points(self):
        return not self.has_points()

    def set_uniforms(self, uniforms):
        for key in uniforms:
            self.uniforms[key] = uniforms[key]  # Copy?
        return self

    # Match other mobject properties

    def match_color(self, mobject):
        return self.set_color(mobject.get_color())

    def match_dim_size(self, mobject, dim, **kwargs):
        return self.rescale_to_fit(mobject.length_over_dim(dim), dim, **kwargs)

    def match_width(self, mobject, **kwargs):
        return self.match_dim_size(mobject, 0, **kwargs)

    def match_height(self, mobject, **kwargs):
        return self.match_dim_size(mobject, 1, **kwargs)

    def match_depth(self, mobject, **kwargs):
        return self.match_dim_size(mobject, 2, **kwargs)

    def match_coord(self, mobject, dim, direction=ORIGIN):
        return self.set_coord(
            mobject.get_coord(dim, direction),
            dim=dim,
            direction=direction,
        )

    def match_x(self, mobject, direction=ORIGIN):
        return self.match_coord(mobject, 0, direction)

    def match_y(self, mobject, direction=ORIGIN):
        return self.match_coord(mobject, 1, direction)

    def match_z(self, mobject, direction=ORIGIN):
        return self.match_coord(mobject, 2, direction)

    def align_to(self, mobject_or_point, direction=ORIGIN, alignment_vect=UP):
        """Examples:
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
        if self.opengl:
            if recurse:
                return self.family
            else:
                return [self]
        else:
            sub_families = list(map(Mobject.get_family, self.submobjects))
            all_mobjects = [self] + list(it.chain(*sub_families))
            return remove_list_redundancies(all_mobjects)

    def assemble_family(self):
        if not self.opengl:
            return
        sub_families = (sm.get_family() for sm in self.submobjects)
        self.family = [self, *it.chain(*sub_families)]
        self.refresh_bounding_box()
        for parent in self.parents:
            parent.assemble_family()
        return self

    def family_members_with_points(self):
        return [m for m in self.get_family() if m.has_points()]

    def arrange(
        self,
        direction=RIGHT,
        buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        center=True,
        **kwargs,
    ):
        """Sorts mobjects next to each other on screen.

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
        n_rows=None,
        n_cols=None,
        buff=None,
        h_buff=None,
        v_buff=None,
        buff_ratio=None,
        h_buff_ratio=0.5,
        v_buff_ratio=0.5,
        aligned_edge=ORIGIN,
        fill_rows_first=True,
    ):
        submobs = self.submobjects
        if n_rows is None and n_cols is None:
            n_rows = int(np.sqrt(len(submobs)))
        if n_rows is None:
            n_rows = len(submobs) // n_cols
        elif n_cols is None:
            n_cols = len(submobs) // n_rows

        if buff is not None:
            h_buff = buff
            v_buff = buff
        else:
            if buff_ratio is not None:
                v_buff_ratio = buff_ratio
                h_buff_ratio = buff_ratio
            if h_buff is None:
                h_buff = h_buff_ratio * self[0].get_width()
            if v_buff is None:
                v_buff = v_buff_ratio * self[0].get_height()

        x_unit = h_buff + max([sm.get_width() for sm in submobs])
        y_unit = v_buff + max([sm.get_height() for sm in submobs])

        for index, sm in enumerate(submobs):
            if fill_rows_first:
                x, y = index % n_cols, index // n_cols
            else:
                x, y = index // n_rows, index % n_rows
            sm.move_to(ORIGIN, aligned_edge)
            sm.shift(x * x_unit * RIGHT + y * y_unit * DOWN)
        self.center()
        return self

    def get_grid(self, n_rows, n_cols, height=None, **kwargs):
        """
        Returns a new mobject containing multiple copies of this one
        arranged in a grid
        """
        grid = self.get_group_class()(*(self.copy() for n in range(n_rows * n_cols)))
        grid.arrange_in_grid(n_rows, n_cols, **kwargs)
        if height is not None:
            grid.set_height(height)
        return grid

    def sort(self, point_to_num_func=lambda p: p[0], submob_func=None):
        if submob_func is None:
            submob_func = lambda m: point_to_num_func(m.get_center())
        self.submobjects.sort(key=submob_func)
        return self

    def shuffle(self, recursive=False):
        if recursive:
            for submob in self.submobjects:
                submob.shuffle(recursive=True)
        random.shuffle(self.submobjects)

    def invert(self, recursive=False):
        if recursive:
            for submob in self.submobjects:
                submob.invert(recursive=True)
        list.reverse(self.submobjects)

    # Just here to keep from breaking old scenes.
    def arrange_submobjects(self, *args, **kwargs):
        return self.arrange(*args, **kwargs)

    def sort_submobjects(self, *args, **kwargs):
        return self.sort(*args, **kwargs)

    def shuffle_submobjects(self, *args, **kwargs):
        return self.shuffle(*args, **kwargs)

    # Alignment
    def align_data(self, mobject):
        if self.opengl:
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
        else:
            self.null_point_align(mobject)
            self.align_submobjects(mobject)
            self.align_points(mobject)
            # Recurse
            for m1, m2 in zip(self.submobjects, mobject.submobjects):
                m1.align_data(m2)

    def get_point_mobject(self, center=None):
        """The simplest mobject to be transformed to or from self.
        Should by a point of the appropriate type
        """
        msg = f"get_point_mobject not implemented for {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def align_points(self, mobject):
        if self.opengl:
            max_len = max(self.get_num_points(), mobject.get_num_points())
            for mob in (self, mobject):
                mob.resize_points(max_len, resize_func=resize_preserving_order)
            return self
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
        """If a mobject with points is being aligned to
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
        copy.set_submobjects([])
        self.reset_points()
        self.add(copy)
        return self

    def add_n_more_submobjects(self, n):
        if n == 0:
            return

        curr = len(self.submobjects)
        if curr == 0:
            # If empty, simply add n point mobjects
            if self.opengl:
                null_mob = self.copy()
                null_mob.set_points([self.get_center()])
                self.set_submobjects([null_mob.copy() for k in range(n)])
            else:
                self.submobjects = [self.get_point_mobject() for k in range(n)]
            return self

        target = curr + n
        # TODO, factor this out to utils so as to reuse
        # with VMobject.insert_n_curves
        repeat_indices = (np.arange(target) * curr) // target
        split_factors = [sum(repeat_indices == i) for i in range(curr)]
        new_submobs = []
        for submob, sf in zip(self.submobjects, split_factors):
            new_submobs.append(submob)
            for _ in range(1, sf):
                if self.opengl:
                    new_submob = submob.copy()
                    # If the submobject is at all transparent, then
                    # make the copy completely transparent
                    if submob.get_opacity() < 1:
                        new_submob.set_opacity(0)
                new_submobs.append(submob.copy().fade(1))
        self.set_submobjects(new_submobs)
        return self

    def set_opacity(self, opacity, recurse=True):
        self.set_rgba_array(color=None, opacity=opacity, recurse=False)
        if recurse:
            for submob in self.submobjects:
                submob.set_opacity(opacity, recurse=True)
        return self

    def get_opacity(self):
        return self.data["rgbas"][0, 3]

    def repeat_submobject(self, submob):
        return submob.copy()

    def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
        """Turns this mobject into an interpolation between ``mobject1``
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
        if self.opengl:
            for key in self.data:
                if key in self.locked_data_keys:
                    continue
                if len(self.data[key]) == 0:
                    continue
                if key not in mobject1.data or key not in mobject2.data:
                    continue

                if key in ("points", "bounding_box"):
                    func = path_func
                else:
                    func = interpolate

                self.data[key][:] = func(mobject1.data[key], mobject2.data[key], alpha)
            for key in self.uniforms:
                self.uniforms[key] = interpolate(
                    mobject1.uniforms[key], mobject2.uniforms[key], alpha
                )
        else:
            self.points = path_func(mobject1.points, mobject2.points, alpha)
            self.interpolate_color(mobject1, mobject2, alpha)
        return self

    def interpolate_color(self, mobject1, mobject2, alpha):
        raise NotImplementedError("Please override in a child class.")

    def pointwise_become_partial(self, mobject, a, b):
        raise NotImplementedError("Please override in a child class.")

    def align_family(self, mobject):
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

    def become(self, mobject: "Mobject", copy_submobjects: bool = True):
        """Edit points, colors and submobjects to be identical
        to another mobject

        Examples
        --------
        .. manim:: BecomeScene

            class BecomeScene(Scene):
                def construct(self):
                    circ = Circle(fill_color=RED)
                    square = Square(fill_color=BLUE)
                    self.add(circ)
                    self.wait(0.5)
                    circ.become(square)
                    self.wait(0.5)
        """
        if self.opengl:
            self.align_family(mobject)
            for sm1, sm2 in zip(self.get_family(), mobject.get_family()):
                sm1.set_data(sm2.data)
                sm1.set_uniforms(sm2.uniforms)
            self.refresh_bounding_box(recurse_down=True)
        else:
            self.align_data(mobject)
            for sm1, sm2 in zip(self.get_family(), mobject.get_family()):
                sm1.points = np.array(sm2.points)
                sm1.interpolate_color(sm1, sm2, 1)
        return self

    # Locking data

    def lock_data(self, keys):
        """
        To speed up some animations, particularly transformations,
        it can be handy to acknowledge which pieces of data
        won't change during the animation so that calls to
        interpolate can skip this, and so that it's not
        read into the shader_wrapper objects needlessly
        """
        if self.updaters:
            return
        # Be sure shader data has most up to date information
        self.refresh_shader_data()
        self.locked_data_keys = set(keys)

    def lock_matching_data(self, mobject1, mobject2):
        for sm, sm1, sm2 in zip(
            self.get_family(), mobject1.get_family(), mobject2.get_family()
        ):
            keys = sm.data.keys() & sm1.data.keys() & sm2.data.keys()
            sm.lock_data(
                list(
                    filter(
                        lambda key: np.all(sm1.data[key] == sm2.data[key]),
                        keys,
                    )
                )
            )
        return self

    def unlock_data(self):
        for mob in self.get_family():
            mob.locked_data_keys = set()

    # Errors
    def throw_error_if_no_points(self):
        if self.has_no_points():
            caller_name = sys._getframe(1).f_code.co_name
            raise Exception(
                f"Cannot call Mobject.{caller_name} for a Mobject with no points"
            )

    # Operations touching shader uniforms

    @affects_shader_info_id
    def fix_in_frame(self):
        self.uniforms["is_fixed_in_frame"] = 1.0
        return self

    @affects_shader_info_id
    def unfix_from_frame(self):
        self.uniforms["is_fixed_in_frame"] = 0.0
        return self

    @affects_shader_info_id
    def apply_depth_test(self):
        self.depth_test = True
        return self

    @affects_shader_info_id
    def deactivate_depth_test(self):
        self.depth_test = False
        return self

    # Shader code manipulation

    def replace_shader_code(self, old, new):
        # TODO, will this work with VMobject structure, given
        # that it does not simpler return shader_wrappers of
        # family?
        for wrapper in self.get_shader_wrapper_list():
            wrapper.replace_code(old, new)
        return self

    def set_color_by_code(self, glsl_code):
        """
        Takes a snippet of code and inserts it into a
        context which has the following variables:
        vec4 color, vec3 point, vec3 unit_normal.
        The code should change the color variable
        """
        self.replace_shader_code("///// INSERT COLOR FUNCTION HERE /////", glsl_code)
        return self

    # For shader data

    # def refresh_shader_wrapper_id(self):
    #     self.shader_wrapper.refresh_id()
    #     return self

    def get_shader_wrapper(self):
        from ..renderer.shader_wrapper import ShaderWrapper

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

    def get_shader_wrapper_list(self):
        shader_wrappers = it.chain(
            [self.get_shader_wrapper()],
            *[sm.get_shader_wrapper_list() for sm in self.submobjects],
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

    def append_points(self, new_points):
        self.points = np.vstack([self.points, new_points])
        self.refresh_bounding_box()
        return self

    def clear_points(self):
        self.resize_points(0)

    def check_data_alignment(self, array, data_key):
        # Makes sure that self.data[key] can be brodcast into
        # the given array, meaning its length has to be either 1
        # or the length of the array
        d_len = len(self.data[data_key])
        if d_len != 1 and d_len != len(array):
            self.data[data_key] = resize_with_interpolation(
                self.data[data_key], len(array)
            )
        return self

    def get_resized_shader_data_array(self, length):
        # If possible, try to populate an existing array, rather
        # than recreating it each frame
        shader_data = np.zeros(len(self.points), dtype=self.shader_dtype)
        return shader_data

    def read_data_to_shader(self, shader_data, shader_data_key, data_key):
        if data_key in self.locked_data_keys:
            return
        self.check_data_alignment(shader_data, data_key)
        shader_data[shader_data_key] = self.data[data_key]

    def get_shader_data(self):
        shader_data = self.get_resized_shader_data_array(self.get_num_points())
        self.read_data_to_shader(shader_data, "point", "points")
        return shader_data

    def refresh_shader_data(self):
        self.get_shader_data()

    def get_shader_uniforms(self):
        return self.uniforms

    def get_shader_vert_indices(self):
        return self.shader_indices

    # Event Handlers
    """
        Event handling follows the Event Bubbling model of DOM in javascript.
        Return false to stop the event bubbling.
        To learn more visit https://www.quirksmode.org/js/events_order.html

        Event Callback Argument is a callable function taking two arguments:
            1. Mobject
            2. EventData
    """

    def init_event_listners(self):
        self.event_listners = []

    def add_event_listner(self, event_type, event_callback):
        event_listner = EventListner(self, event_type, event_callback)
        self.event_listners.append(event_listner)
        EVENT_DISPATCHER.add_listner(event_listner)
        return self

    def remove_event_listner(self, event_type, event_callback):
        event_listner = EventListner(self, event_type, event_callback)
        while event_listner in self.event_listners:
            self.event_listners.remove(event_listner)
        EVENT_DISPATCHER.remove_listner(event_listner)
        return self

    def clear_event_listners(self, recurse=True):
        self.event_listners = []
        if recurse:
            for submob in self.submobjects:
                submob.clear_event_listners(recurse=recurse)
        return self

    def get_event_listners(self):
        return self.event_listners

    def get_family_event_listners(self):
        return list(it.chain(*[sm.get_event_listners() for sm in self.get_family()]))

    def get_has_event_listner(self):
        return any(mob.get_event_listners() for mob in self.get_family())

    def add_mouse_motion_listner(self, callback):
        self.add_event_listner(EventType.MouseMotionEvent, callback)

    def remove_mouse_motion_listner(self, callback):
        self.remove_event_listner(EventType.MouseMotionEvent, callback)

    def add_mouse_press_listner(self, callback):
        self.add_event_listner(EventType.MousePressEvent, callback)

    def remove_mouse_press_listner(self, callback):
        self.remove_event_listner(EventType.MousePressEvent, callback)

    def add_mouse_release_listner(self, callback):
        self.add_event_listner(EventType.MouseReleaseEvent, callback)

    def remove_mouse_release_listner(self, callback):
        self.remove_event_listner(EventType.MouseReleaseEvent, callback)

    def add_mouse_drag_listner(self, callback):
        self.add_event_listner(EventType.MouseDragEvent, callback)

    def remove_mouse_drag_listner(self, callback):
        self.remove_event_listner(EventType.MouseDragEvent, callback)

    def add_mouse_scroll_listner(self, callback):
        self.add_event_listner(EventType.MouseScrollEvent, callback)

    def remove_mouse_scroll_listner(self, callback):
        self.remove_event_listner(EventType.MouseScrollEvent, callback)

    def add_key_press_listner(self, callback):
        self.add_event_listner(EventType.KeyPressEvent, callback)

    def remove_key_press_listner(self, callback):
        self.remove_event_listner(EventType.KeyPressEvent, callback)

    def add_key_release_listner(self, callback):
        self.add_event_listner(EventType.KeyReleaseEvent, callback)

    def remove_key_release_listner(self, callback):
        self.remove_event_listner(EventType.KeyReleaseEvent, callback)

    # About z-index
    def set_z_index(self, z_index_value: Union[int, float]):
        """Sets the mobject's :attr:`z_index` to the value specified in `z_index_value`.

        Parameters
        ----------
        z_index_value
            The new value of :attr:`z_index` set.

        Returns
        -------
        :class:`Mobject`
            The Mobject itself, after :attr:`z_index` is set. (Returns `self`.)
        """
        self.z_index = z_index_value
        return self

    def set_z_index_by_z_coordinate(self):
        """Sets the mobject's z coordinate to the value of :attr:`z_index`.

        Returns
        -------
        :class:`Mobject`
            The Mobject itself, after :attr:`z_index` is set. (Returns `self`.)
        """
        z_coord = self.get_center()[-1]
        self.set_z_index(z_coord)
        return self


class Group(Mobject):
    """Groups together multiple Mobjects."""

    def __init__(self, *mobjects, **kwargs):
        Mobject.__init__(self, **kwargs)
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
                "Animation arguments must be passed before accessing methods and can only be passed once"
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
                "overridden animations"
            )

        def update_target(*method_args, **method_kwargs):
            if has_overridden_animation:
                self.overridden_animation = method._override_animate(
                    self.mobject, *method_args, **method_kwargs
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

        from manim import Circle, Scene, Create, Text, Uncreate, VGroup

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
            def _clear_content_animation(self):
                anim = Uncreate(self.content)
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
