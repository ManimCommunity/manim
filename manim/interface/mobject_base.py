
import numpy as np

from functools import partialmethod
from typing import Callable, Dict, Iterable, Type

from ..animation.animation_utils import _AnimationBuilder
from ..utils.exceptions import MultiAnimationOverrideException

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
        cls.animation_overrides: Dict[
            Type["Animation"],
            Callable[["MobjectBase"], "Animation"],
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
            return Group(self.submobjects.__getitem__(value))
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

    
    ### Mobject points and related methods ###

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, pts):
        self._points = pts
        self.invalidate_bounding_box()

    def has_points(self):
        """Checks whether this mobject has any points set.
        """
        return len(self.points) > 0

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            self.update_bounding_box()
        return self._bounding_box

    def update_bounding_box(self):
        if self.has_submobjects():
            points = np.vstack(
                [
                    self.points,
                    *(
                        mob.bounding_box
                        for mob in self.get_family()[1:]
                        if mob.has_points()
                    )
                ]
            )

        else:
            points = self.points

        if len(points) == 0:
            points = np.zeros((1,3))

        min_point = points.min(0)
        max_point = points.max(0)
        mid_point = (min_point + max_point)/2
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
        return self.bounding_box[1]
