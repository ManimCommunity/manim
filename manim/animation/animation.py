from copy import deepcopy

import attr
import numpy as np
import typing

from ..mobject.mobject import Mobject
from ..utils.config_ops import digest_config
from ..utils.rate_functions import smooth
from ..utils.dataclasses import dclass

if typing.TYPE_CHECKING:
    from ..scene.scene import Scene


DEFAULT_ANIMATION_RUN_TIME = 1.0
DEFAULT_ANIMATION_LAG_RATIO = 0

@dclass
class Animation(object):
    """Represents a generic animation.

    Attributes
    ----------
    mobject : :class:`~.Mobject`
        The Mobject which will go through an animation.
    run_time : :class:`float`
        How long this animation will run for.
    rate_func : Callable[[:class:`float`, :class:`float`], :class:`float`]
        Function that determines the rate of the animation.
    name : Optional[:class:`str`]
        Name of the animation.
    remover : :class:`bool`
        If `True`, this animation removes a Mobject from the screen.
    lag_ratio : Union[:class:`int`, :class:`float`]
        - If 0, the animation is applied to all submobjects at the same time.
        - If 1, it is applied to each successively.
        - If 0 < lag_ratio < 1, it's applied to each with lagged start times.
    """
    mobject: Mobject = attr.ib(validator=lambda x: isinstance(x, Mobject))
    run_time: float = DEFAULT_ANIMATION_RUN_TIME
    rate_func: typing.Union[typing.Callable[[float, float], float], typing.Callable[[float], float]] \
        = smooth
    name: typing.Optional[str] = None
    remover: bool = False
    suspend_mobject_updating: bool = True

    CONFIG = {
        "run_time": DEFAULT_ANIMATION_RUN_TIME,
        "rate_func": smooth,
        "name": None,
        # Does this animation add or remove a mobject form the screen
        "remover": False,
        # If 0, the animation is applied to all submobjects
        # at the same time
        # If 1, it is applied to each successively.
        # If 0 < lag_ratio < 1, its applied to each
        # with lagged start times
        "lag_ratio": DEFAULT_ANIMATION_LAG_RATIO,
        "suspend_mobject_updating": True,
    }

    def __str__(self):
        if self.name:
            return self.name
        return self.__class__.__name__ + str(self.mobject)

    def begin(self) -> None:
        """Begins the animation."""
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
        """Finishes the animation."""
        self.interpolate(1)
        if self.suspend_mobject_updating:
            self.mobject.resume_updating()

    def clean_up_from_scene(self, scene: "Scene") -> None:
        """Cleans the scene up after this animation has been concluded.

        Parameters
        ----------
        scene : :class:`~.Scene`
            The scene to be cleaned.
        """
        if self.is_remover():
            scene.remove(self.mobject)

    def create_starting_mobject(self) -> Mobject:
        """Copies the Mobject in order to keep track of how and where it starts.

        Returns
        -------
        :class:`~.Mobject`
            The copy of the animation's Mobject.
        """
        # Keep track of where the mobject starts
        return self.mobject.copy()

    def get_all_mobjects(self) -> typing.Tuple[Mobject, Mobject]:
        """Returns the current state of the Mobject and its starting value.

        Notes
        -----
        This ordering must match the ordering of arguments to :meth:`interpolate_submobject`.
        """
        return self.mobject, self.starting_mobject

    def get_all_families_zipped(self) -> typing.Iterator[typing.Tuple[Mobject, ...]]:
        """Returns the families of all Mobjects involved in this animation.

        Returns
        -------
        Iterator[Tuple[:class:`~.Mobject`, ...]]
            The families (an iterator - made with :func:`zip` - whose 'elements' are the
            respective families of the Mobjects)

        See Also
        --------
        :meth:`get_all_mobjects`
        """
        return zip(
            *[mob.family_members_with_points() for mob in self.get_all_mobjects()]
        )

    def get_all_families_zipped(self):
        return zip(
            *[mob.family_members_with_points() for mob in self.get_all_mobjects()]
        )

    def update_mobjects(self, dt: float) -> None:
        """Updates things like :attr:`starting_mobject`, and (for
        :class:`~.Transform` s) :attr:`~.Transform.target_mobject`.

        Note that, since typically (always?) `self.mobject` will have its updating suspended during the
        animation, this will do nothing to it.

        Parameters
        ----------
        dt : :class:`float`
            The timespan for which to update the :class:`~.Mobject` (see :meth:`~.Mobject.update`).

        See Also
        --------
        :meth:`~.Mobject.update`
        """
        for mob in self.get_all_mobjects_to_update():
            mob.update(dt)

    def get_all_mobjects_to_update(self) -> typing.List[Mobject]:
        """Returns the Mobjects to be updated, which excludes `self.mobject`.

        Returns
        -------
        List[:class:`~.Mobject`]
        """
        # The surrounding scene typically handles
        # updating of self.mobject.  Besides, in
        # most cases its updating is suspended anyway
        return list(filter(lambda m: m is not self.mobject, self.get_all_mobjects()))

    def copy(self) -> "Animation":
        """Copies this animation.

        Returns
        -------
        :class:`Animation`
            The generated copy.
        """
        return deepcopy(self)

    def update_config(self, **kwargs):
        digest_config(self, kwargs)
        return self

    # Methods for interpolation, the mean of an Animation
    def interpolate(self, alpha):
        alpha = np.clip(alpha, 0, 1)
        self.interpolate_mobject(self.rate_func(alpha))

    def update(self, alpha):
        """
        This method shouldn't exist, but it's here to
        keep many old scenes from breaking
        """
        self.interpolate(alpha)

    def interpolate_mobject(self, alpha):
        families = list(self.get_all_families_zipped())
        for i, mobs in enumerate(families):
            sub_alpha = self.get_sub_alpha(alpha, i, len(families))
            self.interpolate_submobject(*mobs, sub_alpha)

    def interpolate_submobject(self, submobject, starting_sumobject, alpha):
        # Typically ipmlemented by subclass
        pass

    def get_sub_alpha(self, alpha, index, num_submobjects):
        # TODO, make this more understanable, and/or combine
        # its functionality with AnimationGroup's method
        # build_animations_with_timings
        lag_ratio = self.lag_ratio
        full_length = (num_submobjects - 1) * lag_ratio + 1
        value = alpha * full_length
        lower = index * lag_ratio
        return np.clip((value - lower), 0, 1)

    # Getters and setters
    def set_run_time(self, run_time):
        self.run_time = run_time
        return self

    def get_run_time(self):
        return self.run_time

    def set_rate_func(self, rate_func):
        self.rate_func = rate_func
        return self

    def get_rate_func(self):
        return self.rate_func

    def set_name(self, name):
        self.name = name
        return self

    def is_remover(self):
        return self.remover
