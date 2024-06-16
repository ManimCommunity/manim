from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from manim.mobject.mobject import Mobject
    from manim.opengl.shader import Object3D


__all__ = [
    "MobjectTimeBasedUpdater",
    "MobjectNonTimeBasedUpdater",
    "MobjectUpdater",
    "MeshTimeBasedUpdater",
    "MeshNonTimeBasedUpdater",
    "MeshUpdater",
    "SceneUpdater",
    "MobjectUpdaterWrapper",
    "MeshUpdaterWrapper",
]


MobjectTimeBasedUpdater: TypeAlias = Callable[["Mobject", float], "Mobject"]
MobjectNonTimeBasedUpdater: TypeAlias = Callable[["Mobject"], "Mobject"]
MobjectUpdater: TypeAlias = MobjectNonTimeBasedUpdater | MobjectTimeBasedUpdater

MeshTimeBasedUpdater: TypeAlias = Callable[["Object3D", float], "Object3D"]
MeshNonTimeBasedUpdater: TypeAlias = Callable[["Object3D"], "Object3D"]
MeshUpdater: TypeAlias = MeshNonTimeBasedUpdater | MeshTimeBasedUpdater

SceneUpdater: TypeAlias = Callable[[float], None]


class AbstractUpdaterWrapper:
    """Wraps an :class:`Updater` function, inspects its signature and
    calculates whether it's time-based or not.

    If it has a single parameter (with no default value), it's considered
    non-time-based: it doesn't depend on time.

    If it has two parameters (with no default values), it's considered
    time-based: it depends on time, and the affected Mobject has a change
    on every frame which depends on the frame's duration dt.

    .. note::
        It's not mandatory that the parameters are named ``mob`` and ``dt``.

    **Only parameters with no default values are considered in when determining
    whether the updater is time-based or not.** For example, an updater
    ``lambda mob, rate=5: ...`` is considered non-time-based since the 2nd
    parameter ``rate`` has a default value of 5. **This allows for passing
    functions with more than 2 parameters, as long as the extra parameters have
    default values.**

    A ``ValueError`` is raised if a function is passed which has 0 or more than
    2 parameters with no default values.

    When passing an instance method, the first parameter `self` is excluded
    from the count. When passing a class method, the first parameter `cls` is
    also excluded.

    .. note ::
        It is fine to call the 1st parameter ``self`` if the updater is not an
        instance method: it will still be counted as a parameter. The rule
        above only applies for instance methods. For example,
        ``lambda self: self.move_to(square)`` is a valid non-time-based
        updater, and ``lambda self, dt: self.rotate(dt)`` is time-based.

    .. warning::
        Do **NOT** name the 1st parameter ``cls`` if the function is not a class
        method.

    Attributes
    ----------
    updater
        An updater function, whose first parameter is a Mobject and might
        optionally have a second parameter which should be a float
        representing a time change dt.

    Raises
    ------
    ValueError
        If an updater is passed with 0 or more than 2 parameters with no
        default values.
    """

    __slots__ = ["updater", "is_time_based"]

    def __init__(self, updater: MobjectUpdater | MeshUpdater):
        self.updater = updater

        signature = inspect.signature(updater)
        parameters = [str(param) for param in signature.parameters.values()]

        for i, param in enumerate(parameters):
            # Stop when finding **kwargs or parameters with default values
            if param.startswith("**") or "=" in param:
                num_non_default_parameters = i
                break
        else:
            num_non_default_parameters = len(parameters)

        if num_non_default_parameters == 0:
            self._raise_error(0)

        # If this is a method being called from an instance, exclude the 1st
        # parameter if it's called "self"
        if inspect.ismethod(updater) and parameters[0] == "self":
            num_non_default_parameters -= 1
        # Exclude the "cls" parameter from class methods
        if parameters[0] == "cls":
            num_non_default_parameters -= 1

        # Handle functions containing *args, assuming that all can be passed
        # a 2nd parameter dt
        if 1 <= num_non_default_parameters <= 3 and parameters[-1].startswith("*"):
            num_non_default_parameters = 2

        if num_non_default_parameters == 1:
            self.is_time_based = False
        elif num_non_default_parameters == 2:
            self.is_time_based = True
        else:
            self._raise_error(num_non_default_parameters)

    def _raise_error(self, num_non_default_parameters: int):
        updater_name = self.updater.__code__.co_qualname
        signature = str(inspect.signature(self.updater))
        full_name = updater_name + signature

        if num_non_default_parameters == 0:
            num_non_default_parameters = "no"

        raise ValueError(
            "An updater function must accept either 1 or 2 parameters without "
            "default values (not including 'self' or 'cls' for methods), but "
            f"the function {full_name} has {num_non_default_parameters} such "
            "parameters."
        )


class MobjectUpdaterWrapper(AbstractUpdaterWrapper):
    def __init__(self, updater: MobjectUpdater):
        super().__init__(updater)


class MeshUpdaterWrapper(AbstractUpdaterWrapper):
    def __init__(self, updater: MeshUpdater):
        super().__init__(updater)
