from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Concatenate

    from manim.mobject.opengl.opengl_mobject import OpenGLMobject

M = TypeVar("M", bound="OpenGLMobject")


class Updater(Generic[M]):
    """The internal representation of an updater.

    Note that you can pass a raw instance of this class into the :meth:`.OpenGLMobject.add_updater`
    and :meth:`.OpenGLMobject.add_dt_updater` methods. This is to allow for decorators in the
    future such as ``@affects_points`` to return instances of this class.

    Parameters
    ----------
        updater: the update function
        dt: whether the update function requires the time step `dt` as an argument

    Examples
    --------

        .. code-block:: pycon

            >>> from manim import Updater, Square
            >>> def update(mob):
            ...     print("Updating mob")
            >>> updater = Updater(update)
            >>> updater(Square(), dt=0)  # still needs dt argument
            'Updating mob'
            >>> def dt_update(mob, dt):
            ...     print("Updating mob with dt")
            >>> dt_updater = Updater(dt_update, dt=True)
            >>> dt_updater(Square(), dt=0.1)
            'Updating mob with dt'


        .. code-block:: pycon

            >>> from manim import Updater, Square
            >>> def update(mob):
            ...     print("Updating mob")
            >>> updater = Updater(update)
            >>> s = Square()
            >>> s.add_updater(updater)
            >>> s.update(dt=0)
            'Updating mob'
    """

    def __init__(
        self,
        updater: Callable[Concatenate[M, ...], object]
        | Callable[Concatenate[M, float, ...], object],
        *,
        dt: bool = False,
    ) -> None:
        self._updater = updater
        self.needs_dt = dt
        # TODO: add fields such as "affects_points", "affects_color", etc.
        # so that updater calls can be optimized

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Updater):
            return self._updater == other._updater
        return self._updater == other

    def __hash__(self) -> int:
        return hash((self._updater, self.needs_dt))

    def __call__(self, mob: M, dt: float) -> None:
        if self.needs_dt:
            self._updater(mob, dt)
        else:
            self._updater(mob)  # type: ignore
