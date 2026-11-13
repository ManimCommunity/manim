"""What a run produces: a render writes media, an evaluation only runs the scene.

Each run's loop has a mode, chosen with the loop (see :meth:`.Manager._run`). The
Manager asks the mode of the current run where the two kinds of run differ,
instead of checking a flag in each place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..manager import Manager
    from ..typing import RGBAPixelArray

__all__: list[str] = []


class _RunMode:
    """A render: the run writes media, a log file and the scene's sections."""

    #: The name of the mode in messages, such as "evaluation".
    name = "render"
    #: The run's ``mode`` in an exported timeline (see :doc:`/guides/timeline`).
    timeline_mode: Literal["render", "evaluate"] = "render"
    #: Whether the run writes the scene's log file.
    writes_log = True
    #: Whether frames added outside of a play must be drawn when they are added.
    draws_added_frames = True

    def check(self, what: str) -> None:
        """Raise if ``what`` (for example "Media output") is unavailable in this
        run.
        """

    def rollback(self, manager: Manager) -> Callable[[], None]:
        """Return a callback that undoes the output bookkeeping of a play that
        fails; see :meth:`.Manager._play_rollback`.
        """
        return manager._output_rollback()

    def add_frames(
        self, manager: Manager, pixels: RGBAPixelArray | None, repeat: int
    ) -> None:
        """Show ``repeat`` frames outside of a play, as the renderers'
        ``add_frame()`` and ``freeze_current_frame()`` do.

        A render writes them (``pixels`` is the drawn image) and advances the
        clock whether or not they are part of the output, so that scene time
        does not depend on the cache. Only plays mark scene time as shown in the
        movie: frames added during a play lie in its span, and frames added
        between plays are not written to a movie, which has no open segment
        then.
        """
        assert pixels is not None
        manager._execution.frame += repeat
        if not manager.skip_animations:
            manager.file_writer.write_frame(pixels, repeat=repeat)


class _EvaluationRun(_RunMode):
    """An evaluation: the run only runs the scene. No media, log file or
    interactive preview; a failed play leaves the scene where it stopped.

    The mode stays after the evaluation: an evaluated scene does not produce
    media later either.
    """

    name = "evaluation"
    timeline_mode: Literal["render", "evaluate"] = "evaluate"
    writes_log = False
    draws_added_frames = False

    def check(self, what: str) -> None:
        raise RuntimeError(f"{what} is unavailable in an evaluated scene.")

    def rollback(self, manager: Manager) -> Callable[[], None]:
        # Evaluation produces no output to keep consistent, and the clock stays
        # where the mobjects were left, for inspection after a failure.
        return lambda: None

    def add_frames(
        self, manager: Manager, pixels: RGBAPixelArray | None, repeat: int
    ) -> None:
        # The frames go to the evaluation's listeners, drawn only if requested.
        manager._loop.add_frames(pixels, repeat)
