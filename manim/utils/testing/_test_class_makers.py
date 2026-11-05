from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from manim._run.loop import _RunListener
from manim.scene.scene import Scene

if TYPE_CHECKING:
    from manim._run.loop import _Frame
    from manim.manager import Manager

    from ._frames_testers import _FramesTester


class _FrameChecker(_RunListener):
    """Pass every frame of the rendered output to a frames tester, in order."""

    def __init__(self, manager: Manager, tester: _FramesTester) -> None:
        self._manager = manager
        self._tester = tester
        self._count = 0

    def frame(self, frame: _Frame) -> None:
        if frame.play.in_output and not self._manager.skip_animations:
            self.check(frame.pixels())

    def check(self, pixels: np.ndarray) -> None:
        self._tester.check_frame(self._count, pixels)
        self._count += 1


def _make_test_scene_class(
    base_scene: type[Scene],
    construct_test: Callable[[Scene], None],
    frames_tester: _FramesTester | None,
) -> type[Scene]:
    """A scene class running ``construct_test``.

    With a ``frames_tester``, every frame of the output is checked, followed by
    one more frame showing the end of the last animation.
    """

    # TODO: Get the type annotation right for the base_scene argument.
    class _TestedScene(base_scene):  # type: ignore[valid-type, misc]
        def setup(self) -> None:
            super().setup()
            self._frame_checker = None
            if frames_tester is not None:
                manager = self._get_manager()
                self._frame_checker = _FrameChecker(manager, frames_tester)
                manager._add_listener(self._frame_checker)

        def construct(self) -> None:
            construct_test(self)

            # The frames of a play end one frame before its end: go to the end of
            # the last play, and check that state too.
            if self.animations is not None:
                self.update_to_time(self.get_run_time(self.animations))
                if self._frame_checker is not None:
                    self._frame_checker.check(np.asarray(self.get_image()))

    return _TestedScene
