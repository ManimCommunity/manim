"""The listeners a render uses: encoder, preview, and progress bar."""

from __future__ import annotations

import datetime
import platform
from bisect import bisect_right
from typing import TYPE_CHECKING

import srt
from tqdm import tqdm

from .. import config, logger
from ..animation.animation import Wait
from .frames import frame_count
from .loop import _Caption, _RunListener, _Section, _Sound

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ..manager import Manager, RenderedFrame
    from .loop import _Frame, _Play

__all__: list[str] = []


class _Encoder(_RunListener):
    """Write the frames of each drawn play to the file writer.

    A play whose partial movie file is already cached (see :class:`._SegmentCache`)
    is not drawn or encoded again; its cached file is used instead.
    """

    def __init__(self, manager: Manager) -> None:
        self._manager = manager
        self._writing = False
        self._start = 0.0
        # The sounds and captions, to warn at the end about those after the scene.
        self._starts: list[tuple[str, str, float]] = []

    def play_started(self, play: _Play) -> None:
        manager = self._manager
        writer = manager.file_writer
        self._start = manager.time
        self._writing = play.drawn
        writer.add_partial_movie_file(play.cache_key)
        manager._execution.animations_hashes.append(play.cache_key)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(manager._execution.animations_hashes[:5])},
        )
        writer.begin_animation(self._writing, animation_index=play.index)

    def frame(self, frame: _Frame) -> None:
        manager = self._manager
        if (
            self._writing
            and not manager.scene.skip_animation_preview
            and manager._wants_frame_pixels
        ):
            manager.file_writer.write_frame(frame.pixels(), repeat=frame.repeat)

    def play_finished(self, play: _Play) -> None:
        writer = self._manager.file_writer
        writer.end_animation(self._writing)
        if play.in_output:
            writer.add_movie_interval(self._start, self._manager.time)
        self._writing = False

    def play_failed(self, play: _Play) -> None:
        # Manager.play discards the play's segment.
        self._writing = False

    def marker(self, marker: _Section | _Caption | _Sound) -> None:
        writer = self._manager.file_writer
        if isinstance(marker, _Section):
            writer.next_section(marker.name, marker.type, marker.skip_animations)
        elif isinstance(marker, _Caption):
            self._starts.append(("Subcaption", repr(marker.content), marker.start))
            writer.subcaptions.append(
                srt.Subtitle(
                    index=len(writer.subcaptions),
                    content=marker.content,
                    start=datetime.timedelta(seconds=marker.start),
                    end=datetime.timedelta(seconds=marker.end),
                )
            )
        else:
            self._starts.append(("Sound", repr(marker.file), marker.start))
            writer.add_sound(
                marker.file,
                marker.start,
                marker.gain,
                gain_to_background=marker.gain_to_background,
            )

    def run_finished(self, ended_early: bool) -> None:
        end = self._manager.time
        for kind, label, start in self._starts:
            if start >= end:
                logger.warning(
                    "%(kind)s %(label)s starts at %(start).2f s, when the scene has "
                    "already ended at %(end).2f s; it is cut.",
                    {"kind": kind, "label": label, "start": start, "end": end},
                )


class _Preview(_RunListener):
    """Draw each frame of the output into the renderer's target and show it.

    Only OpenGL shows frames, in its preview window, paced by wall-clock time.
    """

    def __init__(self, manager: Manager) -> None:
        self._manager = manager

    def frame(self, frame: _Frame) -> None:
        manager = self._manager
        scene = manager.scene
        if not frame.play.drawn or scene.skip_animation_preview:
            return
        frame.draw()
        if frame.play.frozen:
            manager.renderer._present_frozen_frame(scene, frame.play.run_time)
        else:
            manager.renderer._present_frame(scene, frame.offset)


class _MeshGuard(_RunListener):
    """Reject GPU-backed meshes, which evaluation cannot run without a context."""

    def __init__(self, manager: Manager) -> None:
        self._manager = manager

    def check(self) -> None:
        if self._manager.scene.meshes:
            raise RuntimeError(
                "GPU-backed meshes are unsupported in an evaluated scene."
            )

    def play_started(self, play: _Play) -> None:
        self.check()

    def frame(self, frame: _Frame) -> None:
        self.check()


class _FrameRequests(_RunListener):
    """Pass the frames shown at requested times to a function.

    Each frame is drawn when the run reaches it, at most once.
    """

    def __init__(
        self,
        manager: Manager,
        times: Iterable[float],
        on_frame: Callable[[RenderedFrame], object],
    ) -> None:
        self._manager = manager
        self._times = sorted(times)
        self._next = 0
        """The index of the first requested time not yet passed to ``on_frame``."""
        self._on_frame = on_frame

    def frame(self, frame: _Frame) -> None:
        from ..manager import RenderedFrame

        offset = self._manager._execution.time_offset
        rate = self._manager.session_spec.frame_rate

        def start(index: int) -> float:
            return offset + index / rate

        end = frame.index + frame.repeat
        times, limit = self._times, start(end)
        while self._next < len(times) and times[self._next] < limit:
            requested = times[self._next]
            self._next += 1
            # Compare with the start times of the frames: multiplying the requested
            # time by the frame rate can round across a frame boundary.
            starts = range(frame.index, end)
            index = frame.index + bisect_right(starts, requested, key=start) - 1
            self._on_frame(
                RenderedFrame(
                    image=frame.image(),
                    requested_time=requested,
                    time=start(index),
                    frame_index=index,
                )
            )


class _Progress(_RunListener):
    """Show a progress bar for each play, except for frozen waits.

    The bar counts the play's frames, as the run loop steps through them; a play
    whose frames are not observed jumps to its last frame in one step.
    """

    def __init__(self, manager: Manager) -> None:
        self._manager = manager
        self._bar: tqdm[float] | None = None

    def play_started(self, play: _Play) -> None:
        if play.frozen:
            # A frozen wait has a single frame; a bar would stop at 1/N.
            return
        stop_condition = _stop_condition(play)
        if stop_condition is not None:
            # Ends when the condition holds: show the frames without a total.
            description = f"Waiting for {stop_condition.__name__}"
            total = None
        else:
            animations = play.animations
            if len(animations) == 1 and isinstance(animations[0], Wait):
                description = f"Waiting {play.index}"
            else:
                description = f"Animation {play.index}" + "".join(
                    [": ", str(animations[0]), ", etc." if len(animations) > 1 else ""]
                    if animations
                    else []
                )
            total = frame_count(play.run_time, self._manager.session_spec.frame_rate)
        self._bar = tqdm(
            total=total,
            desc=description,
            leave=config["progress_bar"] == "leave",
            ascii=True if platform.system() == "Windows" else None,
            disable=config["progress_bar"] == "none",
        )
        self._manager.scene.time_progression = self._bar

    def frame(self, frame: _Frame) -> None:
        if self._bar is not None:
            # The frames of the play up to and including this one.
            self._bar.update(
                frame.index + frame.repeat - frame.play.first_frame - self._bar.n
            )

    def play_finished(self, play: _Play) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def play_failed(self, play: _Play) -> None:
        self.play_finished(play)


def _stop_condition(play: _Play) -> Callable[[], bool] | None:
    """The stop condition of a play that is a single wait with one, else None."""
    if len(play.animations) == 1 and isinstance(play.animations[0], Wait):
        return play.animations[0].stop_condition
    return None
