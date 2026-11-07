"""The listeners a render uses: encoder, preview, and progress bar."""

from __future__ import annotations

import datetime
import platform
from typing import TYPE_CHECKING

import srt
from tqdm import tqdm

from .. import config, logger
from .._config.video_encoder import video_encoder_fingerprint
from ..animation.animation import Wait
from ..utils.hashing import get_hash_from_play_call
from .frames import frame_count
from .loop import _Caption, _RunListener, _Section, _Sound

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..manager import Manager
    from .loop import _Frame, _Play

__all__: list[str] = []

_CLOCK_VERSION = "sample-v3"
"""Part of every cache key. Change it when a play could give different frames
from the same scene state, so that older cached movies are not reused."""


class _Encoder(_RunListener):
    """Write the frames of each play in the output to the file writer.

    A play whose partial movie file is already cached is not drawn or encoded
    again; its cached file is used instead.
    """

    def __init__(self, manager: Manager) -> None:
        self._manager = manager
        self._writing = False

    def play_started(self, play: _Play) -> None:
        manager = self._manager
        writer = manager.file_writer
        key = self._cache_key(play)
        # When a wait with a stop condition stops can depend on state that the
        # cache key does not include, so such waits are always rendered again.
        if (
            key is not None
            and not config.disable_caching
            and manager.scene.stop_condition is None
            and writer.is_already_cached(key)
        ):
            logger.info(
                f"Animation {play.index} : Using cached data (hash : %(key)s)",
                {"key": key},
            )
            manager.skip_animations = True
        self._writing = play.in_output and not manager.skip_animations
        writer.add_partial_movie_file(key)
        manager._execution.animations_hashes.append(key)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(manager._execution.animations_hashes[:5])},
        )
        writer.begin_animation(self._writing, animation_index=play.index)

    def _cache_key(self, play: _Play) -> str | None:
        manager = self._manager
        if not play.in_output:
            logger.debug(f"Skipping animation {play.index}")
            return None
        if config.disable_caching:
            logger.info("Caching disabled.")
            return f"uncached_{play.index:05}"
        scene = manager.scene
        backend, drawing_state = manager.renderer._animation_cache_identity(scene)
        return get_hash_from_play_call(
            scene,
            manager.camera,
            list(play.animations),
            scene.mobjects,
            backend=backend,
            encoder_fingerprint=video_encoder_fingerprint(
                manager.session_spec.video_encoder
            ),
            renderer_state={
                "drawing": drawing_state,
                "execution": {
                    "clock": _CLOCK_VERSION,
                    # Plays can read the scene time, so the same inputs at another
                    # time can give other frames.
                    "time": manager.time,
                    "frame_rate": manager.session_spec.frame_rate,
                },
            },
        )

    def frame(self, frame: _Frame) -> None:
        manager = self._manager
        if (
            self._writing
            and not manager.scene.skip_animation_preview
            and manager._wants_frame_pixels
        ):
            manager.file_writer.write_frame(frame.pixels(), repeat=frame.repeat)

    def play_finished(self, play: _Play, end_frame: int) -> None:
        self._manager.file_writer.end_animation(self._writing)
        self._writing = False

    def play_failed(self, play: _Play) -> None:
        # Manager.play discards the play's segment.
        self._writing = False

    def marker(self, marker: _Section | _Caption | _Sound) -> None:
        writer = self._manager.file_writer
        if isinstance(marker, _Section):
            writer.next_section(marker.name, marker.type, marker.skip_animations)
        elif isinstance(marker, _Caption):
            writer.subcaptions.append(
                srt.Subtitle(
                    index=len(writer.subcaptions),
                    content=marker.content,
                    start=datetime.timedelta(seconds=marker.start),
                    end=datetime.timedelta(seconds=marker.end),
                )
            )
        elif marker.in_output:
            writer.add_sound(marker.file, marker.start, marker.gain, **marker.options)


class _Preview(_RunListener):
    """Draw each frame of the output into the renderer's target and show it.

    Only OpenGL shows frames, in its preview window, paced by wall-clock time.
    """

    def __init__(self, manager: Manager) -> None:
        self._manager = manager

    def frame(self, frame: _Frame) -> None:
        manager = self._manager
        scene = manager.scene
        if manager.skip_animations or scene.skip_animation_preview:
            return
        frame.draw()
        if frame.play.frozen:
            manager.renderer._present_frozen_frame(scene, frame.play.run_time)
        else:
            manager.renderer._present_frame(scene, frame.offset)


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

    def play_finished(self, play: _Play, end_frame: int) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def play_failed(self, play: _Play) -> None:
        self.play_finished(play, 0)


def _stop_condition(play: _Play) -> Callable[[], bool] | None:
    """The stop condition of a play that is a single wait with one, else None."""
    if len(play.animations) == 1 and isinstance(play.animations[0], Wait):
        return play.animations[0].stop_condition
    return None
