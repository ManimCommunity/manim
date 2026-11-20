"""Find out whether a play's partial movie file was rendered before."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .. import config, logger
from .._config.video_encoder import video_encoder_fingerprint
from ..utils.hashing import get_hash_from_play_call

if TYPE_CHECKING:
    from ..manager import Manager
    from .loop import _Play

__all__: list[str] = []

_CLOCK_VERSION = "sample-v3"
"""Part of every cache key. Change it when a play could give different frames
from the same scene state, so that older cached movies are not reused."""


class _SegmentCache:
    """The partial movie files of a render, keyed by what each play shows.

    The run loop asks it about each play before the listeners hear of the play,
    so whether a play is drawn is decided once (:attr:`._Play.drawn`).
    """

    def __init__(self, manager: Manager) -> None:
        self._manager = manager

    def look_up(self, play: _Play) -> _Play:
        """Return ``play`` with its cache key, and whether its file is cached."""
        manager = self._manager
        key = self._key(play)
        # When a wait with a stop condition stops can depend on state that the
        # cache key does not include, so such waits are always rendered again.
        cached = (
            key is not None
            and not config.disable_caching
            and manager.scene.stop_condition is None
            and manager.file_writer.is_already_cached(key)
        )
        if cached:
            logger.info(
                f"Animation {play.index} : Using cached data (hash : %(key)s)",
                {"key": key},
            )
        return replace(play, cache_key=key, cached=cached)

    def _key(self, play: _Play) -> str | None:
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
