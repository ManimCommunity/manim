from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import config, logger
from .._config.video_encoder import video_encoder_fingerprint
from ..utils.hashing import get_hash_from_play_call

__all__ = [
    "clear_segment_cache",
    "handle_caching_play",
    "prune_segment_cache",
]

_SEGMENT_EXTENSIONS = frozenset({".mov", ".mp4", ".webm"})
_SEGMENT_MANIFEST = "partial_movie_file_list.txt"


def _segment_cache_files(directory: Path) -> list[Path]:
    """Return recognized segment files currently present in ``directory``."""
    try:
        entries = directory.iterdir()
        return [
            entry
            for entry in entries
            if not entry.name.startswith(".")
            and entry.name != _SEGMENT_MANIFEST
            and entry.suffix.lower() in _SEGMENT_EXTENSIONS
            and entry.is_file()
        ]
    except FileNotFoundError:
        return []


def prune_segment_cache(directory: Path, max_files: int) -> None:
    """Remove the least recently accessed segments beyond ``max_files``.

    ``max_files=-1`` leaves the cache unlimited. Files that disappear during
    pruning count toward the number selected for removal, preventing a race
    from evicting an additional live segment.
    """
    if max_files == -1:
        return
    if max_files < 0:
        raise ValueError("max_files must be non-negative or -1 for unlimited")

    cached_segments = _segment_cache_files(directory)
    excess = len(cached_segments) - max_files
    if excess <= 0:
        return

    def access_time(path: Path) -> float:
        try:
            return path.stat().st_atime
        except FileNotFoundError:
            return float("-inf")

    for segment in sorted(cached_segments, key=access_time)[:excess]:
        segment.unlink(missing_ok=True)

    logger.info(
        "The segment cache exceeded %d files; removed %d least recently used "
        "segment(s). Change max_files_cached to adjust this limit.",
        max_files,
        excess,
    )


def clear_segment_cache(directory: Path) -> int:
    """Delete recognized segment files from ``directory`` and return the count."""
    removed = 0
    for segment in _segment_cache_files(directory):
        try:
            segment.unlink()
        except FileNotFoundError:
            continue
        removed += 1
    return removed


if TYPE_CHECKING:
    from manim.renderer.opengl_renderer import OpenGLRenderer
    from manim.scene.scene import Scene


def handle_caching_play(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator that returns a wrapped version of func that will compute
    the hash of the play invocation.

    The returned function will act according to the computed hash: either skip
    the animation because it's already cached, or let the invoked function
    play normally.

    Parameters
    ----------
    func
        The play like function that has to be written to the video file stream.
        Take the same parameters as `scene.play`.
    """
    # NOTE : This is only kept for OpenGL renderer.
    # The play logic of the cairo renderer as been refactored and does not need this function anymore.
    # When OpenGL renderer will have a proper testing system,
    # the play logic of the latter has to be refactored in the same way the cairo renderer has been, and thus this
    # method has to be deleted.

    def wrapper(self: OpenGLRenderer, scene: Scene, *args: Any, **kwargs: Any) -> None:
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()
        animations = scene.compile_animations(*args, **kwargs)
        scene.add_mobjects_from_animations(animations)
        if self.skip_animations:
            logger.debug(f"Skipping animation {self.num_plays}")
            func(self, scene, *args, **kwargs)
            # If the animation is skipped, we mark its hash as None.
            # When sceneFileWriter will start combining partial movie files, it won't take into account None hashes.
            self.animations_hashes.append(None)
            self.file_writer.add_partial_movie_file(None)
            return
        if not config["disable_caching"]:
            mobjects_on_scene = scene.mobjects
            hash_play = get_hash_from_play_call(
                scene,
                self.camera,
                animations,
                mobjects_on_scene,
                backend="opengl",
                encoder_fingerprint=video_encoder_fingerprint(
                    self.file_writer.video_encoder,
                ),
                renderer_state={
                    "meshes": scene.meshes,
                    "background_color": self.background_color,
                    "anti_alias_width": self.anti_alias_width,
                },
            )
            if self.file_writer.is_already_cached(hash_play):
                logger.info(
                    f"Animation {self.num_plays} : Using cached data (hash : %(hash_play)s)",
                    {"hash_play": hash_play},
                )
                self.skip_animations = True
        else:
            hash_play = f"uncached_{self.num_plays:05}"
        self.animations_hashes.append(hash_play)
        self.file_writer.add_partial_movie_file(hash_play)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self.animations_hashes[:5])},
        )
        func(self, scene, *args, **kwargs)

    return wrapper
