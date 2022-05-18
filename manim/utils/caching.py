from __future__ import annotations

from .. import config, logger
from ..utils.hashing import get_hash_from_play_call


def handle_caching_play(func):
    """Decorator that returns a wrapped version of func that will compute
    the hash of the play invocation.

    The returned function will act according to the computed hash: either skip
    the animation because it's already cached, or let the invoked function
    play normally.

    Parameters
    ----------
    func : Callable[[...], None]
        The play like function that has to be written to the video file stream.
        Take the same parameters as `scene.play`.
    """

    # NOTE : This is only kept for OpenGL renderer.
    # The play logic of the cairo renderer as been refactored and does not need this function anymore.
    # When OpenGL renderer will have a proper testing system,
    # the play logic of the latter has to be refactored in the same way the cairo renderer has been, and thus this
    # method has to be deleted.

    def wrapper(self, scene, *args, **kwargs):
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
                self,
                self.camera,
                animations,
                mobjects_on_scene,
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
