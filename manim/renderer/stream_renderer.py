from manim.renderer.cairo_renderer import CairoRenderer
from manim._config import config, logger
from manim.scene.stream_file_writer import StreamFileWriter

from manim.utils.hashing import get_hash_from_play_call


class StreamCairoRenderer(CairoRenderer):
    def init_scene(self, scene):
        """For compatibility with the __init__ from scene that's not being
        directly overridden
        """
        self.file_writer = StreamFileWriter(self)

    def play(self, scene, *args, **kwargs):
        """Meant to attach some things

        Args:
            scene ([type]): [description]
        """
        scene.compile_animation_data(*args, **kwargs)
        if not config["disable_caching"] and not self.skip_animations:
            hash_current_animation = get_hash_from_play_call(
                scene, self.camera, scene.animations, scene.mobjects
            )
        else:
            hash_current_animation = f"uncached_{self.num_plays:05}"
        self.file_writer.add_partial_movie_file(hash_current_animation)
        super().play(scene, *args, **kwargs)