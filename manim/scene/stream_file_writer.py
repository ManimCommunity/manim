from abc import ABCMeta

import os
import subprocess

from .. import config, logger
from ..constants import FFMPEG_BIN
from .scene import Scene
from .scene_file_writer import SceneFileWriter
from ..utils.file_ops import guarantee_existence


class StreamFileWriter(SceneFileWriter):
    def __init__(self, renderer):
        super().__init__(renderer, "")
        vars(self).update(config["streaming_config"])
        path = os.path.join(config.get_dir("streaming_dir"), "clips")
        self.FOLDER_PATH = os.path.relpath(guarantee_existence(path))
        # To prevent extensive overwriting
        self.partial_movie_directory = self.FOLDER_PATH

    def init_output_directories(self, scene_name):
        """The original :class:`SceneFileWriter` uses this method while initializing.
        I need most of that initialization, minus this. Hence kicked to the curb.
        """
        pass

    @property
    def file_path(self):
        return self.partial_movie_files[-1]

    def end_animation(self, allow_write=False):
        """The point in the animation where the file exists."""
        super().end_animation(allow_write=allow_write)
        self.stream()

    def combine_movie_files(self):
        """Also to reduce overriding code."""
        pass

    def stream(self):
        logger.info(
            "Houston, we are ready to launch. Sending over to %(url)s",
            {"url": {self.streaming_url}},
        )
        command = [
            FFMPEG_BIN,
            "-re",
            "-i",
            self.file_path,
            "-vcodec",
            "copy",
            "-an",
            "-loglevel",
            "quiet",
        ]

        if self.streaming_protocol == "rtp":
            command += ["-sdp_file", self.sdp_path]
        command += [
            "-f",
            (
                self.streaming_protocol
                if self.streaming_protocol == "rtp"
                else "mpegts"
            ),  # udp protocol didn't work for me but if it does for you congrats
            self.streaming_url,
        ]
        os.system(" ".join(command))

    def open_movie_pipe(self):
        fps = config["frame_rate"]
        height = config["pixel_height"]
        width = config["pixel_width"]

        command = [
            FFMPEG_BIN,
            "-y",  # overwrite output file if it exists
            "-f",
            "rawvideo",
            "-s",
            "%dx%d" % (width, height),  # size of one frame
            "-pix_fmt",
            "rgba",
            "-r",
            str(fps),  # frames per second
            "-i",
            "-",  # The imput comes from a pipe
            "-an",  # Tells FFMPEG not to expect any audio
            "-loglevel",
            "error",
        ]
        if config["transparent"]:
            command += ["-vcodec", "qtrle"]
        else:
            command += ["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
        command += [self.file_path]
        self.writing_process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def close_movie_pipe(self):
        self.writing_process.stdin.close()
        self.writing_process.wait()
        logger.info(
            f"Animation {self.renderer.num_plays}: File at %(path)s",
            {"path": {self.file_path}},
        )
