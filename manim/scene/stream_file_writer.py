from abc import ABCMeta

import os
import subprocess
import datetime

from .. import config, logger
from ..constants import FFMPEG_BIN
from .scene import Scene
from .scene_file_writer import SceneFileWriter
from ..utils.file_ops import guarantee_existence


class StreamFileWriter(SceneFileWriter):

    FOLDER_PATH = ""

    def __init__(self, renderer, video_quality_config, **kwargs):
        super().__init__(renderer, video_quality_config, "", **kwargs)
        self.refresh_folder_path()
        self.partial_movie_directory = self.FOLDER_PATH

    def init_output_directories(self, scene_name):
        pass

    @classmethod
    def refresh_folder_path(cls):
        path = os.path.join(config.get_dir("streaming_dir"), "clips")
        if path != cls.FOLDER_PATH:
            cls.FOLDER_PATH = os.path.relpath(guarantee_existence(path))

    @property
    def file_path(self):
        return self.partial_movie_files[-1]
        # file_name = self.renderer.num_plays
        # return os.path.join(
        #     self.FOLDER_PATH,
        #     "{:05}{}".format(file_name, config["movie_file_extension"]),
        # )

    # @property
    # def stream_path(self):
    #     return os.path.join(self.FOLDER_PATH, os.listdir(self.FOLDER_PATH)[-1])

    # def begin_animation(self, allow_write=False):
    #     if os.path.exists()

    def end_animation(self, allow_write=False):
        super().end_animation(allow_write=allow_write)
        self.stream()

    def combine_movie_files(self):
        pass

    def stream(self):
        logger.info(
            "Houston, we are ready to launch. Sending over to %(url)s",
            {"url": {self.streaming_url}},
        )
        sdp_path = os.path.join(
            config.get_dir("streaming_dir"),
            "stream_{}.sdp".format(self.streaming_protocol),
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
            command += ["-sdp_file", sdp_path]
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
