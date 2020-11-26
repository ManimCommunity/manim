from abc import ABCMeta

import os
import subprocess
import datetime

from .. import config, logger
from ..constants import FFMPEG_BIN
from .scene import Scene
from .scene_file_writer import SceneFileWriter
from ..renderer.cairo_renderer import CairoRenderer
from ..utils.file_ops import guarantee_existence


class StreamFileWriter(SceneFileWriter):

    FOLDER_PATH = ""

    def __init__(self, renderer, video_quality_config, **kwargs):
        super().__init__(renderer, video_quality_config, "", **kwargs)
        self.refresh_folder_path()
        self.partial_movie_directory = self.FOLDER_PATH

    def init_output_directories(self, scene_name):
        pass

    def init_audio(self):
        pass  # Same deal here

    @classmethod
    def refresh_folder_path(cls):
        date = datetime.datetime.now()
        attributes = ["year", "month", "day", "hour"]
        details = [getattr(date, attr) for attr in attributes]
        folder_name = "{}-{}-{}-{:04}h".format(*details)
        path = os.path.join(config.get_dir("streaming_dir"), folder_name)
        if path != cls.FOLDER_PATH:
            cls.FOLDER_PATH = os.path.relpath(guarantee_existence(path))

    @property
    def file_path(self):
        file_name = self.renderer.num_plays
        return os.path.join(
            self.FOLDER_PATH,
            "{:05}{}".format(file_name, config["movie_file_extension"]),
        )

    @property
    def stream_path(self):
        return os.path.join(self.FOLDER_PATH, os.listdir(self.FOLDER_PATH)[-1])
        return None

    def combine_movie_files(self):
        pass

    def stream(self):
        print("Alrighty, now streaming...")
        sdp_path = os.path.join(config.get_dir("streaming_dir"), "streams.sdp")
        command = [
            FFMPEG_BIN,
            "-re",
            "-i",
            self.stream_path,
            "-vcodec",
            "copy",
            "-an",
            "-loglevel",
            "quiet",
            "-sdp_file",
            sdp_path,
            "-f",
            self.streaming_protocol,
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


class StreamCairoRenderer(CairoRenderer):
    def init(self, scene):
        streaming_config = config["streaming_config"].copy()
        self.file_writer = StreamFileWriter(
            self, self.video_quality_config, **streaming_config
        )


class StreamMeta(ABCMeta):
    """The metaclass kind of 'hijacks' the process of scene instantiation, throwing a
    file writer class which is the adequately created Streamer. I didn't want to have
    any streaming classes inherited from anything other than Scene to have to do this.
    But since not everyone has seen a metaclass before, the Stream class shrouds the
    actual purpose of a Scene that can stream; to use this as a metaclass. The updated
    streaming info should reflect this, if I don't forget.
    """

    def __new__(mcs, name, bases, namespace, scene=Scene, **kwargs):
        namespace["renderer_class"] = StreamCairoRenderer
        return super().__new__(mcs, name, tuple([scene]), namespace)


class Stream(metaclass=StreamMeta):
    pass
