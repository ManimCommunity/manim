import datetime
import os
import subprocess

from ..constants import FFMPEG_BIN
from .scene import Scene
from .scene_file_writer import SceneFileWriter
from ..utils.file_ops import guarantee_existence
from ..config import file_writer_config



# Again, only because of my affiliation with the original
# When this works I'll make it cleaner


class Streamer(SceneFileWriter):
    CONFIG = {"movie_file_extension": ".mp4",
              "write_to_movie": True,
              "file_name": None,
              "output_directory": None}

    FOLDER_PATH = ""
    STREAM_FOLDER = os.path.join(file_writer_config["media_dir"], "streams")

    def __init__(self, scene, **kwargs):
        super().__init__(scene, **kwargs)
        self.refresh_folder_path()

    def init_output_directories(self):
        pass   # Instead of fully overriding __init__

    def init_audio(self):
        pass   # Same deal here

    @classmethod
    def refresh_folder_path(cls):
        date = datetime.datetime.now()
        attributes = ["year", "month", "day", "hour"]
        details = [getattr(date, attr) for attr in attributes]
        folder_name = "{}-{}-{}-{:04}h".format(*details)
        path = os.path.join(cls.STREAM_FOLDER, folder_name)
        if path != cls.FOLDER_PATH:
            cls.FOLDER_PATH = os.path.relpath(guarantee_existence(path))

    @property
    def file_path(self):
        file_name = len(os.listdir(self.FOLDER_PATH))
        return os.path.join(self.FOLDER_PATH,
                            "{:05}{}".format(file_name,
                                             self.movie_file_extension))

    @property
    def stream_path(self):
        if os.listdir(self.FOLDER_PATH):
            return os.path.join(self.FOLDER_PATH,
                                os.listdir(self.FOLDER_PATH)[-1])
        return None

    def finish(self):
        if hasattr(self, "writing_process"):
            self.writing_process.terminate()

    def stream(self):
        print("Alrighty, now streaming...")        
        sdp_path = os.path.join(self.STREAM_FOLDER, "streams.sdp")
        command = [FFMPEG_BIN,
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
                    self.streaming["streaming_protocol"],
                    self.streaming["streaming_url"]]
        if self.stream_path:
            os.system(" ".join(command))        

    def open_movie_pipe(self):
        fps = self.scene.camera.frame_rate
        height = self.scene.camera.get_pixel_height()
        width = self.scene.camera.get_pixel_width()

        command = [FFMPEG_BIN,
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
                   "-",       # The imput comes from a pipe
                   "-an",     # Tells FFMPEG not to expect any audio
                   "-loglevel",
                   "error"]
        if file_writer_config["movie_file_extension"] == ".mov":
            command += ["-vcodec", "qtrle"]
        else:
            command += ["-vcodec",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p"]
        command += [self.file_path]
        self.writing_process = subprocess.Popen(command, stdin=subprocess.PIPE)        

    def close_movie_pipe(self):
        self.writing_process.stdin.close() 
        self.writing_process.wait()        


class StreamMeta(type):
    """The metaclass kind of 'hijacks' the process of scene instantiation, throwing a
    file writer class which is the adequately created Streamer. I didn't want to have
    any streaming classes inherited from anything other than Scene to have to do this.
    But since not everyone has seen a metaclass before, the Stream class shrouds the 
    actual purpose of a Scene that can stream; to use this as a metaclass. The updated
    streaming info should reflect this, if I don't forget.
    """
    def __new__(mcs, name, bases, namespace, scene=Scene, **kwargs):
        namespace["file_writer_class"] = Streamer
        return super().__new__(mcs, name, tuple([scene]), namespace)


class Stream(metaclass=StreamMeta):
    pass
