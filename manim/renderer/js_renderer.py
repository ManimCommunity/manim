import copy
import grpc
from ..grpc.gen import renderserver_pb2
from ..grpc.gen import renderserver_pb2_grpc
from .. import logger


class JsRenderer:
    def __init__(self, frame_server):
        self.frame_server = frame_server
        self.camera = JsCamera()
        self.num_plays = 0

    def init_scene(self, scene):
        pass

    def scene_finished(self, scene):
        pass

    def play(self, scene, *args, **kwargs):
        self.num_plays += 1
        self.frame_server.keyframes.append(copy.deepcopy(scene))
        scene.play_internal(*args, **kwargs)

    def update_frame(  # TODO Description in Docstring
        self,
        scene,
        mobjects=None,
        background=None,
        include_submobjects=True,
        ignore_skipping=True,
        **kwargs,
    ):
        pass


class JsCamera:
    def __init__(self, use_z_index=True):
        self.use_z_index = use_z_index
