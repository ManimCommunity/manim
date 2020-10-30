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

        with grpc.insecure_channel("localhost:50052") as channel:
            stub = renderserver_pb2_grpc.RenderServerStub(channel)
            try:
                stub.AnimationStatus(renderserver_pb2.EmptyRequest())
            except grpc._channel._InactiveRpcError as e:
                logger.error(e)


class JsCamera:
    def __init__(self, use_z_index=True):
        self.use_z_index = use_z_index
