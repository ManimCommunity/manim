from ... import config
from ...scene import scene
from ..gen import frameserver_pb2
from ..gen import frameserver_pb2_grpc
from ..gen import renderserver_pb2
from ..gen import renderserver_pb2_grpc
from concurrent import futures
from google.protobuf import json_format
from watchdog.events import LoggingEventHandler, FileSystemEventHandler
from watchdog.observers import Observer
import grpc
import subprocess as sp
import threading
import time
import traceback
import ctypes
from ...utils.module_ops import (
    get_module,
    get_scene_classes_from_module,
    get_scenes_to_render,
)
from ... import logger
from ...constants import JS_RENDERER_INFO
from ...renderer.js_renderer import JsRenderer
from ...utils.family import extract_mobject_family_members


class FrameServer(frameserver_pb2_grpc.FrameServerServicer):
    def animation_index_is_cached(self, animation_index):
        return animation_index < len(self.keyframes)

    def __init__(self, server, scene_class):
        self.server = server
        self.keyframes = []
        self.renderer = JsRenderer(self)
        self.scene = scene_class(self.renderer)
        self.scene_thread = threading.Thread(
            target=lambda s: s.render(), args=(self.scene,)
        )

        # If a javascript renderer is running, notify it of the scene being served. If
        # not, spawn one and it will request the scene when it starts.
        with grpc.insecure_channel("localhost:50052") as channel:
            stub = renderserver_pb2_grpc.RenderServerStub(channel)
            request = renderserver_pb2.NewSceneRequest(name=str(self.scene))
            try:
                stub.NewScene(request)
            except grpc._channel._InactiveRpcError:
                logger.warning("No frontend was detected at localhost:50052.")
                try:
                    sp.Popen(config["js_renderer_path"])
                except PermissionError:
                    logger.info(JS_RENDERER_INFO)
                    self.server.stop(None)
                    return

        self.scene_thread.start()

    def GetFrameAtTime(self, request, context):
        try:
            # Find the requested scene.
            requested_scene_index = 0
            requested_scene = self.keyframes[requested_scene_index]
            requested_scene_end_time = requested_scene.duration
            scene_finished = False
            while requested_scene_end_time < request.scene_offset:
                if requested_scene_index + 1 < len(self.keyframes):
                    requested_scene_index += 1
                    requested_scene = self.keyframes[requested_scene_index]
                    requested_scene_end_time += requested_scene.duration
                else:
                    scene_finished = True
                    break

            # Update to the requested time.
            if not scene_finished:
                requested_scene_start_time = (
                    requested_scene_end_time - requested_scene.duration
                )
                requested_scene_time_offset = (
                    request.scene_offset - requested_scene_start_time
                )
                requested_scene.update_to_time(requested_scene_time_offset)
            else:
                requested_scene.update_to_time(requested_scene.duration)

            # Serialize the scene's mobjects.
            mobjects = extract_mobject_family_members(
                requested_scene.mobjects, only_those_with_points=True
            )
            serialized_mobjects = [serialize_mobject(mobject) for mobject in mobjects]

            resp = frameserver_pb2.FrameResponse(
                mobjects=serialized_mobjects,
                frame_pending=False,
                animation_finished=False,
                scene_finished=scene_finished,
                duration=requested_scene.duration,
                animations=map(
                    lambda anim: anim.__class__.__name__, requested_scene.animations
                ),
                animation_index=requested_scene_index,
            )
            return resp
        except Exception as e:
            traceback.print_exc()

    def RendererStatus(self, request, context):
        response = frameserver_pb2.RendererStatusResponse()
        response.scene_name = str(self.scene)
        return response

    # def UpdateSceneLocation(self, request, context):
    #     # Reload self.scene.
    #     print(scene_classes_to_render)

    #     response = frameserver_pb2.SceneLocationResponse()
    #     return response


def serialize_mobject(mobject):
    mob_proto = frameserver_pb2.MobjectData()

    needs_redraw = False
    point_hash = hash(tuple(mobject.points.flatten()))
    if mobject.point_hash != point_hash:
        mobject.point_hash = point_hash
        needs_redraw = True
    mob_proto.needs_redraw = needs_redraw

    for point in mobject.points:
        point_proto = mob_proto.points.add()
        point_proto.x = point[0]
        point_proto.y = point[1]
        point_proto.z = point[2]

    mob_style = mobject.get_style(simple=True)
    mob_proto.style.fill_color = mob_style["fill_color"]
    mob_proto.style.fill_opacity = float(mob_style["fill_opacity"])
    mob_proto.style.stroke_color = mob_style["stroke_color"]
    mob_proto.style.stroke_opacity = float(mob_style["stroke_opacity"])
    mob_proto.style.stroke_width = float(mob_style["stroke_width"])

    mob_proto.id = id(mobject)
    return mob_proto


class UpdateFrontendHandler(FileSystemEventHandler):
    """Logs all the events captured."""

    def __init__(self, frame_server):
        super().__init__()
        self.frame_server = frame_server

    def on_moved(self, event):
        super().on_moved(event)
        raise NotImplementedError("Update not implemented for moved files.")

    def on_deleted(self, event):
        super().on_deleted(event)
        raise NotImplementedError("Update not implemented for deleted files.")

    def on_modified(self, event):
        super().on_modified(event)
        module = get_module(config["input_file"])
        all_scene_classes = get_scene_classes_from_module(module)
        scene_classes_to_render = get_scenes_to_render(all_scene_classes)
        scene_class = scene_classes_to_render[0]

        # Get the old thread's ID.
        old_thread_id = None
        old_thread = self.frame_server.scene_thread
        if hasattr(old_thread, "_thread_id"):
            old_thread_id = old_thread._thread_id
        if old_thread_id is None:
            for thread_id, thread in threading._active.items():
                if thread is old_thread:
                    old_thread_id = thread_id

        # Stop the old thread.
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            old_thread_id, ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(old_thread_id, 0)
            print("Exception raise failure")
        old_thread.join()

        # Start a new thread.
        self.frame_server.initialize_scene(scene_class, start_animation=1)
        self.frame_server.scene.reached_start_animation.wait()

        # Serialize data on Animations up to the target one.
        animations = []
        for scene in self.frame_server.keyframes:
            if scene.animations:
                animation_duration = scene.run_time
                if len(scene.animations) == 1:
                    animation_name = str(scene.animations[0])
                else:
                    animation_name = f"{str(scene.animations[0])}..."
            else:
                animation_duration = scene.duration
                animation_name = "Wait"
            animations.append(
                renderserver_pb2.Animation(
                    name=animation_name,
                    duration=animation_duration,
                )
            )

        # Reset the renderer.
        with grpc.insecure_channel("localhost:50052") as channel:
            stub = renderserver_pb2_grpc.RenderServerStub(channel)
            request = renderserver_pb2.ManimStatusRequest(
                scene_name=str(self.frame_server.scene), animations=animations
            )
            try:
                stub.ManimStatus(request)
            except grpc._channel._InactiveRpcError:
                sp.Popen(config["js_renderer_path"])


def get(scene_class):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    frameserver_pb2_grpc.add_FrameServerServicer_to_server(
        FrameServer(server, scene_class), server
    )
    server.add_insecure_port("localhost:50051")
    return server
