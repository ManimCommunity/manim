from abc import ABCMeta

from ..renderer.stream_renderer import StreamCairoRenderer
from .scene import Scene


class StreamMeta(ABCMeta):
    """The metaclass kind of 'hijacks' the process of scene instantiation, throwing a
    renderer which is the adequately created StreamCairoRenderer. I didn't want to have
    any streaming classes inherited from anything other than Scene to have to do this.
    But since not everyone has seen a metaclass before, the Stream class shrouds the
    actual purpose of a Scene that can stream; to use this as a metaclass. The updated
    streaming info should reflect this, if I don't forget.
    """

    def __new__(mcs, name, bases, namespace, scene=Scene, **kwargs):
        namespace["renderer"] = StreamCairoRenderer(
            camera_class=scene.CONFIG["camera_class"]
        )
        return super().__new__(mcs, name, tuple([scene]), namespace)


class Stream(metaclass=StreamMeta):
    pass
