from .cairo_renderer import CairoRenderer
from .. import config
from ..scene.stream_file_writer import StreamFileWriter


class StreamCairoRenderer(CairoRenderer):
    def init(self, scene):
        """For compatibility with the __init__ from scene that's not being
        directly overridden
        """
        self.file_writer = StreamFileWriter(self)
