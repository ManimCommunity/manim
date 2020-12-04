from .cairo_renderer import CairoRenderer
from .. import config
from ..scene.stream_file_writer import StreamFileWriter


class StreamCairoRenderer(CairoRenderer):
    """I wish there was another way to have a renderer that uses the
    file writer created for the purpose. However, I can't do that without
    the original code being aware of the extra implementation, which is probably
    undesirable style.
    """

    def init(self, scene):
        streaming_config = config["streaming_config"].copy()
        self.file_writer = StreamFileWriter(self, **streaming_config)
