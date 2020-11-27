from .cairo_renderer import CairoRenderer
from .. import config
from ..scene.stream_file_writer import StreamFileWriter


class StreamCairoRenderer(CairoRenderer):
    def init(self, scene):
        streaming_config = config["streaming_config"].copy()
        self.file_writer = StreamFileWriter(
            self, self.video_quality_config, **streaming_config
        )
