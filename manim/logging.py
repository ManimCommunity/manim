import logging
from rich.logging import RichHandler
from rich.progress import Progress

global_progress = None

class GlobalProgress(Progress):
    """
    A Progress which stores itself to a global variable when its context is
    entered and removes itself when its context is exited.
    """
    def __enter__(self):
        global global_progress
        global_progress = self
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global global_progress
        super().__exit__(exc_type, exc_val, exc_tb)
        global_progress = None

class GlobalProgressRichHandler(RichHandler):
    """
    A RichHandler which checks a global variable for a GlobalProgress and logs
    without breaking its output if one is set.
    """
    def emit(self, record):
        if global_progress is not None:
            console = global_progress.console
            live_render = global_progress._live_render
            if console.is_terminal:
                console.print(live_render.position_cursor())
            super().emit(record)
            if console.is_terminal:
                console.print(live_render)
        else:
            super().emit(record)

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[GlobalProgressRichHandler()]
)

logger = logging.getLogger("rich")
