import code

from colorama import Fore
from colorama import Style
from . import config

from .scene.streaming_scene import Stream

# from .config import file_writer_config


__all__ = ["livestream", "stream"]


class BasicStreamer(Stream):
    pass


def livestream():
    "Main purpose code"
    variables = {"Stream": Stream, "manim": BasicStreamer()}
    shell = code.InteractiveConsole(variables)
    shell.push("from manim import *")
    # To identify the scene area in a black background
    shell.push("_ = manim.add(FullScreenRectangle())")
    banner = config["streaming_config"]["streaming_console_banner"]
    shell.interact(banner=f"{Fore.GREEN}{banner}{Style.RESET_ALL}")


def stream():
    """For a quick import and livestream eg:

    >>> from manim import stream, Circle, ShowCreation
    >>> manim = stream()
    >>> circ = Circle()
    >>> manim.play(ShowCreation(circ))
    """
    return BasicStreamer()
