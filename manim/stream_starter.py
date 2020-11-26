import code

from colorama import Fore
from colorama import Style
from . import config

from manim.scene.streaming_scene import Stream

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
    shell.push("manim.add(FullScreenRectangle())")
    # To trigger the opening of the ffplay window(This one's mine)
    shell.push("manim.wait(0.5)")
    banner = config["streaming_config"]["streaming_console_banner"]
    shell.interact(banner=f"{Fore.BLUE}{banner}{Style.RESET_ALL}")


def stream():
    """For a quick import and livestream eg:

    >>> from manim import `stream, stuff and other stuff`
    >>> manim = stream()
    >>> manim.play(...)
    """
    return BasicStreamer()
