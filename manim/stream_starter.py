import code
import os
import readline
import rlcompleter
import subprocess

from colorama import Fore, Style

from . import config, logger
from .scene.scene import Scene
from .scene.streaming_scene import get_streamer, play_scene

__all__ = ["livestream", "stream"]


info = """
Manim is now running in streaming mode. Stream animations by passing
them to manim.play(), e.g.

>>> c = Circle()
>>> manim.play(ShowCreation(c))

The current streaming class under the name `manim` inherits from the
original Scene class. To create a streaming class which inherits from 
another scene class, e.g. MovingCameraScene, create it with the syntax:

>>> manim2 = get_streamer(MovingCameraScene)
>>> 

To view an image of the current state of the scene or mobject, use: 

>>> manim.show_frame()        #For Scene
>>> c = Circle()
>>> c.show()                  #For mobject
"""


def open_ffplay(streaming_client=None, sdp_path=None, **kwargs):
    command = [
        streaming_client,
        "-x",
        "1280",
        "-y",
        "360",  # For a resizeable window
        "-loglevel",
        "quiet",
        "-protocol_whitelist",
        "file,rtp,udp",
        "-i",
        sdp_path,
        "-reorder_queue_size",
        "0",
    ]
    subprocess.Popen(command)


def _hide_wait_output(sdp_path=None, **kwargs):
    """Ensures, if required, that the sdp file exists,
    while supressing the loud info message given out by this process
    """
    # TODO: Bound for improvement later
    # Also, OCD motivated
    logger.debug("Ensuring sdp file exists: Running Wait() animation")
    logger.disabled = True
    if not os.path.exists(sdp_path):
        kicker = get_streamer()
        kicker.wait()
        del kicker
    logger.disabled = False


def livestream():
    """Main function, intended for use from module execution
    Also has its application in a REPL, though the less activated version of this
    might be more suitable for quick sanity and testing checks."""
    variables = {
        "manim": get_streamer(),
        "get_streamer": get_streamer,
        "play_scene": play_scene,
    }
    readline.set_completer(rlcompleter.Completer(variables).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(variables)
    shell.push("from manim import *")
    _hide_wait_output(**config["streaming_config"])
    open_ffplay(**config["streaming_config"])
    shell.interact(banner=f"{Fore.GREEN}{info}{Style.RESET_ALL}")


def stream():
    """For a quick import and livestream eg:

    >>> from manim import stream, Circle, ShowCreation
    >>> manim = stream()
    >>> circ = Circle()
    >>> manim.play(ShowCreation(circ))
    """
    _hide_wait_output(**config["streaming_config"])
    streamer = get_streamer()
    open_ffplay(**config["streaming_config"])
