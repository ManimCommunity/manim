import code
import functools
import os
import readline
import rlcompleter
import subprocess

from colorama import Fore, Style

from . import config, logger
from .scene.streaming_scene import get_streamer, play_scene


__all__ = ["livestream", "stream", "open_client"]


globals().update(config.streaming_config)


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

Want to render the animation of an entire pre-baked scene? Here's an example:

>>> from example_scenes import basic
>>> play_scene(basic.WarpSquare)
>>> play_scene(basic.OpeningManimExample, start=0, end=5)

To view an image of the current state of the scene or mobject, use: 

>>> manim.show_frame()        #For Scene
>>> c = Circle()
>>> c.show()                  #For mobject
"""


def open_client(client=None):
    command = [
        client or streaming_client,
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


def _disable_logging(func):
    """Decorator for running trigger Wait() animations without showing the
    usual output expected from this action
    """
    functools.wraps(func)

    def action(*args, **kwargs):
        logger.disabled = True
        func(*args, **kwargs)
        logger.disabled = False

    return action


@_disable_logging
def _guarantee_sdp_file(*args):
    """Ensures, if required, that the sdp file exists,
    while supressing the loud info message given out by this process
    """
    if not os.path.exists(sdp_path):
        kicker = get_streamer()
        kicker.wait()
        del kicker


@_disable_logging
def _popup_window(shell):
    """Triggers the opening of the window. May lack utility for a streaming
    client like vlc
    """
    shell.push("get_streamer().wait(0.5)")


def livestream():
    """Main function, intended for use from module execution
    Also has its application in a REPL, though the less activated version of this
    might be more suitable for quick sanity and testing checks."""
    variables = {
        "manim": get_streamer(),
        "get_streamer": get_streamer,
        "play_scene": play_scene,
        "open_client": open_client,
    }
    readline.set_completer(rlcompleter.Completer(variables).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(variables)
    shell.push("from manim import *")

    logger.debug("Ensuring sdp file exists: Running Wait() animation")
    _guarantee_sdp_file()

    open_client()

    logger.debug("Triggering streaming client window: Running Wait() animation")
    _popup_window(shell)

    shell.interact(banner=f"{Fore.GREEN}{info}{Style.RESET_ALL}")


def stream():
    """For a quick import and livestream eg:

    >>> from manim import stream, open_client, Circle, ShowCreation
    >>> manim = stream()
    >>> open_client()
    >>> circ = Circle()
    >>> manim.play(ShowCreation(circ))
    """
    _guarantee_sdp_file()
    streamer = get_streamer()
    open_client()
    return streamer
