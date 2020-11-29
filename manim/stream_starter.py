import code

from colorama import Fore
from colorama import Style
from . import logger

from .scene.streaming_scene import Stream


__all__ = ["livestream", "stream"]


class BasicStreamer(Stream):
    pass


info = """Manim is now running in streaming mode. Stream animations by passing
them to manim.play(), e.g.

>>> c = Circle()
>>> manim.play(ShowCreation(c))

The current streaming class under the name `manim` inherits from the
original Scene class. To create a streaming class which inherits from 
another scene class, e.g. MovingCameraScene, create it with the syntax:

>>> class AnotherStreamer(Stream, scene=MovingCameraScene):
...     pass
... 
>>> manim2 = AnotherStreamer()

To view an image of the current state of the scene or mobject, use: 

>>> manim.show_frame()        #For Scene
>>> c = Circle()
>>> c.show()                  #For mobject
"""


def livestream():
    "Main purpose code"
    variables = {"Stream": Stream, "manim": BasicStreamer()}
    shell = code.InteractiveConsole(variables)
    shell.push("from manim import *")
    # To identify the scene area in a black background
    shell.push("_ = manim.add(FullScreenRectangle())")
    logger.info(info)
    shell.interact(banner="")


def stream():
    """For a quick import and livestream eg:

    >>> from manim import stream, Circle, ShowCreation
    >>> manim = stream()
    >>> circ = Circle()
    >>> manim.play(ShowCreation(circ))
    """
    return BasicStreamer()
