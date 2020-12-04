import code

from colorama import Fore
from colorama import Style

from .scene.streaming_scene import get_streamer
from .scene.scene import Scene


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

Did you say multiple inheritance?

>>> manim3 = get_streamer(MovingCameraScene, LinearTransformationScene)
>>> 

To view an image of the current state of the scene or mobject, use: 

>>> manim.show_frame()        #For Scene
>>> c = Circle()
>>> c.show()                  #For mobject
"""


def livestream():
    "Main purpose code"
    variables = {"manim": get_streamer(), "get_streamer": get_streamer}
    shell = code.InteractiveConsole(variables)
    shell.push("from manim import *")
    shell.interact(banner=f"{Fore.GREEN}{info}{Style.RESET_ALL}")


def stream():
    """For a quick import and livestream eg:

    >>> from manim import stream, Circle, ShowCreation
    >>> manim = stream()
    >>> circ = Circle()
    >>> manim.play(ShowCreation(circ))
    """
    return get_streamer()
