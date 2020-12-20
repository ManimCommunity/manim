"""The most correct namespace to use in IPython would be
that in a file.
"""

from manim import *
from manim.scene.streaming_scene import get_streamer, play_scene
from manim.stream_starter import open_client


def main():
    from colorama import Fore, Style
    from manim.stream_starter import info, _guarantee_sdp_file, _popup_window

    open_client()
    _guarantee_sdp_file()
    _popup_window()
    print(f"{Fore.GREEN}{info}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
    manim = get_streamer()
    del main
