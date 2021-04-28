import sys
from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from manim import __version__
from manim.__main__ import main

from .utils.video_tester import *


def test_manim_version():
    command = ["--version"]

    command = [
        sys.executable,
        "-m",
        "manim",
        "--version",
    ]
    out, err, exit_code = capture(command)
    assert exit_code == 0, err
    assert __version__ in out


def test_manim_cfg_subcommand():
    command = ["cfg"]
    runner = CliRunner()
    result = runner.invoke(main, command, prog_name="manim")
    expected_output = """\
Usage: manim cfg [OPTIONS] COMMAND [ARGS]...

  Manages Manim configuration files.

Options:
  -h, --help  Show this message and exit.

Commands:
  export
  show
  write

  Made with <3 by Manim Community developers.
"""
    assert dedent(expected_output) == result.stdout


def test_manim_plugins_subcommand():
    command = ["plugins"]
    runner = CliRunner()
    result = runner.invoke(main, command, prog_name="manim")
    expected_output = """\
Usage: manim plugins [OPTIONS]

  Manages Manim plugins.

Options:
  -l, --list  List available plugins.
  -h, --help  Show this message and exit.

  Made with <3 by Manim Community developers.
"""
    assert dedent(expected_output) == result.output


def test_manim_init_subcommand():
    command = ["init"]
    runner = CliRunner()
    runner.invoke(main, command, prog_name="manim")
    expected_output = """\
[CLI]
frame_rate = 30
pixel_height = 480
pixel_width = 854
background_color = BLACK
background_opacity = 1
scene_names = DefaultScene
from manim import *
class default(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation
  """
    manim_cfg_path = Path("manim.cfg")
    manim_cfg_content = ""
    main_py_path = Path("main.py")
    main_py_content = ""
    with open(manim_cfg_path) as f:
        manim_cfg_content = f.read()

    with open(main_py_path) as f:
        main_py_content = f.read()
        print(manim_cfg_content + main_py_content)

    manim_cfg_path.unlink()
    main_py_path.unlink()

    assert dedent(expected_output) == manim_cfg_content + main_py_content
