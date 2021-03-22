import subprocess
import sys

from manim.__main__ import manim, __version__
from click.testing import CliRunner
from .test_plugins.test_plugins import function_like_plugin


def test_manim_version():
    command = ["--version"]

    runner = CliRunner()
    result = runner.invoke(manim, command)
    assert result.exit_code == 0
    assert __version__ in result.output


def test_manim_cfg_subcommand():
    command = ["cfg"]
    runner = CliRunner()
    result = runner.invoke(manim, command)
    expected_output = """Usage: manim cfg [OPTIONS] COMMAND [ARGS]...

  Manages Manim configuration files.

Options:
  -h, --help  Show this message and exit.

Commands:
  export
  show
  write

  Made with <3 by Manim Community developers.
"""
    assert expected_output == result.stdout


def test_manim_plugins_subcommand():
    command = ["plugins"]
    runner = CliRunner()
    result = runner.invoke(manim, command)
    expected_output = """Usage: manim plugins [OPTIONS]

  Manages Manim plugins.

Options:
  -l, --list  List available plugins
  -h, --help  Show this message and exit.

  Made with <3 by Manim Community developers.
"""
    assert expected_output == result.output
