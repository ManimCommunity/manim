from manim import __version__
from manim.__main__ import main
from click.testing import CliRunner
from textwrap import dedent


def test_manim_version():
    command = ["--version"]

    runner = CliRunner()
    result = runner.invoke(main, command)
    assert result.exit_code == 0
    assert __version__ in result.output


def test_manim_cfg_subcommand():
    command = ["cfg"]
    runner = CliRunner()
    result = runner.invoke(main, command)
    expected_output = """\
Usage: main cfg [OPTIONS] COMMAND [ARGS]...

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
    result = runner.invoke(main, command)
    expected_output = """\
Usage: main plugins [OPTIONS]

  Manages Manim plugins.

Options:
  -l, --list  List available plugins.
  -h, --help  Show this message and exit.

  Made with <3 by Manim Community developers.
"""
    assert dedent(expected_output) == result.output
