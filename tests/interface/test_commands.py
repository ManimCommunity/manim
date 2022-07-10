from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from manim import __version__, capture
from manim.__main__ import main


def test_manim_version():
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
  --help  Show this message and exit.

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
  --help      Show this message and exit.

Made with <3 by Manim Community developers.
"""
    assert dedent(expected_output) == result.output


def test_manim_init_subcommand():
    command = ["init"]
    runner = CliRunner()
    result = runner.invoke(main, command, prog_name="manim")
    expected_output = """\
Usage: manim init [OPTIONS] COMMAND [ARGS]...

  Create a new project or insert a new scene.

Options:
  --help  Show this message and exit.

Commands:
  project  Creates a new project.
  scene    Inserts a SCENE to an existing FILE or creates a new FILE.

Made with <3 by Manim Community developers.
"""
    assert dedent(expected_output) == result.output


def test_manim_init_project(tmp_path):
    command = ["init", "project", "--default", "testproject"]
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as tmp_dir:
        result = runner.invoke(main, command, prog_name="manim", input="Default\n")
        assert not result.exception
        assert (Path(tmp_dir) / "testproject/main.py").exists()
        assert (Path(tmp_dir) / "testproject/manim.cfg").exists()


def test_manim_new_command():
    command = ["new"]
    runner = CliRunner()
    result = runner.invoke(main, command, prog_name="manim")
    expected_output = """\
Usage: manim new [OPTIONS] COMMAND [ARGS]...

  (DEPRECATED) Create a new project or insert a new scene.

Options:
  --help  Show this message and exit.

Commands:
  project  Creates a new project.
  scene    Inserts a SCENE to an existing FILE or creates a new FILE.

Made with <3 by Manim Community developers.
"""
    assert dedent(expected_output) == result.output
