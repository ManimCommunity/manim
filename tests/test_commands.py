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

    expected_manim_cfg = ""
    expected_main_py = ""

    with open(
        Path.resolve(Path(__file__).parent.parent / "manim/templates/template.cfg"),
    ) as f:
        expected_manim_cfg = f.read()

    with open(
        Path.resolve(Path(__file__).parent.parent / "manim/templates/Default.mtp"),
    ) as f:
        expected_main_py = f.read()

    manim_cfg_path = Path("manim.cfg")
    manim_cfg_content = ""
    main_py_path = Path("main.py")
    main_py_content = ""
    with open(manim_cfg_path) as f:
        manim_cfg_content = f.read()

    with open(main_py_path) as f:
        main_py_content = f.read()

    manim_cfg_path.unlink()
    main_py_path.unlink()

    assert (
        dedent(expected_manim_cfg + "from manim import *\n" + expected_main_py)
        == manim_cfg_content + main_py_content
    )


def test_manim_new_command():
    command = ["new"]
    runner = CliRunner()
    result = runner.invoke(main, command, prog_name="manim")
    expected_output = """\
Usage: manim new [OPTIONS] COMMAND [ARGS]...

  Create a new project or insert a new scene.

Options:
  -h, --help  Show this message and exit.

Commands:
  project  Creates a new project.
  scene    Inserts a SCENE to an existing FILE or creates a new FILE.

  Made with <3 by Manim Community developers.
"""
    assert dedent(expected_output) == result.output
