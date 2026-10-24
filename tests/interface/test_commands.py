from __future__ import annotations

import shutil
import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from click.testing import CliRunner

from manim import __version__, capture
from manim.__main__ import main
from manim.cli.checkhealth.checks import HEALTH_CHECKS


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
    expected_output = f"""\
Manim Community v{__version__}

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
    assert dedent(expected_output) == result.output


def test_manim_plugins_subcommand():
    command = ["plugins"]
    runner = CliRunner()
    result = runner.invoke(main, command, prog_name="manim")
    expected_output = f"""\
Manim Community v{__version__}

Usage: manim plugins [OPTIONS]

  Manages Manim plugins.

Options:
  -l, --list  List available plugins.
  --help      Show this message and exit.

Made with <3 by Manim Community developers.
"""
    assert dedent(expected_output) == result.output


def test_manim_checkhealth_subcommand():
    command = ["checkhealth"]
    runner = CliRunner()
    result = runner.invoke(main, command)
    output_lines = result.output.split("\n")
    num_passed = len([line for line in output_lines if "PASSED" in line])
    assert num_passed == len(HEALTH_CHECKS), (
        f"Some checks failed! Full output:\n{result.output}"
    )
    assert "No problems detected, your installation seems healthy!" in output_lines


def test_manim_checkhealth_failing_subcommand():
    command = ["checkhealth"]
    runner = CliRunner()
    true_f = shutil.which

    def mock_f(s):
        if s == "latex":
            return None

        return true_f(s)

    with patch.object(shutil, "which", new=mock_f):
        result = runner.invoke(main, command)

    output_lines = result.output.split("\n")
    assert "- Checking whether latex is available ... FAILED" in output_lines
    assert "- Checking whether dvisvgm is available ... SKIPPED" in output_lines


def test_manim_init_subcommand():
    command = ["init"]
    runner = CliRunner()
    result = runner.invoke(main, command, prog_name="manim")
    expected_output = f"""\
Manim Community v{__version__}

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


def test_manim_init_scene(tmp_path):
    command_named = ["init", "scene", "NamedFileTestScene", "my_awesome_file.py"]
    command_unnamed = ["init", "scene", "DefaultFileTestScene"]
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as tmp_dir:
        result = runner.invoke(
            main, command_named, prog_name="manim", input="Default\n"
        )
        assert not result.exception
        assert (Path(tmp_dir) / "my_awesome_file.py").exists()
        file_content = (Path(tmp_dir) / "my_awesome_file.py").read_text()
        assert "NamedFileTestScene(Scene):" in file_content
        result = runner.invoke(
            main, command_unnamed, prog_name="manim", input="Default\n"
        )
        assert (Path(tmp_dir) / "main.py").exists()
        file_content = (Path(tmp_dir) / "main.py").read_text()
        assert "DefaultFileTestScene(Scene):" in file_content
