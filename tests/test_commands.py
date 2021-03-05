import subprocess
import sys

from .test_plugins.test_plugins import function_like_plugin

import manim


def call_command(command, cwd=None, env=None):
    a = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
        cwd=cwd,
        env=env,
    )
    return a


def test_manim_version_from_command_line():
    a = call_command(
        [
            sys.executable,
            "-m",
            "manim",
            "--version",
        ]
    )
    version = manim.__version__
    assert version in a.stdout
    assert a.stdout.strip() == f"Manim Community v{version}"


def test_manim_cfg_subcommand_no_subcommand():
    a = call_command(
        [
            sys.executable,
            "-m",
            "manim",
            "cfg",
        ]
    )
    assert "No subcommand provided; Exiting..." in a.stdout


def test_manim_plugis_subcommand_no_subcommand():
    a = call_command(
        [
            sys.executable,
            "-m",
            "manim",
            "plugins",
        ]
    )
    assert "No flag provided; Exiting..." in a.stdout


def test_manim_plugis_subcommand_listing(function_like_plugin):
    # Check whether `test_plugin` is in plugins list
    a = call_command([sys.executable, "-m", "manim", "plugins", "--list"])
    assert "test_plugin" in a.stdout
