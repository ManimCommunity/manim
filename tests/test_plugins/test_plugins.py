from __future__ import annotations

import random
import string
import textwrap
from pathlib import Path

import pytest

from manim import capture

plugin_pyproject_template = textwrap.dedent(
    """\
    [project]
    name = "{plugin_name}"
    authors = [{name = "ManimCE Test Suite"},]
    version = "0.1.0"
    description = "A fantastic Manim plugin"
    requires-python = ">=3.9"

    [project.entry-points."manim.plugins"]
    "{plugin_name}" = "{plugin_entrypoint}"

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"
    """,
)

plugin_init_template = textwrap.dedent(
    """\
    from manim import *
    {all_dec}
    class {class_name}(VMobject):
        def __init__(self):
            super().__init__()
            dot1 = Dot(fill_color=GREEN).shift(LEFT)
            dot2 = Dot(fill_color=BLUE)
            dot3 = Dot(fill_color=RED).shift(RIGHT)
            self.dotgrid = VGroup(dot1, dot2, dot3)
            self.add(self.dotgrid)

        def update_dot(self):
            self.dotgrid.become(self.dotgrid.shift(UP))
    def {function_name}():
        return [{class_name}]
    """,
)

cfg_file_contents = textwrap.dedent(
    """\
        [CLI]
        plugins = {plugin_name}
    """,
)


@pytest.fixture
def simple_scenes_path():
    return Path(__file__).parent / "simple_scenes.py"


def cfg_file_create(cfg_file_contents, path):
    file_loc = (path / "manim.cfg").absolute()
    file_loc.write_text(cfg_file_contents)
    return file_loc


@pytest.fixture
def random_string():
    all_letters = string.ascii_lowercase
    a = random.Random()
    final_letters = [a.choice(all_letters) for _ in range(8)]
    return "".join(final_letters)


def test_plugin_warning(tmp_path, python_version, simple_scenes_path):
    cfg_file = cfg_file_create(
        cfg_file_contents.format(plugin_name="DNEplugin"),
        tmp_path,
    )
    scene_name = "SquareToCircle"
    command = [
        python_version,
        "-m",
        "manim",
        "-ql",
        "--media_dir",
        str(cfg_file.parent),
        "--config_file",
        str(cfg_file),
        str(simple_scenes_path),
        scene_name,
    ]
    out, err, exit_code = capture(command, cwd=str(cfg_file.parent))
    assert exit_code == 0, err
    assert "Missing Plugins" in out, "Missing Plugins isn't in Output."


@pytest.fixture
def create_plugin(tmp_path, python_version, random_string):
    plugin_dir = tmp_path / "plugin_dir"
    plugin_name = random_string

    def _create_plugin(entry_point, class_name, function_name, all_dec=""):
        entry_point = entry_point.format(plugin_name=plugin_name)
        module_dir = plugin_dir / plugin_name
        module_dir.mkdir(parents=True)
        (module_dir / "__init__.py").write_text(
            plugin_init_template.format(
                class_name=class_name,
                function_name=function_name,
                all_dec=all_dec,
            ),
        )
        (plugin_dir / "pyproject.toml").write_text(
            plugin_pyproject_template.format(
                plugin_name=plugin_name,
                plugin_entrypoint=entry_point,
            ),
        )
        command = [
            python_version,
            "-m",
            "pip",
            "install",
            str(plugin_dir.absolute()),
        ]
        out, err, exit_code = capture(command, cwd=str(plugin_dir))
        print(out)
        assert exit_code == 0, err
        return {
            "module_dir": module_dir,
            "plugin_name": plugin_name,
        }

    yield _create_plugin
    command = [python_version, "-m", "pip", "uninstall", plugin_name, "-y"]
    out, err, exit_code = capture(command)
    print(out)
    assert exit_code == 0, err
