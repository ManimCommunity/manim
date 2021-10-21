import random
import string
import tempfile
import textwrap
from pathlib import Path

import pytest

from manim import capture

plugin_pyproject_template = textwrap.dedent(
    """\
    [tool.poetry]
    name = "{plugin_name}"
    authors = ["ManimCE Test Suite"]
    version = "0.1.0"
    description = ""

    [tool.poetry.dependencies]
    python = "^3.7"

    [tool.poetry.plugins."manim.plugins"]
    "{plugin_name}" = "{plugin_entrypoint}"

    [build-system]
    requires = ["poetry-core>=1.0.0"]
    build-backend = "poetry.core.masonry.api"
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
    yield str(Path(__file__).parent / "simple_scenes.py")


def cfg_file_create(cfg_file_contents, path):
    file_loc = (path / "manim.cfg").absolute()
    with open(file_loc, "w") as f:
        f.write(cfg_file_contents)
    return file_loc


@pytest.fixture
def random_string():
    all_letters = string.ascii_lowercase
    a = random.Random()
    final_letters = [a.choice(all_letters) for _ in range(8)]
    yield "".join(final_letters)


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
        simple_scenes_path,
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
        with open(module_dir / "__init__.py", "w") as f:
            f.write(
                plugin_init_template.format(
                    class_name=class_name,
                    function_name=function_name,
                    all_dec=all_dec,
                ),
            )
        with open(plugin_dir / "pyproject.toml", "w") as f:
            f.write(
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


@pytest.mark.slow
def test_plugin_function_like(
    tmp_path,
    create_plugin,
    python_version,
    simple_scenes_path,
):
    function_like_plugin = create_plugin(
        "{plugin_name}.__init__:import_all",
        "FunctionLike",
        "import_all",
    )
    cfg_file = cfg_file_create(
        cfg_file_contents.format(plugin_name=function_like_plugin["plugin_name"]),
        tmp_path,
    )
    scene_name = "FunctionLikeTest"
    command = [
        python_version,
        "-m",
        "manim",
        "-ql",
        "--media_dir",
        str(cfg_file.parent),
        "--config_file",
        str(cfg_file),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command, cwd=str(cfg_file.parent))
    print(out)
    print(err)
    assert exit_code == 0, err


@pytest.mark.slow
def test_plugin_no_all(tmp_path, create_plugin, python_version):
    create_plugin = create_plugin("{plugin_name}", "NoAll", "import_all")
    plugin_name = create_plugin["plugin_name"]
    cfg_file = cfg_file_create(
        cfg_file_contents.format(plugin_name=plugin_name),
        tmp_path,
    )
    test_class = textwrap.dedent(
        f"""\
        from manim import *
        class NoAllTest(Scene):
            def construct(self):
                assert "{plugin_name}" in globals()
                a = {plugin_name}.NoAll()
                self.play(FadeIn(a))
        """,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        delete=False,
    ) as tmpfile:
        tmpfile.write(test_class)
    scene_name = "NoAllTest"
    command = [
        python_version,
        "-m",
        "manim",
        "-ql",
        "--media_dir",
        str(cfg_file.parent),
        "--config_file",
        str(cfg_file),
        tmpfile.name,
        scene_name,
    ]
    out, err, exit_code = capture(command, cwd=str(cfg_file.parent))
    print(out)
    print(err)
    assert exit_code == 0, err
    Path(tmpfile.name).unlink()


@pytest.mark.slow
def test_plugin_with_all(tmp_path, create_plugin, python_version, simple_scenes_path):
    create_plugin = create_plugin(
        "{plugin_name}",
        "WithAll",
        "import_all",
        all_dec="__all__=['WithAll']",
    )
    plugin_name = create_plugin["plugin_name"]
    cfg_file = cfg_file_create(
        cfg_file_contents.format(plugin_name=plugin_name),
        tmp_path,
    )
    scene_name = "WithAllTest"
    command = [
        python_version,
        "-m",
        "manim",
        "-ql",
        "--media_dir",
        str(cfg_file.parent),
        "--config_file",
        str(cfg_file),
        simple_scenes_path,
        scene_name,
    ]
    out, err, exit_code = capture(command, cwd=str(cfg_file.parent))
    print(out)
    print(err)
    assert exit_code == 0, err
