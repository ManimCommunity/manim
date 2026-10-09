from __future__ import annotations

import types

import pytest

from manim.utils import module_ops


@pytest.fixture
def tmp_files(tmp_path, monkeypatch):
    """
    File structure:
    -tmp_path (cwd)
        -parent_dir (package_root)
             __init__.py
             parent_file.py
             -sub_dir
                 __init__.py
                 scene.py
    """
    parent_dir = tmp_path / "parent_dir"
    sub_dir = parent_dir / "sub_dir"
    sub_dir.mkdir(parents=True)

    # Change cwd to tmp_path
    monkeypatch.chdir(tmp_path)

    (tmp_path / "parent_dir" / "__init__.py").write_text("")
    (tmp_path / "parent_dir" / "sub_dir" / "__init__.py").write_text("")
    parent_file = tmp_path / "parent_dir" / "parent.py"
    scene_file = tmp_path / "parent_dir" / "sub_dir" / "scene.py"
    parent_file.write_text("a = 42")
    scene_file.write_text("")
    return parent_file, scene_file


def test_get_module_from_file_returns_valid_spec(tmp_files):
    module = module_ops.get_module(tmp_files[1])

    assert isinstance(module, types.ModuleType)
    assert module.__spec__.name == "parent_dir.sub_dir.scene"
    assert module.__spec__.loader is not None


def test_relative_import_works_from_subdir_to_parent(tmp_files):
    parent_file, scene_file = tmp_files
    scene_file.write_text("from ..parent import a\nresult = a")

    module = module_ops.get_module(scene_file)

    assert hasattr(module, "result")
    assert module.result == 42


def test_absolute_imports_outside_cwd(tmp_path):
    file = tmp_path / "scene.py"
    file.write_text("")
    module = module_ops.get_module(file)

    assert module.__spec__.name == "scene"
    assert module.__spec__.loader is not None
