from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, call

from manim import *
from manim.utils import file_ops
from tests.assert_utils import assert_dir_exists, assert_file_not_exists
from tests.utils.video_tester import *


def test_guarantee_existence(tmp_path: Path):
    test_dir = tmp_path / "test"
    guarantee_existence(test_dir)
    # test if file dir got created
    assert_dir_exists(test_dir)
    with (test_dir / "test.txt").open("x") as f:
        pass
    # test if file didn't get deleted
    guarantee_existence(test_dir)


def test_guarantee_empty_existence(tmp_path: Path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    with (test_dir / "test.txt").open("x"):
        pass

    guarantee_empty_existence(test_dir)
    # test if dir got created
    assert_dir_exists(test_dir)
    # test if dir got cleaned
    assert_file_not_exists(test_dir / "test.txt")


def test_open_media_file_can_reveal_and_preview(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "scene.mp4"
    file_writer = Mock(final_file_path=artifact)
    open_file = Mock()
    monkeypatch.setattr(file_ops, "open_file", open_file)

    file_ops.open_media_file(
        file_writer,
        preview=True,
        show_in_file_browser=True,
    )

    assert open_file.call_args_list == [
        call(artifact, True),
        call(artifact, False),
    ]
