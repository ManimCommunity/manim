from __future__ import annotations

from pathlib import Path

from manim import *
from tests.assert_utils import assert_dir_exists, assert_file_not_exists
from tests.utils.video_tester import *


def test_guarantee_existence(tmp_path: Path):
    test_dir = tmp_path / "test"
    guarantee_existence(test_dir)
    # test if file dir got created
    assert_dir_exists(test_dir)
    with open(test_dir / "test.txt", "x") as f:
        pass
    # test if file didn't get deleted
    guarantee_existence(test_dir)


def test_guarantee_empty_existence(tmp_path: Path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    with open(test_dir / "test.txt", "x"):
        pass

    guarantee_empty_existence(test_dir)
    # test if dir got created
    assert_dir_exists(test_dir)
    # test if dir got cleaned
    assert_file_not_exists(test_dir / "test.txt")
