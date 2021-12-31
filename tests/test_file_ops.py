import os

from manim import *

from .assert_utils import assert_dir_exists, assert_file_not_exists
from .utils.video_tester import *


def test_guarantee_existence(tmp_path):
    test_dir = os.path.join(tmp_path, "test")
    guarantee_existence(test_dir)
    # test if file dir got created
    assert_dir_exists(test_dir)
    f = open(os.path.join(test_dir, "test.txt"), "x")
    f.close()
    # test if file didn't get deleted
    guarantee_existence(test_dir)


def test_guarantee_empty_existence(tmp_path):
    test_dir = os.path.join(tmp_path, "test")
    os.mkdir(test_dir)
    f = open(os.path.join(test_dir, "test.txt"), "x")
    f.close()

    guarantee_empty_existence(test_dir)
    # test if dir got created
    assert_dir_exists(test_dir)
    # test if dir got cleaned
    assert_file_not_exists(os.path.join(test_dir, "test.txt"))
