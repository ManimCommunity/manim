import subprocess
import os
from shutil import rmtree
import pytest

def capture(command,instream=None):
    proc = subprocess.Popen(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=instream
                            )
    out, err = proc.communicate()
    return out, err, proc.returncode


def test_logging_to_file(python_version):
    """Test logging Terminal output to a log file."""
    path_basic_scene = os.path.join("tests", "tests_data", "basic_scenes.py")
    path_output = os.path.join("tests_cache", "media_temp")
    command = [python_version, "-m", "manim", path_basic_scene,
               "SquareToCircle", "-l", "--log_to_file", "--log_dir",os.path.join(path_output,"logs"), "--media_dir", path_output]
    out, err, exitcode = capture(command)
    assert exitcode == 0, err
    assert os.path.exists(os.path.join(
        path_output, "logs", "SquareToCircle.log")), err
    rmtree(path_output)
