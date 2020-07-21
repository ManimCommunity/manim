import subprocess
import os
from shutil import rmtree
import pytest
import re

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
    log_file_path=os.path.join(path_output, "logs", "SquareToCircle.log")
    assert exitcode == 0, err
    assert os.path.exists(log_file_path), err
    with open (log_file_path) as logfile:
        if os.sep =="\\":
            logs=re.sub(r"(\d{2}:\d{2}:\d{2})|([A-Z]:\\.*) +","",logfile.read())
        else:
            logs=re.sub(r"(\d{2}:\d{2}:\d{2})|(\.?/.+) +","",logfile.read())
    with open(os.path.join(os.path.dirname(__file__), "expected.log")) as ideal:
        expected=ideal.read()
    assert logs==expected, logs
    rmtree(path_output)
