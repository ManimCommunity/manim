import subprocess
import os
import sys
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
    expected=['INFO', 'Read', 'configuration', 'files:', 'config.py:92', 'INFO',
        'scene_file_writer.py:531', 'File', 'ready', 'at']
    path_output = os.path.join("tests_cache", "media_temp")
    command = [python_version, "-m", "manim", path_basic_scene,
               "SquareToCircle", "-l", "--log_to_file", "--log_dir",os.path.join(path_output,"logs"), "--media_dir", path_output]
    out, err, exitcode = capture(command)
    log_file_path=os.path.join(path_output, "logs", "SquareToCircle.log")
    assert exitcode == 0, err
    assert os.path.exists(log_file_path), err
    if sys.platform.startswith("win32") or sys.platform.startswith("win32"):
        enc="Windows-1252"
    else:
        enc="utf-8"
    with open(log_file_path,encoding=enc) as logfile:
        logs=logfile.read()
    logs=[e for e in logs if not any(x in e for x in ["\\","/",".mp4","[","]"])]
    assert logs==expected, err
