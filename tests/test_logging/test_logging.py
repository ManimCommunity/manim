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
    """Test logging Terminal output to a log file.
    `rich` formats it's output based on the size of the terminal it is outputting to.
    As such, since there is no way to obtain the terminal size of the testing device
    before running the test, this test employs a workaround where instead of the exact
    text of the log (which can differ in whitespace and truncation) only the constant
    parts, such as keywords, are compared.
    """
    path_basic_scene = os.path.join("tests", "tests_data", "basic_scenes.py")
    expected=['INFO', 'Read', 'configuration', 'files:', 'config.py:', 'INFO',
        'scene_file_writer.py:', 'File', 'ready', 'at']
    path_output = os.path.join("tests_cache", "media_temp")
    command = [python_version, "-m", "manim", path_basic_scene,
               "SquareToCircle", "-l", "--log_to_file", "--log_dir",os.path.join(path_output,"logs"), "--media_dir", path_output]
    out, err, exitcode = capture(command)
    log_file_path=os.path.join(path_output, "logs", "SquareToCircle.log")
    assert exitcode == 0, err
    assert os.path.exists(log_file_path), err
    if sys.platform.startswith("win32") or sys.platform.startswith("cygwin"):
        enc="Windows-1252"
    else:
        enc="utf-8"
    with open(log_file_path,encoding=enc) as logfile:
        logs=logfile.read().split()
    logs=[e for e in logs if not any(x in e for x in ["\\","/",".mp4","[","]"])]
    logs=[re.sub('[0-9]', '', i) for i in logs]
    assert logs==expected, err
