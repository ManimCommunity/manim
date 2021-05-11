import sys
import threading
from pathlib import Path
from textwrap import dedent

from manim.utils.lockfile import LockMediaDir

from .utils.commands import capture


def test_locking_using_threading(python_version, tmpdir, capsys):
    test_file = Path(tmpdir) / "test.py"
    with test_file.open("w") as f:
        f.write(
            dedent(
                """\
            from manim import Scene, Text

            class Sample(Scene):
                def construct(self):
                    self.add(Text("hello"))"""
            )
        )

    def run_manim():
        out, _, statuscode = capture(
            [
                python_version,
                "-m",
                "manim",
                "-ql",
                f"--media_dir={tmpdir}",
                str(test_file),
            ],
            cwd=tmpdir,
        )
        if statuscode == 1:
            sys.stderr.write("".join([i.strip() for i in out.strip().split("\n")]))

    t1 = threading.Thread(target=run_manim)
    t2 = threading.Thread(target=run_manim)

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    captured = capsys.readouterr()
    assert (
        "Some other Manim process is already using this media directory" in captured.err
    )


def test_LockMediaDir(tmpdir):
    t1 = Path(tmpdir)
    with LockMediaDir(tmpdir, True):
        lock_file = t1 / "manim.lock"
        assert lock_file.exists()
    t2 = t1 / "temp"
    with LockMediaDir(t2, False):
        lock_file = t2 / "manim.lock"
        assert not lock_file.exists()
