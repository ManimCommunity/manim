import sys
import typing as T
from pathlib import Path

from .. import console

__all__ = ["LockMediaDir"]
if sys.platform.startswith("win32"):
    import msvcrt

    fcntl = None
else:
    import fcntl

    msvcrt = None


class LockMediaDir:
    def __init__(self, media_dir: str, lock_media_dir: bool) -> None:
        self.lockfilename = Path(media_dir) / "manim.lock"
        self.media_dir = Path(media_dir).absolute()
        self.lock_media_dir = lock_media_dir

    def __enter__(self) -> None:
        if self.lock_media_dir:
            self.lockfile = self.lockfilename.open("w")
            if msvcrt:
                try:
                    # see https://docs.python.org/3/library/msvcrt.html?highlight=msvcrt#msvcrt.locking
                    msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_NBLCK, 1)
                except (BlockingIOError, PermissionError):
                    self.lockfile.close()
                    console.print(
                        f"[red]Some other Manim process is already using this media directory ({self.media_dir}). Exiting.[/red]",
                    )
                    sys.exit(1)
            else:
                try:
                    fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (BlockingIOError, PermissionError):
                    self.lockfile.close()
                    console.print(
                        f"[red]Some other Manim process is already using this media directory ({self.media_dir}). Exiting.[/red]",
                    )
                    sys.exit(1)

    def __exit__(self, *args: T.Any) -> None:
        if self.lock_media_dir:
            if msvcrt:
                msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.lockfile, fcntl.LOCK_UN)
            self.lockfile.close()
