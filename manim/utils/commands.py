from __future__ import annotations

import json
import os
from pathlib import Path
from subprocess import run
from typing import Generator

__all__ = [
    "capture",
    "get_video_metadata",
    "get_dir_layout",
]


def capture(command, cwd=None, command_input=None):
    p = run(
        command,
        cwd=cwd,
        input=command_input,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    out, err = p.stdout, p.stderr
    return out, err, p.returncode


def get_video_metadata(path_to_video: str | os.PathLike) -> dict[str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration,avg_frame_rate,codec_name",
        "-print_format",
        "json",
        str(path_to_video),
    ]
    config, err, exitcode = capture(command)
    assert exitcode == 0, f"FFprobe error: {err}"
    return json.loads(config)["streams"][0]


def get_dir_layout(dirpath: Path) -> Generator[str, None, None]:
    """Get list of paths relative to dirpath of all files in dir and subdirs recursively."""
    for p in dirpath.iterdir():
        if p.is_dir():
            yield from get_dir_layout(p)
            continue
        yield str(p.relative_to(dirpath))
