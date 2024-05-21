from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from subprocess import run

import av

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
    with av.open(str(path_to_video)) as container:
        stream = container.streams.video[0]
        ctxt = stream.codec_context
        rate = stream.average_rate
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
            num_frames = stream.frames
        else:
            num_frames = sum(1 for _ in container.decode(video=0))
            duration = float(num_frames / stream.base_rate)

        return {
            "width": ctxt.width,
            "height": ctxt.height,
            "nb_frames": str(num_frames),
            "duration": f"{duration:.6f}",
            "avg_frame_rate": f"{rate.numerator}/{rate.denominator}",  # Can be a Fraction
            "codec_name": stream.codec_context.name,
            "pix_fmt": stream.codec_context.pix_fmt,
        }


def get_dir_layout(dirpath: Path) -> Generator[str, None, None]:
    """Get list of paths relative to dirpath of all files in dir and subdirs recursively."""
    for p in dirpath.iterdir():
        if p.is_dir():
            yield from get_dir_layout(p)
            continue
        yield str(p.relative_to(dirpath))
