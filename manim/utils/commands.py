import json
from subprocess import run
from typing import Any, Dict

__all__ = [
    "capture",
    "get_video_metadata",
]


def capture(command, cwd=None, command_input=None):
    p = run(command, cwd=cwd, input=command_input, capture_output=True, text=True)
    out, err = p.stdout, p.stderr
    return out, err, p.returncode


def get_video_metadata(path_to_video: str) -> Dict[str, Any]:
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
        path_to_video,
    ]
    config, err, exitcode = capture(command)
    assert exitcode == 0, f"FFprobe error: {err}"
    return json.loads(config)["streams"][0]
