"""Print a Manim timeline and optionally write an HTML view of it.

Example: python examples/timeline_reader.py timeline.json --source-root example_scenes --html timeline.html
"""

import argparse
import hashlib
import html
import json
from pathlib import Path


def read_timeline(path):
    """Read a timeline file and check its schema and version."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "manim.timeline" or data.get("version") != 1:
        raise ValueError("Expected a manim.timeline document with version 1.")
    return data


def source_status(data, root):
    """Describe whether the primary source file still matches the timeline."""
    source = data["source"]
    if root is None or source is None:
        return "source not checked"
    path = Path(root) / source["path"]
    if not path.is_file():
        return "source not found"
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        return "source changed since the timeline was recorded"
    return "source unchanged"


def location(entry):
    source = entry["source"] or {}
    return f"{source.get('path') or '?'}:{source.get('line') or '?'}"


def describe(entry):
    kind = entry["kind"]
    if kind in ("play", "wait"):
        span = f"{entry['start']:.3f}–{entry['end']:.3f}s"
        return f"{kind} {span}" + (" (skipped)" if entry["skipped"] else "")
    if kind == "section":
        return f"section {entry['name']!r} at {entry['start']:.3f}s"
    if kind == "caption":
        return f"caption {entry['content']!r} {entry['start']:.3f}–{entry['end']:.3f}s"
    length = "" if entry["duration"] is None else f" for {entry['duration']:.3f}s"
    return f"sound {entry['file']} at {entry['start']:.3f}s{length}"


def render_html(data, status, root=None):
    total = max(data["duration"], 1e-9)
    rows = []
    for entry in data["entries"]:
        label = html.escape(f"{describe(entry)} · {location(entry)}")
        source = entry["source"]
        if root is not None and source and source["path"]:
            url = (
                f"{(Path(root) / source['path']).resolve().as_uri()}#L{source['line']}"
            )
            label = f'<a href="{html.escape(url, quote=True)}">{label}</a>'
        bar = ""
        end = entry.get("end")
        if entry["kind"] == "sound" and entry["duration"] is not None:
            end = entry["start"] + entry["duration"]
        if end is not None and entry["kind"] != "section":
            # Sounds and captions can start before the scene or end after it.
            left = min(max(0.0, 100 * entry["start"] / total), 100.0)
            width = max(0.0, min(100 * end / total, 100.0) - left)
            bar = (
                f"<div class='track'><span class='{entry['kind']}' "
                f"style='margin-left:{left:.4f}%;width:{width:.4f}%'></span></div>"
            )
        rows.append(f"<li>{label}{bar}</li>")
    return (
        "<!doctype html><meta charset='utf-8'><title>Manim timeline</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2em auto}"
        ".track{background:#eee}.track span{display:block;height:12px;"
        "background:#2980b9}.track span.caption{background:#8e44ad}"
        ".track span.sound{background:#27ae60}"
        "li{margin:1em 0}</style>"
        f"<h1>{html.escape(data['scene'])}</h1><p>{html.escape(status)}</p>"
        f"<ol>{''.join(rows)}</ol>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Directory of the scene's source file, used to check it and link to it.",
    )
    parser.add_argument("--html", type=Path, help="Write an HTML view to this file.")
    args = parser.parse_args()
    data = read_timeline(args.timeline)
    status = source_status(data, args.source_root)
    print(f"{data['scene']}: {data['duration']:.3f}s ({data['mode']}); {status}")
    for entry in data["entries"]:
        print(f"{describe(entry)} · {location(entry)}")
    if args.html:
        args.html.write_text(
            render_html(data, status, args.source_root), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
