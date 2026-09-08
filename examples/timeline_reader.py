"""Read an experimental timeline and optionally create an HTML view.

Uses Python's standard library to read the report and check its primary source file.
Example: python examples/timeline_reader.py report.json --source-root path/to/scenes --html view.html
"""

import argparse
import hashlib
import html
import json
from pathlib import Path


def read_timeline(path):
    """Check the version, completion flag and content hash, then return the data."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data.get("complete") is not True
    ):
        raise ValueError("Unsupported timeline document")
    if (data.get("schema"), data.get("version"), data.get("complete")) != (
        "manim.execution-timeline",
        1,
        True,
    ):
        raise ValueError("Unsupported or incomplete timeline")
    revision = data.pop("revision", None)
    content = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != revision:
        raise ValueError("Timeline content revision mismatch")
    data["revision"] = revision
    return data


def source_status(data, root):
    """Compare the primary file on disk with its recorded hash, when available."""
    source = data["source"]
    if root is None or source["path"] is None:
        return "source unverified"
    root = Path(root).resolve()
    path = (root / source["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return "source unavailable"
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        return "source STALE"
    if (
        source.get("provenance")
        == "captured-before-loading-primary-bytes-compiled-directly"
    ):
        return "source matches captured bytes (primary compiled directly; dependencies unverified)"
    return "source matches captured bytes (loaded-code identity not proven)"


def render_html(data, status, root=None):
    total = max(data["end"] - data["start"], 1e-9)
    rows = []
    for event in data["events"]:
        source = event["source"] or {}
        location = f"{source.get('path') or '?'}:{source.get('line') or '?'}"
        left = 100 * (event["start"] - data["start"]) / total
        width = 100 * (event["end"] - event["start"]) / total
        label = f"{event['id']} {event['kind']} {event['start']:.3f}–{event['end']:.3f}s {location}"
        label_html = html.escape(label)
        if root is not None and source.get("path") and source.get("line"):
            directory = Path(root).resolve()
            target = (directory / source["path"]).resolve()
            if target.is_relative_to(directory):
                url = f"{target.as_uri()}#L{int(source['line'])}"
                label_html = (
                    f'<a href="{html.escape(url, quote=True)}">{label_html}</a>'
                )
        rows.append(
            f"<li>{label_html}<div class='track'><span style='margin-left:{left:.4f}%;width:{width:.4f}%'></span></div></li>"
        )
    declarations = html.escape(
        json.dumps(data["declarations"], indent=2, ensure_ascii=False)
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>Execution timeline</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2em auto}.track{background:#eee}"
        ".track span{display:block;height:12px;background:#2980b9}li{margin:1em 0}</style>"
        f"<h1>{html.escape(data['scene']['name'])}</h1><p>{html.escape(status)}</p>"
        f"<ol>{''.join(rows)}</ol><h2>Declarations (sound duration may be unknown)</h2><pre>{declarations}</pre>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    data = read_timeline(args.timeline)
    status = source_status(data, args.source_root)
    print(f"{data['scene']['name']}: {data['end'] - data['start']:.3f}s; {status}")
    for event in data["events"]:
        source = event["source"] or {}
        print(
            f"{event['id']} {event['kind']} {event['start']:.3f}..{event['end']:.3f} "
            f"{source.get('path') or '?'}:{source.get('line') or '?'}"
        )
    for declaration in data["declarations"]:
        print(f"{declaration['kind']} at {declaration['at']:.3f}")
    if args.html:
        args.html.write_text(
            render_html(data, status, args.source_root), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
