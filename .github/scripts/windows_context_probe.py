"""Windows-only diagnostic; run each case in a fresh process, without media output.

From the lifecycle checkout with its existing environment:
    uv run python PATH/TO/windows-context-probe.py
    uv run python PATH/TO/windows-context-probe.py --case snapshot --restore set-current
    uv run python PATH/TO/windows-context-probe.py --case snapshot --restore native

The two restoration modes are experiments, not proposed product implementations.
Do not resolve wglChoosePixelFormatARB early: doing so could hide the failing lookup.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

CASES = ("fresh", "raw-release", "snapshot", "render-release", "preview-cycle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--restore", choices=("none", "set-current", "native"), default="none")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This diagnostic requires Windows; a macOS run cannot validate WGL.")
    if args.case is None:
        failed = []
        for case in CASES:
            print(f"\n=== {case}, restoration={args.restore} ===", flush=True)
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--case", case,
                 "--restore", args.restore],
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=45,
            )
            if result.returncode:
                failed.append(case)
        print(json.dumps({"failed_cases": failed}), flush=True)
        return bool(failed)

    # Read native handles without resolving any extension function or changing context.
    dll = ctypes.WinDLL("opengl32")
    for name in ("wglGetCurrentContext", "wglGetCurrentDC"):
        function = getattr(dll, name)
        function.argtypes = []
        function.restype = ctypes.c_void_p
    dll.wglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.wglMakeCurrent.restype = ctypes.c_int
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint
    name = ctypes.create_unicode_buffer(32768)
    if not kernel.GetModuleFileNameW(dll._handle, name, len(name)):
        raise ctypes.WinError(ctypes.get_last_error())

    import manim
    import moderngl
    from manim import Manager, Scene, Square, tempconfig
    # This creates Pyglet's shadow context on Windows, as the unit-test imports do.
    from manim.renderer.opengl.window import Window  # noqa: F401
    from pyglet import gl
    from pyglet.gl import lib_wgl

    def emit(stage, **extra):
        context = gl.current_context
        print(json.dumps({
            "stage": stage, "case": args.case, "restore": args.restore,
            "native_context": dll.wglGetCurrentContext(),
            "native_dc": dll.wglGetCurrentDC(),
            "pyglet_context": repr(context),
            "pyglet_native_context": ctypes.cast(
                getattr(context, "_context", None), ctypes.c_void_p).value,
            **extra,
        }), flush=True)

    # Observe the *original* first lookup, not a warm-up query.
    lookup = lib_wgl.wglGetProcAddress

    def record_lookup(function_name):
        address = lookup(function_name)
        if function_name in (b"wglChoosePixelFormatARB", "wglChoosePixelFormatARB"):
            emit("choose-pixel-format-lookup", address=ctypes.cast(address, ctypes.c_void_p).value)
        return address

    lib_wgl.wglGetProcAddress = record_lookup
    emit("imported", manim_source=manim.__file__, gl_dll=name.value,
         python=sys.version, versions={package: importlib.metadata.version(package)
             for package in ("pyglet", "moderngl", "moderngl-window", "glcontext")})

    def render(live):
        with tempconfig({"live_preview": live}):
            scene = Scene()
            scene.add(Square())
            with Manager(scene) as manager:
                manager.render()
                emit("rendered-preview" if live else "rendered-standalone")
            emit("closed-preview" if live else "closed-standalone")

    try:
        with tempfile.TemporaryDirectory(prefix="manim-wgl-probe-") as media:
            with tempconfig({"renderer": "opengl", "format": "none", "live_preview": False,
                             "pixel_width": 64, "pixel_height": 64, "frame_rate": 4,
                             "window_size": (64, 64), "media_dir": media}):
                if args.case == "raw-release":
                    context = moderngl.create_context(standalone=True)
                    emit("raw-standalone-open")
                    context.release()
                elif args.case == "snapshot":
                    scene = Scene()
                    scene.add(Square())
                    with Manager(scene):
                        assert scene.get_image().size == (64, 64)
                elif args.case == "render-release":
                    render(False)
                elif args.case == "preview-cycle":
                    render(True)
                    render(False)
                emit("before-preview")
                if args.restore != "none":
                    context = gl.current_context
                    assert context is not None, "No Pyglet context to restore"
                    if args.restore == "set-current":
                        context.set_current()
                    else:
                        # Experimental control: rebind the already-existing context.
                        # Do not change Pyglet's globals or create a new bootstrap context.
                        assert dll.wglMakeCurrent(context.canvas.hdc, context._context)
                    emit("after-experimental-restore")
                render(True)
        emit("passed")
        return 0
    except BaseException:
        emit("failed")
        traceback.print_exc()
        return 1
    finally:
        lib_wgl.wglGetProcAddress = lookup


if __name__ == "__main__":
    raise SystemExit(main())
