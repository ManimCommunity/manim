"""Interactive IPython embed loop for the WebGPU renderer.

Called by :meth:`~manim.scene.scene.Scene.interactive_embed` when the WebGPU
renderer is active.  Mirrors the OpenGL :meth:`~manim.scene.scene.Scene.interact`
loop but drives the ``rendercanvas`` event loop instead of moderngl-window.

Thread model
------------
* **Main thread** — window event loop + scene method execution.  Drains
  ``scene.queue`` and calls scene methods (``play``, ``add``, …) so that all
  GPU work stays on the thread that owns the WebGPU device.
* **IPython thread** (daemon) — blocking readline / prompt_toolkit.  Scene
  method calls are not executed here; they are posted to ``scene.queue`` and
  picked up by the main thread.

After every IPython cell ``post_run_cell`` schedules a re-render sentinel on
``scene.queue`` so the window immediately reflects any changes (``add``,
``remove``, property mutations …) that did not go through ``play`` / ``wait``.

File watching
-------------
A ``watchdog.Observer`` watches the scene's source file.  When the file is
saved on disk the observer posts ``SceneInteractRerun("file")`` to
``scene.queue``.  The main loop then tears down the IPython session and raises
:class:`~manim.utils.exceptions.RerunSceneException`, which propagates through
``construct()`` back to the render command's rerun loop.

``rerun()`` in the IPython shell triggers the same mechanism manually.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manim.scene.scene import Scene

    from .webgpu_renderer import WebGPURenderer

# Frames per second at which the main loop polls OS events when idle (no
# scene method is running).  Higher values make mouse / keyboard more
# responsive; lower values save CPU.
_POLL_HZ: int = 60


def interactive_embed(
    scene: Scene,
    renderer: WebGPURenderer,
    local_namespace: dict[str, Any],
) -> bool:
    """Run an interactive IPython session alongside the WebGPU preview window.

    Parameters
    ----------
    scene:
        The running scene instance.
    renderer:
        The active :class:`~.WebGPURenderer`.
    local_namespace:
        The caller's (``construct()``'s) local variables — captured in
        :meth:`~manim.scene.scene.Scene.interactive_embed` before this
        function is called.  Scene shortcuts (``play``, ``wait``, ``add``,
        ``remove``) and the full ``manim`` namespace are injected here so
        the user can type commands without a ``self.`` prefix.

    Returns
    -------
    bool
        ``True`` if the session ended because ``rerun()`` was called or the
        source file changed on disk — the caller should raise
        :class:`~manim.utils.exceptions.RerunSceneException` in that case.
        ``False`` for a normal exit (shell closed / window closed).
    """
    import threading

    import manim
    from manim import config, logger
    from manim.data_structures import MethodWithArgs
    from manim.scene.scene import SceneInteractContinue, SceneInteractRerun

    window = renderer.window

    # ── IPython imports ──────────────────────────────────────────────────
    try:
        from sqlite3 import connect

        from IPython.core.getipython import get_ipython
        from IPython.terminal.embed import InteractiveShellEmbed
        from traitlets.config import Config as IPConfig
    except ImportError:
        logger.error(
            "IPython is required for the interactive WebGPU embed.\n"
            "Install it with:  pip install ipython",
        )
        return False

    # ── File watcher ─────────────────────────────────────────────────────
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class _FileHandler(FileSystemEventHandler):
            def on_modified(self, event: Any) -> None:
                scene.queue.put(SceneInteractRerun("file"))

        file_observer = Observer()
        file_observer.schedule(_FileHandler(), config["input_file"], recursive=True)
        file_observer.start()
    except Exception:
        logger.debug("watchdog not available — file-watching disabled.")
        file_observer = None

    # ── Build shell ──────────────────────────────────────────────────────
    ipcfg = IPConfig()
    ipcfg.TerminalInteractiveShell.confirm_exit = False

    existing = get_ipython()
    if existing is None:
        shell = InteractiveShellEmbed.instance(config=ipcfg)
    else:
        shell = InteractiveShellEmbed(config=ipcfg)

    # Make the SQLite history database thread-safe so IPython history works
    # correctly from the daemon keyboard thread.
    hist = get_ipython().history_manager
    hist.db = connect(hist.hist_file, check_same_thread=False)

    # ── Populate namespace ───────────────────────────────────────────────
    # Pre-import the full manim namespace so users don't need import
    # statements inside the session.
    for name in dir(manim):
        local_namespace[name] = getattr(manim, name)

    # Proxy scene methods: posting to scene.queue keeps all GPU work on the
    # main thread (matching the OpenGL embedded_method pattern).
    def _make_proxy(method_name: str):
        method = getattr(scene, method_name)

        def _proxy(*args: Any, **kwargs: Any) -> None:
            scene.queue.put(MethodWithArgs(method, args, kwargs))

        _proxy.__name__ = method_name
        return _proxy

    for _name in ("play", "wait", "add", "remove"):
        local_namespace[_name] = _make_proxy(_name)

    # rerun(): tear down this session and re-run the scene from scratch,
    # picking up any edits saved to the source file.
    def _rerun(*args: Any, **kwargs: Any) -> None:
        scene.queue.put(SceneInteractRerun("keyboard"))
        shell.exiter()

    local_namespace["rerun"] = _rerun

    # ── After every cell: schedule a re-render on the main thread ────────
    # _post_cell runs in the IPython thread — GPU calls must not happen here.
    # Posting a sentinel to scene.queue ensures update_frame() is called on
    # the main thread, which owns the WebGPU device.
    _RENDER = object()  # sentinel: "please re-render"

    def _post_cell(*_a: Any, **_kw: Any) -> None:
        scene.queue.put(_RENDER)

    shell.events.register("post_run_cell", _post_cell)

    # ── IPython thread ───────────────────────────────────────────────────
    def _keyboard_thread() -> None:
        shell(local_ns=local_namespace)
        # Signal the main loop that the user closed the shell (not a rerun).
        scene.queue.put(SceneInteractContinue("keyboard"))

    keyboard_thread = threading.Thread(target=_keyboard_thread)
    # Run as a daemon so the thread is killed if the main thread exits
    # (e.g. the window is closed before the shell prompt is answered).
    if not shell.pt_app:
        keyboard_thread.daemon = True
    keyboard_thread.start()

    # ── Helpers ──────────────────────────────────────────────────────────
    def _stop_file_observer() -> None:
        if file_observer is not None:
            file_observer.unschedule_all()
            file_observer.stop()
            file_observer.join()

    def _exit_keyboard_thread() -> None:
        if shell.pt_app:
            try:
                shell.pt_app.app.exit(exception=EOFError)
            except Exception:
                pass
        keyboard_thread.join()
        while not scene.queue.empty():
            scene.queue.get()

    # ── Main thread: event loop ──────────────────────────────────────────
    scene.quit_interaction = False
    keyboard_thread_needs_join = shell.pt_app is not None
    sleep_s = 1.0 / _POLL_HZ
    rerun_requested = False

    while not (window.is_closing or scene.quit_interaction):
        if not scene.queue.empty():
            action = scene.queue.get_nowait()

            if isinstance(action, SceneInteractRerun):
                rerun_requested = True
                _stop_file_observer()
                if action.sender == "keyboard":
                    # rerun() was called from the shell — thread already
                    # exiting via shell.exiter(); just join it.
                    keyboard_thread.join()
                else:
                    # File changed — kill the prompt and join.
                    _exit_keyboard_thread()
                # Drain stale queue items and signal rerun to the caller.
                while not scene.queue.empty():
                    scene.queue.get()
                break

            elif isinstance(action, SceneInteractContinue):
                # IPython shell exited normally (user typed exit / Ctrl-D).
                keyboard_thread.join()
                while not scene.queue.empty():
                    scene.queue.get()
                keyboard_thread_needs_join = False
                break

            elif isinstance(action, MethodWithArgs):
                # Execute the proxied scene method on the main thread.
                action.method(*action.args, **action.kwargs)
                # Re-render so the result is visible immediately.
                # (play/wait already render internally; add/remove do not.)
                if renderer._device is not None:
                    renderer.update_frame(scene)
                    window._canvas.force_draw()

            elif action is _RENDER:
                # Triggered by _post_cell — re-render after any IPython cell
                # that mutated the scene without going through a proxy method.
                if renderer._device is not None:
                    renderer.update_frame(scene)
                    window._canvas.force_draw()
        else:
            # Idle — process OS events so mouse/keyboard controls stay live.
            window._canvas._process_events()
            time.sleep(sleep_s)

    # ── Teardown ─────────────────────────────────────────────────────────
    _stop_file_observer()

    if keyboard_thread_needs_join:
        _exit_keyboard_thread()

    if window.is_closing:
        window.destroy()

    return rerun_requested
