Export a scene timeline
=======================

A *timeline* lists the play, wait, section, caption, and sound calls of a scene,
with the scene time at which each call happened and the line of code that made
it. Manim can record a timeline while it renders a scene, or while it
:doc:`evaluates <evaluation>` the scene without rendering, which is much faster.
Tools such as editor previews use timelines to show the structure of a scene.

The format and Python API are **experimental** and may change without a
deprecation period.

From Python
-----------

.. code-block:: python

    from manim import Manager, Scene, tempconfig

    class Example(Scene):
        def construct(self):
            self.next_section("opening")
            self.wait(0.3, frozen_frame=False)
            self.wait(0.3, frozen_frame=True)

    with tempconfig({"frame_rate": 4}):
        scene = Example()
        manager = Manager(scene)
        manager.evaluate(capture_timeline=True)
        for entry in manager.timeline.to_dict()["entries"]:
            print(entry["kind"], entry["start"], entry.get("end"))
        manager.timeline.write("timeline.json")

This prints ``section 0.0 None``, ``wait 0.0 0.5`` and ``wait 0.5 0.75``. At
4 fps, the first wait takes two frames; the frozen wait shows a single frame for
0.25 seconds.

Pass ``capture_timeline=True`` to :meth:`.Manager.evaluate` or
:meth:`.Manager.render`, then read :attr:`.Manager.timeline`. The
:class:`.Timeline` object cannot be modified: ``to_dict()`` returns its data as
a new dictionary that you can edit, ``to_json()`` returns the JSON text, and
``write()`` saves it to a file.

From the command line
---------------------

From the repository root, try the included example:

.. code-block:: console

    manim --fps 4 --dry_run --timeline-output timeline.json example_scenes/timeline_scene.py TimelineExample

``--timeline-output`` saves the timeline of one scene to the given path. With
``--dry_run``, Manim evaluates the scene without rendering it, and every play
runs: caching, ``-n`` and skipped sections do not apply. Without ``--dry_run``,
Manim renders the scene as usual and records the timeline of that render. The
timeline is only written if the run succeeds.

Reading a timeline
------------------

The repository includes a small reader script:

.. code-block:: console

    python examples/timeline_reader.py timeline.json --source-root example_scenes --html timeline.html

It prints the entries and their source lines. With ``--html``, it also writes an
HTML view with links to the source lines. ``--source-root`` is the directory
that contains the scene's source file. The reader uses it to find the file and
to check whether it has changed since the timeline was recorded.

Timeline format
---------------

A timeline is a JSON object. All times are scene times in seconds. The same
scene, run with the same settings, gives the same JSON text.

.. code-block:: json

    {
      "schema": "manim.timeline",
      "version": 1,
      "mode": "evaluate",
      "scene": "TimelineExample",
      "backend": "CairoRenderer",
      "frame_rate": 4.0,
      "duration": 2.25,
      "ended_early": false,
      "source": {"path": "timeline_scene.py", "sha256": "30b191df…"},
      "entries": [
        {"kind": "section", "start": 0.0, "name": "opening",
         "type": "default.normal", "skip_animations": false,
         "source": {"path": "timeline_scene.py", "line": 11}},
        {"kind": "play", "start": 0.0, "end": 0.5, "run_time": 0.3,
         "frozen": false, "animations": [{"type": "manim.animation.transform._MethodAnimation",
                         "run_time": 0.3}],
         "skipped": false,
         "source": {"path": "timeline_scene.py", "line": 12}},
        {"kind": "caption", "start": 0.0, "end": 0.5, "content": "move",
         "source": {"path": "timeline_scene.py", "line": 12}}
      ]
    }

Top-level fields
^^^^^^^^^^^^^^^^

``schema``
    Always ``"manim.timeline"``.
``version``
    The format version. This page describes version ``1``.
``mode``
    ``"evaluate"`` if the timeline was recorded by :meth:`.Manager.evaluate`,
    or ``"render"`` if it was recorded while rendering.
``scene``
    The name of the scene class.
``backend``
    The class name of the scene's renderer, for example ``"CairoRenderer"``.
``frame_rate``
    The frame rate of the run.
``duration``
    The scene time when the run ended.
``ended_early``
    ``true`` if the scene ended early by raising ``EndSceneEarlyException``,
    for example because of the ``-n`` option.
``source``
    The scene's *source file*, the file that defines the scene class: its
    ``path`` (the file name) and the ``sha256`` hash of its contents. ``null``
    if the scene class has no source file.
``entries``
    One entry for each play, wait, section, caption, and sound call, in the order
    in which the calls were made. Captions passed to ``play()`` with
    ``subcaption=...`` follow their play entry.

Entries
^^^^^^^

Each entry has a ``kind``, a ``start`` time, and a ``source``: the ``path`` of
the file and the ``line`` of the call in the scene's code. ``path`` is relative
to the directory of the scene's source file, or ``null`` for a file outside that
directory. ``source`` is ``null`` if the call was not made from a file, for
example from code run with ``exec()``.

The other fields depend on the kind:

``play`` and ``wait``
    ``end``
        The scene time when the call ended. It can differ from ``start +
        run_time`` because animations are rounded to whole frames, and because
        a stop condition can end a wait early.
    ``run_time``
        The requested run time.
    ``frozen``
        ``true`` for a wait that shows a single unchanging frame, such as
        ``self.wait()`` without updaters. Manim draws such a wait once and repeats
        that frame.
    ``animations``
        The animations of the call, each with its ``type`` (class name) and
        ``run_time``. A wait has a single :class:`.Wait` animation.
    ``skipped``
        ``true`` if the call is left out of the rendered output, for example
        because of the ``-n`` option or a section with ``skip_animations=True``.
        Always ``false`` in an evaluation timeline.
``section``
    ``name``, ``type`` (the section type), and ``skip_animations``.
``caption``
    ``end``, the time at which the caption disappears, and ``content``. The
    ``start`` and ``end`` times include the caption's offset.
``sound``
    ``file`` (the requested path), ``duration``, ``gain``, and
    ``gain_to_background``. The ``start`` time includes the sound's time offset.
    The duration is read from the file's headers, or ``null`` if they do not
    state one. Capturing a timeline therefore needs the sound files, just as
    rendering does.

Capture errors
--------------

Timeline capture fails if scene time is not finite or goes backwards, for
example after assigning ``renderer.time``. After an error, fix the cause and run
a new scene instance.
