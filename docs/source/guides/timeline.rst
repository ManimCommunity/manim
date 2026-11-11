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

Save this as ``example.py`` and run it with ``python example.py``:

.. code-block:: python

    from manim import Circle, Create, Manager, Scene, tempconfig


    class Example(Scene):
        def construct(self):
            self.next_section("intro")
            self.play(Create(Circle()), run_time=0.5, subcaption="A circle")
            self.wait(0.25)


    with tempconfig({"frame_rate": 4}):
        manager = Manager(Example())
        manager.evaluate(capture_timeline=True)
        timeline = manager.timeline
        for entry in timeline.to_dict()["entries"]:
            print(entry["kind"], entry["start"], entry.get("end"))
        timeline.write("timeline.json")

This prints:

.. code-block:: text

    section 0.0 None
    play 0.0 0.5
    caption 0.0 0.5
    wait 0.5 0.75

Pass ``capture_timeline=True`` to :meth:`.Manager.evaluate` or
:meth:`.Manager.render`, then read :attr:`.Manager.timeline`. The
:class:`.Timeline` object cannot be modified: ``to_dict()`` returns its data as
a new dictionary that you can edit, ``to_json()`` returns the JSON text, and
``write()`` saves it to a file. To read a saved timeline, use
:meth:`.Timeline.from_json`:

.. code-block:: python

    from pathlib import Path

    from manim import Timeline

    timeline = Timeline.from_json(Path("timeline.json").read_text(encoding="utf-8"))
    for entry in timeline.to_dict()["entries"]:
        print(entry["kind"], entry["source"]["line"])

From the command line
---------------------

.. code-block:: console

    manim --fps 4 --dry_run --timeline-output timeline.json example.py Example

``--timeline-output`` saves the timeline of one scene's render to the given
path; it is an extra output and changes nothing else about the render. With
``--dry_run``, the render writes no media files, so only the timeline is
saved. Plays left out of the output, by ``-n`` or by skipped sections, are
in the timeline, marked as skipped. The timeline is only written if the
render succeeds. To capture a timeline without rendering, in which every play
runs, use :meth:`.Manager.evaluate` from Python, as above.

Timeline format
---------------

A timeline is a JSON object. All times are scene times in seconds. The same
scene, run with the same settings, gives the same JSON text. This is the
timeline written by the Python example above, with the hash shortened:

.. code-block:: json

    {
      "schema": "manim.timeline",
      "version": 1,
      "mode": "evaluate",
      "scene": "Example",
      "backend": "CairoRenderer",
      "frame_rate": 4.0,
      "duration": 0.75,
      "ended_early": false,
      "source": {"path": "example.py", "sha256": "…"},
      "entries": [
        {"kind": "section", "start": 0.0, "name": "intro",
         "type": "default.normal", "skip_animations": false,
         "source": {"path": "example.py", "line": 6}},
        {"kind": "play", "start": 0.0, "end": 0.5, "run_time": 0.5,
         "frozen": false,
         "animations": [{"type": "manim.animation.creation.Create",
                         "run_time": 0.5}],
         "skipped": false,
         "source": {"path": "example.py", "line": 7}},
        {"kind": "caption", "start": 0.0, "end": 0.5, "content": "A circle",
         "source": {"path": "example.py", "line": 7}},
        {"kind": "wait", "start": 0.5, "end": 0.75, "run_time": 0.25,
         "frozen": true,
         "animations": [{"type": "manim.animation.animation.Wait",
                         "run_time": 0.25}],
         "skipped": false,
         "source": {"path": "example.py", "line": 8}}
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
