Export an execution timeline
============================

A timeline records when a scene's play and wait calls ran during
:doc:`evaluation`, how many animation steps were processed, and where sections,
captions, and sounds were requested. The format and Python API are
**experimental** and may change without a deprecation period.

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
        manager = scene.manager or Manager(scene)
        manager.evaluate(capture_timeline=True)
        timeline = manager.timeline
        for event in timeline.to_dict()["events"]:
            print(event["kind"], event["start"], event["end"])
        timeline.write("timeline.json")

This prints ``wait 0.0 0.5`` and ``wait 0.5 0.75``. The first wait takes two
animation steps at 4 fps; the frozen wait occupies one frame interval.

Pass ``capture_timeline=True`` to evaluation, then read ``manager.timeline``
after it returns successfully. The returned :class:`.Timeline` is immutable:
``to_dict()`` gives you an independent dictionary, ``to_json()`` returns the
JSON text, and ``write()`` saves the captured report to disk.

From the command line
---------------------

From the repository root, try the included example:

.. code-block:: console

    manim --fps 4 --timeline-output timeline.json example_scenes/timeline_scene.py TimelineExample

``--timeline-output`` calls :meth:`.Manager.evaluate` with capture enabled and
writes the resulting JSON. Supply a source file and select exactly one scene.
Evaluation runs all play calls reached by the scene's Python code, following the
behavior described in :doc:`evaluation`.

A relative output path is resolved from the directory where the command was
invoked, before scene code can change the working directory. The command waits
for evaluation and manager cleanup, checks the primary source file, then writes
a temporary JSON file and replaces the destination. If evaluation fails or is
cancelled before publication, an existing report stays in place and continues
to describe its original execution.

Standalone reader
-----------------

The repository also includes a reader that uses only Python's standard library:

.. code-block:: console

    python examples/timeline_reader.py timeline.json --source-root example_scenes --html timeline.html

It prints events and source locations, and optionally writes an HTML timeline
with source-file links and the recorded section, caption, and sound calls.
``--source-root`` tells it where to look for those files. It compares the primary
file's bytes with the recorded hash and labels a mismatch as stale. After editing
the source, export a fresh timeline to get current line locations. Whether a
file link opens at its line number depends on the viewer.

What the report contains
------------------------

* ``policy`` is ``no-raster-full``, identifying the evaluation mode used for
  capture. ``complete`` is true for a completed capture. ``termination`` is
  ``completed`` for ordinary completion, or ``scene-end-request`` when
  ``construct()`` ends with a handled ``EndSceneEarlyException``.
* Each event represents one play or wait call. It includes an ID, play index
  (``ordinal``), kind, requested duration (``nominal_duration``), observed
  ``start`` and ``end`` times in seconds, and top-level animation types/run times.
* ``samples`` counts animation steps; ``hold_intervals`` counts the frame
  intervals occupied by a frozen wait. In the Python example these are
  ``(2, 0)`` and ``(0, 1)``. When a stop condition ends a wait early, ``end``
  records the time reached by that wait.
* ``declarations`` contains section, caption, and sound calls. Their ``at`` time
  records when the call happened; placement can differ because of an offset.
  Captions have resolved ``start`` and ``end`` times. Sounds have a ``start``;
  ``duration`` is null to indicate an unknown duration. Relative sound paths
  retain the requested string. Sound options must be JSON-serializable.
* Events and declarations share an ``order`` counter. A declaration's
  ``event_id`` refers to a play being prepared or executed, or is null outside
  one. ``event_boundary`` is the number of completed plays when it was declared.

Source locations and source checks
----------------------------------

Source hints identify a file and line on a best-effort basis. ``occurrence``
distinguishes repeated calls at the same site within a capture. Use these
locations with the source revision recorded in the report.

The source root is the primary source file's directory. Paths within that root
are relative to it. Outside-root locations provide a display basename and a null
``path``; absolute sound paths outside the root use the same treatment. Generated
or unavailable locations are marked in the source metadata.

The recorded source hash and the reader's source-status check cover the primary
file. The CLI and Python API obtain its contents at different points:

* The CLI reads the primary file before loading it and compiles those captured
  bytes. It checks the file again after loading, evaluation, and manager cleanup
  before saving.
* Python capture reads the scene class's source file from disk before evaluation,
  when available. Load your scene class from the current source before evaluating
  it in Python.

Format and content revision
---------------------------

Version 1 uses ``schema = "manim.execution-timeline"`` and integer ``version = 1``.
:meth:`.Timeline.from_json` checks the schema name, version, completion flag,
and content revision.

``revision`` is the SHA-256 hash of the document without its ``revision`` field,
serialized with Python's ``json.dumps`` using ``sort_keys=True``,
``separators=(",", ":")``, ``ensure_ascii=False``, and ``allow_nan=False``, then
encoded as UTF-8. The hash is computed before ``to_json()`` appends its trailing
newline. Readers in other languages must preserve this representation: for
example, changing ``4.0`` to ``4`` changes the hash even though they compare
numerically equal. Use the revision to identify a particular report's content.

Capture errors
--------------

Timeline capture requires sequential play/wait calls and finite, nondecreasing
animation time. The recorder checks time at samples, declarations, event
boundaries, and completion. An unfinished event or a declaration-recording
failure leaves the capture incomplete. To obtain a new report after an error,
fix the cause and evaluate a fresh scene.

See :doc:`evaluation` for the underlying execution behavior and requirements.
