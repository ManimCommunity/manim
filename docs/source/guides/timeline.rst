Export an execution timeline
============================

A timeline records completed :doc:`no-raster evaluation <evaluation>`, not estimated
animation durations or a video index. The format and Python API are experimental.

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
        manager = Manager(Example())
        manager.evaluate(capture_timeline=True)
        timeline = manager.timeline
        print(timeline.to_json())
        timeline.write("timeline.json")

``manager.timeline`` is unavailable until requested capture completes successfully.
The snapshot is immutable; ``to_dict()`` returns a fresh mutable copy, not live scene
objects. ``Timeline.from_json()`` checks the experimental version and content revision.
Capture-only Python does not create an output directory unless ``write()`` is called.

From the command line
---------------------

.. code-block:: console

    manim --fps 4 --timeline-output timeline.json example_scenes/timeline_scene.py TimelineExample

``--timeline-output`` requires exactly one scene, without ``--write_all``. It does not
render a video, call a custom ``Scene.render()`` override, or create a scene file writer.
Its evaluation ignores rendered-segment caching and animation skip/range flags, as
``Manager.evaluate()`` does. Scene loading and construction still execute user code.
Do not treat this as a sandbox or a replacement for process isolation.

The explicit JSON path is anchored to the invocation directory before user code runs,
and is replaced atomically only after successful evaluation and scope cleanup. Failure or cancellation leaves a previous JSON file intact. That file
is then an older observation, not evidence that the latest evaluation succeeded.

Read without Manim
------------------

The repository includes a standard-library-only reader:

.. code-block:: console

    python examples/timeline_reader.py timeline.json --source-root example_scenes --html timeline.html

It lists events and source locations, builds an HTML timeline with source-file links,
shows declarations, and reports whether the primary source still matches its captured
bytes. File-link line fragments depend on the viewer; the explicit line number is
always shown. A stale report is not automatically remapped to edited source.
The reader needs no video or Manim import. The accompanying version-1 fixture is
``tests/control_data/timeline-v1.json``.

Version 1 facts and limits
--------------------------

The document identifies ``schema = manim.execution-timeline`` and integer ``version = 1``.
Breaking schema changes require a version bump. Its SHA-256 ``revision`` covers canonical
UTF-8 JSON excluding the revision field itself (sorted keys, compact separators,
unescaped Unicode, no NaN/Infinity). It is a content identifier, not authentication or
a promise that arbitrary Python runs deterministically.

* ``policy`` is ``no-raster-full``. ``complete`` means requested evaluation completed,
  with ``termination`` distinguishing ordinary completion from a handled early scene
  end request. Failed/cancelled captures do not expose a completed snapshot.
* Events have execution ordinal, ID, kind, nominal duration, observed Manager start/end,
  effective top-level animation summaries, evaluated sample count and logical frozen
  hold intervals. A wait produces one event, not both a wait and its internal play.
* At 4 fps, an ordinary 0.3-second request consumes two samples and advances 0.5 seconds;
  a frozen request consumes one hold interval and advances 0.25 seconds. An early stop
  uses its actual observed end, not its nominal maximum duration.
* Counts describe evaluation, **not emitted media frames**. Version 1 has no ``output``
  associations or fabricated artifacts. Optional proxy/video mapping is not implemented.
* Section, caption and sound declarations retain reached order and placement. Caption
  start/end are resolved. Sound duration may be null; relative sound requests remain
  unresolved and no audio is decoded. Sound options must be JSON-serializable.
* Source hints are best-effort, line-only and revision-local. Occurrence counters
  distinguish repeated calls at the same site. They are not AST identities or edit
  remapping keys. Generated/unavailable/outside-root sites are marked explicitly.
* The source root is the primary source file's directory. Outside-root paths are
  redacted to a display basename and are not navigation identities. Absolute checkout
  paths, object representations and wall-clock timestamps are not generated in identity.
* Coverage is **primary-file-only**, not a fingerprint of imported helpers, assets,
  environment or dependencies. CLI compiles its captured primary-file bytes directly,
  bypassing potentially stale primary-module bytecode caches; this does not verify
  imported helpers or later dynamic code changes. Python captures disk bytes before
  evaluation and does not verify them against already-loaded code. Neither claim is a
  full executed-code provenance guarantee. Changing the primary file through CLI
  resource cleanup rejects successful publication.
* Recursive timed calls and backwards observed time are unsupported. Clock checks
  cover samples, declarations, event boundaries and completion, not arbitrary transient
  private-state mutations between observations. A failed timed event or captured
  declaration cannot be hidden by catching its error and publishing a partial-looking
  successful report. Arbitrary custom
  execution schedules and external Python state are not made replayable by capture.

The underlying :doc:`evaluation` resource restrictions still apply. User geometry,
text/layout, asset loading and arbitrary Python code may perform their own work or I/O.
