Evaluate a scene without rendering
==================================

Use :meth:`.Manager.evaluate` to inspect a scene's animation time and mobject
state without rendering a movie. It runs the scene's animation steps, but skips
frame drawing, encoding, and preview:

.. code-block:: python

    from manim import Manager, RIGHT, Scene, Square, tempconfig

    class Motion(Scene):
        def construct(self):
            self.square = Square()
            self.add(self.square)
            self.play(self.square.animate.shift(RIGHT), run_time=1)
            self.wait(0.5)

    with tempconfig({"frame_rate": 30}):
        scene = Motion()
        manager = scene.manager or Manager(scene)
        manager.evaluate()
        print(scene.time)                 # 1.5
        print(scene.square.get_center())  # [1. 0. 0.]

Create and evaluate the scene under the same frame-rate configuration. Start
with a fresh scene: do not play animations or open its renderer's drawing
resources or file writer before evaluating it. Each scene can be evaluated once;
create another instance for another evaluation or a separate render.

What runs
---------

Evaluation calls ``setup()``, ``construct()`` and ``tear_down()``. Play and wait
calls use the same animation loop as rendering with caching and skipping turned
off: interpolation, updaters, stop conditions, and animation finish/cleanup run
as usual. The clock follows the same frame-rate sampling and frozen-wait rounding
rules described in :doc:`deep_dive`. Time comes from running those steps, not
from adding up the requested animation durations.

The manager ignores cached movie segments, animation-range selection, and skip
flags, including those set by sections. All play calls reached by the scene's
Python code are evaluated.

Calls to ``next_section()``, ``add_subcaption()`` and ``add_sound()`` are allowed
but have no media output effect in this mode. To record these calls and the
animation steps in a report, opt into :doc:`timeline` capture. Sound files are
neither checked nor decoded. A successful evaluation therefore does not tell
you whether the same scene's audio files can be rendered.

The manager does not finalize output or capture a last-frame image. It creates
no file writer, encoder, Cairo image buffer, OpenGL context, or preview window,
and opens no file log of its own.

Inspection and failures
-----------------------

After evaluation, read ``scene.time`` and inspect the mobjects or values saved
by your scene. The manager closes automatically unless it is inside a
``with manager:`` block, in which case it closes when the block ends. Keeping
the manager open does not allow a second evaluation of the same scene.

If evaluation fails, the manager attempts cleanup and raises the original
exception. Changes already made to your scene are not rolled back.

To see an image of the resulting state, request it *after* evaluation:

.. code-block:: python

    with tempconfig({"frame_rate": 30}):
        scene = Motion()
        manager = scene.manager or Manager(scene)
        manager.evaluate()
        image = scene.get_image()  # Draws the current mobjects without advancing time.

This separate request creates the drawing resources it needs, even though
evaluation did not. It produces an image of the current state, including changes
made by animation cleanup and ``tear_down()``, not a previously rendered frame.

Limits
------

* During evaluation, calls such as ``scene.get_image()``,
  ``scene.renderer.get_frame()`` and ``scene.renderer.file_writer`` raise an error.
  Opening that renderer's GPU context, starting a nested render of the same scene,
  or entering interactive preview also raises an error.
* Ordinary Cairo and OpenGL mobjects whose state is computed in Python can be
  evaluated. Raw GPU-backed meshes are unsupported, and shader-only visual effects
  cannot be inspected through Python mobject state.
* Constructing geometry, typesetting text, loading images, and running updaters
  still take time. Evaluation does not make expensive Python code instantaneous.
* **This is not a sandbox.** Your code can still read or write files, start
  processes, or create an independent renderer or scene. Those operations are not
  blocked by evaluation. Use separate processes for independent jobs and OS-level
  sandboxing for untrusted code.
* Choose the renderer and frame rate before constructing the scene. Normal
  configuration validation and renderer-specific mobject classes still apply.

Cairo 3D camera queries
-----------------------

Calls such as ``camera.project_point()`` use the camera's current angle trackers,
even when no frame has been drawn. Reading the rotation matrix does not change
the camera's movie-cache key: the matrix is computed from the trackers, not a
separate camera setting.

``rotation_matrix`` is read-only. To rotate the camera, change its angle trackers
or use setters such as ``set_theta()``. Subclasses can customize
``generate_rotation_matrix()`` rather than assigning a matrix to the property.
If that customization depends on state beyond the angle trackers, call
``reset_rotation_matrix()`` when those extra inputs change.
