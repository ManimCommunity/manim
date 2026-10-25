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
        manager = Manager(scene)
        manager.evaluate()
        print(scene.time)                 # 1.5
        print(scene.square.get_center())  # [1. 0. 0.]

Create and evaluate the scene under the same frame-rate configuration.
Evaluation needs an *unused* scene: one that has not played animations or
rendered yet. Each scene can be evaluated once; create another instance for
another evaluation or a separate render.

What runs
---------

Evaluation calls ``setup()``, ``construct()`` and ``tear_down()``. Play and wait
calls use the same animation loop as rendering with caching and skipping turned
off: interpolation, updaters, stop conditions, and animation finish/cleanup run
as usual. The clock follows the same frame-rate sampling and frozen-wait rounding
rules described in :doc:`deep_dive`.

Calls to ``next_section()``, ``add_subcaption()`` and ``add_sound()`` are
allowed and produce no output. In particular, sound files are only opened when the
scene is rendered, so a missing or unreadable sound file goes unnoticed during
evaluation.

Inspection and failures
-----------------------

After evaluation, read ``scene.time`` and inspect the mobjects or values saved
by your scene. The manager closes automatically unless it is inside a
``with manager:`` block, in which case it closes when the block ends.

If evaluation fails, the manager cleans up and raises the original exception.
The scene keeps the state it had reached when the error occurred.

To see an image of the resulting state, request it after evaluation:

.. code-block:: python

    with tempconfig({"frame_rate": 30}):
        scene = Motion()
        manager = Manager(scene)
        manager.evaluate()
        image = scene.get_image()  # Draws the current mobjects without advancing time.

``get_image()`` draws the scene as it is after evaluation, including changes
made by animation cleanup and ``tear_down()``. It draws with a temporary
renderer, so it also works during evaluation, for example from an updater.

Drawing frames during evaluation
--------------------------------

To see the scene at particular times, pass ``frame_times`` and an ``on_frame``
function. When evaluation reaches the frame shown at one of these times, it
draws that frame and calls ``on_frame`` with a :class:`.RenderedFrame`. Only the
requested frames are drawn, so this is much faster than rendering the scene up
to that point:

.. code-block:: python

    def save(frame):
        print(frame.frame_index, frame.time)  # 9 0.3
        frame.image.save(f"frame-{frame.frame_index}.png")

    with tempconfig({"frame_rate": 30}):
        scene = Motion()
        Manager(scene).evaluate(frame_times=[0.31], on_frame=save)

Each frame is the image a render would show at that time: at 30 fps, every time
from 0.3 up to, but not including, 0.3333 gives frame 9, which starts at 0.3
seconds. The evaluation continues to the end of the scene after the last
requested frame. Times at or after the end of the scene give no frame; use
``scene.get_image()`` after evaluation to see the final state.

Limits
------

* During evaluation, ``scene.renderer.file_writer``, a nested render of the same
  scene, and interactive preview raise an error.
* Ordinary Cairo and OpenGL mobjects whose state is computed in Python can be
  evaluated. Raw GPU-backed meshes are unsupported, and shader-only visual effects
  cannot be inspected through Python mobject state.
* **This is not a sandbox.** The scene's Python code runs as usual and can read
  or write files, start processes, or render other scenes. Use OS-level
  sandboxing to run untrusted code.
* Cairo 3D camera queries such as ``camera.project_point()`` reflect the
  camera's current angles throughout evaluation. See :doc:`cameras` for how
  the camera orientation is computed.
