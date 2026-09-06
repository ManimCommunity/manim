Evaluate a scene without rendering
==================================

Use :meth:`.Manager.evaluate` to run a scene's Python animation logic without
producing pixels or media:

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
        print(scene.time)
        print(scene.square.get_center())

Construct and evaluate under the same frame-rate configuration. Start with an unused
scene whose primary drawing resources and file writer have not been opened.

What runs
---------

Evaluation invokes ``setup()``, ``construct()`` and ``tear_down()``. It uses the same
Manager play path, interpolation, updaters, stop checks, animation finish/cleanup and
clock as ordinary uncached, unskipped rendering. Frozen waits retain their ordinary
whole-frame logical span. Evaluation is not summing animation durations or jumping
directly to every animation endpoint.

Rendered-segment cache, animation-range selection and skip flags (including section skip flags)
are ignored. ``next_section()``, ``add_subcaption()`` and ``add_sound()`` requests are
accepted without constructing a writer or decoding audio. This does not validate sound
files or produce a public timeline report; structured export is a separate feature.

Output finalization is not run: there is no final-frame capture, partial movie,
encoder, assembly, media preview, or newly opened execution file log. No primary
Cairo raster target or OpenGL context/window is acquired by evaluation.

Inspection and failures
-----------------------

Time and Python scene state remain readable after evaluation. As with rendering,
success closes the Manager unless an explicit Manager context scope retains it;
failures retire the scope and preserve the original exception.

An image is a separate, explicit operation *after* evaluation:

.. code-block:: python

    with tempconfig({"frame_rate": 30}):
        scene = Motion()
        Manager(scene).evaluate()
        image = scene.get_image()  # This request DOES rasterize the final state.

You can instead attach ``Manager(scene)`` explicitly, as in the first example, or
reuse ``scene.manager`` when a Manager is already attached.

Limits
------

* Requests for images, bound-renderer pixels/GPU resources, output writers or
  interactive preview during evaluation raise an error rather than secretly opening
  rendering resources. Raw GPU-backed meshes are unsupported.
* Ordinary CPU-backed Cairo and OpenGL mobjects can be evaluated. This is not a promise
  that every shader-specific effect has a CPU equivalent.
* Geometry construction, text/layout tools, image loading and arbitrary user code still
  run. This is **not a sandbox**, a zero-user-I/O guarantee, or instant execution.
  External code that opens files, creates its own GPU context or launches another
  scene is not isolated by this API. Use separate processes for independent jobs
  and appropriate OS sandboxing for untrusted code.
* Existing Scene-construction configuration validation still applies. Evaluation does
  not introduce late configuration preparation or remove renderer-dependent mobject
  classes.

Three-dimensional Cairo projection queries now derive their rotation from current
camera trackers, whether or not drawing has occurred. Derived rotation-cache contents
do not participate in the camera's visual cache identity. ``rotation_matrix`` is a
derived read-only property; change camera trackers (or customize
``generate_rotation_matrix()``) rather than assigning that cache directly.
