Writing a renderer
==================

A renderer draws frames. It is given what to draw and never reads a scene, a
mobject or a camera: for each frame the :class:`.Manager` makes a list of render
snapshots (:mod:`manim.renderer.snapshot`) and a :class:`.RenderConfig`, and
calls ``renderer.render(nodes, config)``. Everything else (the animation clock,
the play cache, sections, sound, output files) is the Manager's.

This page builds a small renderer that draws the bounding box of every mobject,
the one Manim's own tests use to check that the interface is enough
(``tests/helpers/plain_renderer.py``).

The renderer
------------

A renderer has three members (the :class:`.Renderer` protocol):

``state``
    Where the renderer is in its life, a :class:`.RendererState`: created
    ``UNOPENED``, ``OPEN`` once it drew, ``CLOSED`` after ``close()``.
``render(nodes, config)``
    Draws the snapshots in order, as ``config`` sets them up, and returns a
    :class:`.Frame`. Resources (buffers, contexts) are acquired when they are
    first needed.
``close()``
    Releases the resources; the renderer draws nothing afterwards.

.. code-block:: python

    import numpy as np

    from manim.renderer.protocol import PixelFrame, RendererState


    class PlainRenderer:
        """Draws the bounding box of each node, ``shade`` bright."""

        def __init__(self, shade):
            self.shade = shade
            self._state = RendererState.UNOPENED

        @property
        def state(self):
            return self._state

        def render(self, nodes, config):
            if not self._state.accepts_work:
                raise RuntimeError(f"The plain renderer is {self._state.value}.")
            self._state = RendererState.OPEN
            width, height = config.pixel_width, config.pixel_height
            pixels = np.empty((height, width, 4), dtype=np.uint8)
            red, green, blue = config.background_color.to_rgb()
            pixels[:] = np.round(
                np.array([red, green, blue, config.background_opacity]) * 255
            )
            camera = config.camera
            scale = np.array(
                [width / camera.frame_width, height / camera.frame_height]
            )
            for node in nodes:
                points = np.asarray(getattr(node.payload, "points", ()), float)
                if points.ndim != 2 or len(points) == 0:
                    continue
                xy = (points[:, :2] - camera.center[:2]) * scale
                low = np.floor(xy.min(axis=0) + [width / 2, height / 2]).astype(int)
                high = np.ceil(xy.max(axis=0) + [width / 2, height / 2]).astype(int)
                x0, x1 = np.clip([low[0], high[0]], 0, width)
                y0, y1 = np.clip([height - high[1], height - low[1]], 0, height)
                pixels[y0:y1, x0:x1] = [self.shade, self.shade, self.shade, 255]
            return PixelFrame(pixels)

        def close(self):
            self._state = RendererState.CLOSED

What a renderer is given
~~~~~~~~~~~~~~~~~~~~~~~~

Each :class:`.RNode` has a ``payload``: a path (:class:`.RPath`), a point cloud
(:class:`.RPoints`), an image (:class:`.RImage`), a mesh (:class:`.RMesh`) or a
camera view (:class:`.RCameraView`). Nodes come in **drawing order**:

1. *Scene order*: family order, stably sorted by ``z_index`` (unless the camera's
   ``use_z_index`` is off). Under a flat camera that is the order to paint in.
2. *Depth*: under a perspective camera (``config.camera.focal_distance`` is not
   ``None``) a renderer that paints far to near sorts nodes with a
   ``depth_reference`` by it; nodes without one keep their order and come after
   them, as the built-in renderers do. Two intersecting shapes are not resolved;
   one is drawn over the other.
3. *Depth test*: meshes with ``depth_test`` hide each other per pixel on renderers
   with a depth buffer.

Nodes **fixed in the frame** (``node.fixed_in_frame``) come after all others, in
scene order among themselves, and are never ordered or hidden by the depth of the
scene. They are drawn as if the camera had its default frame, centered at the
origin. A fixed mesh hides its own triangles but neither tests against nor
leaves the depth of anything else.

``node.key`` tells whether a node changed: equal keys mean the same drawing from
the same view, so a renderer can keep what it made for a node (a path, a texture)
and reuse it while the key stays. A key of ``None`` never equals another; such a
node is drawn anew every time.

``config.camera_views`` lists the pictures of camera views
(:class:`.ImageMobjectFromCamera`): draw each :class:`.CameraView` from its own
``nodes`` and ``config``, then show it on the frame's node with its ``uid`` and on
copies of it, which have the same ``camera_id``. A renderer that cannot draw
something it is given raises ``CapabilityError`` (:mod:`manim.renderer.protocol`).

The frame
~~~~~~~~~

``render`` returns a :class:`.Frame`: its ``size`` and ``pixels()``, RGBA with
row zero at the top. :class:`.PixelFrame` holds pixels that are already in
memory. A renderer that keeps the frame elsewhere (on the GPU) returns its own
frame and reads the pixels the first time they are asked for. ``pixels()`` can be
called until the renderer draws its next frame or closes; the array it returns
belongs to the caller and must not change afterwards.

The factory
-----------

A scene is given a *factory*, not a renderer: a renderer selection with its
options (the :class:`.RendererFactory` protocol). It is immutable, knows what its
renderers can do without creating one, and creates one renderer per scene, plus
one for images apart from the frames of a run (:meth:`.Scene.get_image`,
evaluation).

.. code-block:: python

    from dataclasses import dataclass

    from manim.renderer.protocol import RendererCapabilities


    @dataclass(frozen=True)
    class Plain:
        shade: int = 255

        @property
        def name(self):
            return "plain"

        @property
        def capabilities(self):
            return RendererCapabilities(live_preview=False)

        @property
        def cache_identity(self):
            # The options that change the pixels: a cached movie made with
            # another shade is not reused.
            return {"shade": self.shade}

        def create(self, preview=None):
            if preview is not None:
                raise ValueError("The plain renderer has no preview window.")
            return PlainRenderer(self.shade)

Select it in code:

.. code-block:: python

    class MyScene(Scene):
        def construct(self):
            self.play(Create(Square()))


    MyScene(renderer=Plain(shade=200)).render()

``config.renderer`` and ``--renderer`` only name the built-in renderers
(:mod:`manim.renderer.backends`: ``Cairo()`` and ``OpenGL()``).

A preview window
----------------

A factory whose capabilities include ``live_preview`` is given a
:class:`.PreviewRequest` when the scene renders with a live preview: where the
window sends its input (the scene) and a function that returns the camera state
for mapping window positions to scene points. Its renderer then has a
``preview`` attribute (:class:`.PreviewRenderer`), a :class:`.Preview` handle that
the Manager uses to show frames. Check for it this way:

.. code-block:: python

    from manim.renderer.protocol import PreviewRenderer

    renderer = factory.create(request)
    if isinstance(renderer, PreviewRenderer) and renderer.preview is not None:
        renderer.preview.show()

A factory that promises a live preview but whose renderer has none is reported
as an error by the Manager.
