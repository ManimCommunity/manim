Writing a renderer
==================

A renderer draws frames: it is given only what to draw and never reads the scene,
mobjects or the camera. For each frame the :class:`.Manager` makes a list of
render :mod:`snapshots <manim.renderer.snapshot>` and a :class:`.RenderConfig`,
and calls :meth:`renderer.render(nodes, config) <.Renderer.render>`. Everything
else (the animation clock, the play cache, sections, sound, output files) is
handled by the manager and never passed to the renderer.

This page builds a small renderer that draws the bounding box of every mobject.

The renderer
------------

A :class:`.Renderer` has three members:

``state``
    Where the renderer is in its life, a :class:`.RendererState`: ``UNOPENED``
    :math:`\Rightarrow` ``OPEN`` :math:`\Rightarrow` ``CLOSING``
    :math:`\Rightarrow` ``CLOSED`` or ``CLOSE_FAILED``.
``render(nodes, config)``
    Draws the snapshots in order and returns a :class:`.Frame`.
``close()``
    Releases all resources.

The renderer keeps its state up to date, since only it knows when its resources
are open or released. The manager reads it: it gives work only to a renderer whose
state :attr:`~.RendererState.accepts_work`, and a renderer also refuses work in
other states, as a call can come from elsewhere (:meth:`.Scene.get_image`).

.. code-block:: python

    from collections.abc import Sequence

    import numpy as np

    from manim.renderer.protocol import PixelFrame, RenderConfig, Renderer, RendererState
    from manim.renderer.snapshot import RNode


    class PlainRenderer(Renderer):
        def __init__(self, shade: int) -> None:
            self.shade = shade
            self._state = RendererState.UNOPENED

        @property
        def state(self) -> RendererState:
            return self._state

        def render(self, nodes: Sequence[RNode], config: RenderConfig, /) -> PixelFrame:
            if not self._state.accepts_work:
                raise RuntimeError("The renderer does not accept work.")

            self._state = RendererState.OPEN
            width, height = config.pixel_width, config.pixel_height
            camera = config.camera
            scale = np.array([width / camera.frame_width, height / camera.frame_height])

            pixels = np.zeros((height, width, 4), dtype=np.uint8)
            for node in nodes:
                points = node.payload.points
                if points.ndim != 2 or len(points) == 0:
                    continue
                # the bounding box of the node's points, in pixels
                xy = (points[:, :2] - np.asarray(camera.center)[:2]) * scale
                low = np.floor(xy.min(axis=0) + [width / 2, height / 2]).astype(int)
                high = np.ceil(xy.max(axis=0) + [width / 2, height / 2]).astype(int)
                # clipped to the frame; row zero is the top
                x0, x1 = np.clip([low[0], high[0]], 0, width)
                y0, y1 = np.clip([height - high[1], height - low[1]], 0, height)
                pixels[y0:y1, x0:x1] = [self.shade, self.shade, self.shade, 255]
            return PixelFrame(pixels)

        def close(self) -> None:
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

Nodes **fixed in the frame** (``node.is_fixed_in_frame``) come after all others, in
scene order among themselves, and are never ordered or hidden by the depth of the
scene. They are drawn as if the camera had its default frame, centered at the
origin. A fixed mesh hides its own triangles but neither tests against nor
leaves the depth of anything else.

``node.key`` tells whether a node changed: equal keys mean the same drawing from
the same view, so a renderer can keep what it made for a node (a path, a texture)
and reuse it while the key stays. A key of ``None`` never equals another; such a
node is drawn anew every time.

Colours and light
~~~~~~~~~~~~~~~~~

A path's ``fill``, its stroke's ``paint`` and its background stroke's ``paint`` are
:class:`.RSolidColor` (one RGBA colour), :class:`.RLinearGradient` (colour stops evenly
spaced from ``start`` to ``end``, scene points) or :class:`.RColorField` (a grid of
colours in the scene, sampled where the path is drawn; see :class:`.ColorField`
for the exact sampling). A path's and a mesh's ``material`` is an
:class:`.UnlitMaterial` or a :class:`.PbrMaterial`. ``config.lights``
(:class:`.LightState`) and ``config.ambient`` light those with a
:class:`.PbrMaterial`; without lights, nothing is shaded. Nodes fixed in the frame
are never lit. See :doc:`colors_materials_lights` for what the built-in
renderers approximate.

``config.camera_views`` lists the pictures of camera views
(:class:`.ImageMobjectFromCamera`): draw each :class:`.CameraView` from its own
``nodes`` and ``config``, then show it on the frame's node with its ``uid`` and on
copies of it, which have the same ``camera_id``. A renderer that cannot draw
something it is given raises :class:`~.CapabilityError`.

The frame
~~~~~~~~~

``render`` returns a :class:`.Frame`, which has a ``size`` and a ``pixels()``
method that returns the frame as an RGBA array, with row zero at the top.

- If the pixels are already in memory, return a :class:`.PixelFrame` that holds
  them.
- If the frame is somewhere else, for example on the GPU, return your own
  :class:`.Frame` class that reads the pixels when ``pixels()`` is first called.

The manager calls ``pixels()`` before it asks for the next frame or closes the
renderer, so the frame only needs to stay readable until then. The array that
``pixels()`` returns belongs to the caller: the renderer must not change it
afterwards.

The factory
-----------

A scene is given a *factory*, not a renderer: a renderer selection with its
options (the :class:`.RendererFactory` protocol). It is immutable, knows what its
renderers can do without creating one, and creates one renderer per scene, plus
one for images apart from the frames of a run (:meth:`.Scene.get_image`,
evaluation).

.. code-block:: python

    from collections.abc import Mapping
    from dataclasses import dataclass
    from typing import Any

    from manim.renderer.protocol import (
        PreviewRequest,
        Renderer,
        RendererCapabilities,
        RendererFactory,
    )


    @dataclass(frozen=True)
    class Plain(RendererFactory):
        shade: int = 255

        @property
        def name(self) -> str:
            return "plain"

        @property
        def capabilities(self) -> RendererCapabilities:
            return RendererCapabilities(live_preview=False)

        @property
        def cache_identity(self) -> Mapping[str, Any]:
            # The options that change the pixels: a cached movie made with
            # another shade is not reused.
            return {"shade": self.shade}

        def create(self, preview: PreviewRequest | None = None) -> Renderer:
            if preview is not None:
                raise ValueError("The plain renderer has no preview window.")
            return PlainRenderer(self.shade)

Select it in code:

.. code-block:: python

    class MyScene(Scene):
        def construct(self) -> None:
            self.play(Create(Square()))


    MyScene(renderer=Plain(shade=200)).render()

``config.renderer`` and ``--renderer`` only name the built-in renderers
(:mod:`manim.renderer.backends`).

The preview window
------------------

A renderer can show frames in a window while the scene renders (the live
preview). Its factory says so with ``live_preview`` in its capabilities. When a
scene renders with a live preview, the factory's ``create`` is given a
:class:`.PreviewRequest`, which holds what the window needs:

- where the window sends keyboard and mouse input (the scene), and
- a function that returns the current camera state, to turn window positions into
  scene points.

The renderer it creates then has a ``preview`` attribute
(:class:`.PreviewRenderer`): a :class:`.Preview` handle that the manager uses to
show frames. Check for it this way:

.. code-block:: python

    from manim.renderer.protocol import PreviewRenderer

    renderer = factory.create(request)
    if isinstance(renderer, PreviewRenderer) and renderer.preview is not None:
        renderer.preview.show()

.. note::

    Manim's tests run a fuller version of this renderer through scenes, the play
    cache, images and evaluation, to check that the interface is enough to write
    one (``tests/helpers/plain_renderer.py``). It also fills the background with
    ``config.background_color``.
