Working with cameras and scene images
=====================================

A camera describes the logical view of a scene: its position, visible extent, and
projection. A renderer turns that view into pixels. You normally work with the camera
through ``self.camera`` and request images through the scene, without managing a renderer.

Moving the Cairo camera
-----------------------

The ordinary Cairo :class:`.Camera` has an animatable ``frame``. You do not need a
special scene subclass to pan or zoom::

    class CameraExample(Scene):
        def construct(self):
            square = Square().shift(2 * RIGHT)
            self.add(square)
            self.play(self.camera.frame.animate.move_to(square))
            self.play(self.camera.frame.animate.scale(0.5))

A smaller frame zooms in; a larger frame shows more of the scene. Save and restore the
frame with the usual mobject operations::

    self.camera.frame.save_state()
    self.play(self.camera.auto_zoom([square]))
    self.play(Restore(self.camera.frame))

:class:`.MovingCameraScene` remains available as a descriptive name for this behavior.
These frame examples describe the Cairo camera; OpenGL uses its own camera controls.

Logical view and image resolution
---------------------------------

Frame dimensions are in scene units; pixel dimensions specify the raster resolution.
Configure pixel dimensions before constructing the scene or renderer::

    with tempconfig({"pixel_width": 640, "pixel_height": 360}):
        scene = Scene()
        scene.add(Square())
        image = scene.get_image()

A default Cairo camera preserves ``config.frame_width`` and derives height from the
configured pixel aspect ratio. Square and portrait output therefore preserve ordinary
geometry. One explicit camera dimension determines the other using that aspect ratio;
two dimensions or a custom frame preserve the geometry you specify::

    camera = Camera(frame_width=8, frame_height=4)
    camera.frame.move_to([2, 1, 0])

With both dimensions explicit, choose the same logical and raster aspect ratio when
undistorted output is required. Drawing does not resize your semantic frame. Scene
image requests use the existing renderer dimensions, not later pixel-config edits.

Inspecting the current scene
----------------------------

:meth:`.Scene.get_image` freshly draws the current scene and returns a PIL image.
It includes manual changes since the last animation and the current camera view::

    class InspectExample(Scene):
        def construct(self):
            square = Square()
            self.add(square)
            self.get_image().save("before.png")
            self.play(square.animate.shift(RIGHT))
            self.get_image().save("after.png")

Use ``scene.show()`` to open a fresh image in PIL's external image viewer. In a notebook,
display the returned PIL image directly. Saving and opening an image are explicit;
``get_image()`` itself does not write a media artifact or open a viewer.

An image request does not execute construction, run updaters, advance scene time, or
append a movie frame. It photographs the graph as it stands, even if updater-derived
geometry has not yet been refreshed. The post-animation graph may differ from the last
encoded sample because animation finish/cleanup has already run. This is inspection,
not seeking or replaying an earlier animation position.

Request images between plays or at an idle prompt. OpenGL capture must run on the thread
that owns the rendering context; arbitrary worker-thread calls, including background
embedded-shell calls, are not dispatched automatically. Both Cairo and OpenGL draw into
independent temporary targets rather than replacing the active frame. Returned images
remain usable after those temporary targets are released. Explicit image requests also
work in dry-run mode; they are intentional raster work requested by your Python code.

Inspecting individual mobjects
------------------------------

For ordinary Cairo mobjects, use :meth:`.Mobject.get_image` or :meth:`.Mobject.show`::

    Square().show()
    Group(Square().shift(LEFT), Circle().shift(RIGHT)).get_image().save("objects.png")
    image = square.get_image(camera=self.camera)

The optional camera selects the view, not the scene contents: only the supplied mobject
and its family are drawn. Without it, a default camera is used. These standalone helpers
are Cairo-specific; use ``scene.get_image()`` for an OpenGL scene, including its meshes.

Three-dimensional and nested views
----------------------------------

Use :class:`.ThreeDScene` and its camera orientation methods for three-dimensional
scenes. Image inspection uses the current projection and fixed-object declarations,
just like ordinary drawing.

For an inset magnified view, :class:`.ZoomedScene` provides the camera and display
relationship::

    class DetailExample(ZoomedScene):
        def construct(self):
            self.add(Square())
            self.activate_zooming(animate=False)
            self.get_image().save("detail.png")

Multiple camera views
---------------------

The Cairo backend supports several camera views within one scene through
:class:`.MultiCamera`. The primary camera draws the overall scene; each secondary
camera supplies an image displayed by an :class:`.ImageMobjectFromCamera` mobject.
This is live composition of the same scene, not separate Scene executions or separate
video outputs. This API is not supported by the OpenGL backend.

There are two independent controls:

* The secondary camera's ``frame`` selects the region to look at. Move it to pan,
  or shrink it to zoom in.
* The display mobject selects where that view appears in the primary scene. Move or
  scale it like another mobject, without changing the secondary camera's view.

For example, this scene places two detail views above the original objects::

    class TwoCameraViews(Scene):
        def __init__(self, **kwargs):
            super().__init__(camera_class=MultiCamera, **kwargs)

        def construct(self):
            circle = Circle(color=YELLOW).shift(2 * LEFT + DOWN)
            square = Square(color=BLUE).shift(2 * RIGHT + DOWN)
            self.add(circle, square)

            left_camera = Camera(frame_width=4, frame_height=3)
            right_camera = Camera(frame_width=4, frame_height=3)
            left_camera.frame.move_to(circle)
            right_camera.frame.move_to(square)

            left_view = ImageMobjectFromCamera(left_camera)
            right_view = ImageMobjectFromCamera(right_camera)
            left_view.scale_to_fit_width(3).to_corner(UL)
            right_view.scale_to_fit_width(3).to_corner(UR)

            for view in (left_view, right_view):
                view.add_display_frame()
                self.camera.add_image_mobject_from_camera(view)
                self.add(view)

            # Zoom the left view without resizing its display.
            self.play(left_camera.frame.animate.scale(0.5))
            # Pan the right view from the square to the circle.
            self.play(right_camera.frame.animate.move_to(circle))
            self.wait()

Run this example with ``--renderer=cairo``. Selecting ``MultiCamera`` in the constructor
ensures it is installed before the scene's renderer is initialized.

Both registration and scene membership matter: registering a display tells MultiCamera
to produce its view; ``self.add(view)`` places the display in the scene's draw order.
``add_display_frame()`` adds an optional visible border. The secondary camera's own
``frame`` is a view control and is not automatically shown as an outline in the scene.

A display initially matches its source camera's aspect ratio. Scale it uniformly to
preserve that ratio; stretching only its width or height can distort the image.
The renderer chooses the secondary raster size from the display's size relative to
the primary view and manages resizing and pixel transfer automatically.

Secondary cameras share the scene's contents rather than having separate object lists.
Each display and its border are excluded from their own source view. In the example,
the detail cameras look below the insets so neither inset appears in the other.
Sibling views are processed in registration order; do not rely on them recursively
containing each other. For deeper nesting, a secondary camera may itself be a
MultiCamera with its own registered displays. Cyclic camera registrations are rejected
rather than rendered recursively forever.

To remove a view entirely, remove both its visible mobject and its registration::

    self.remove(left_view)
    self.camera.image_mobjects_from_cameras.remove(left_view)

The renderer retires unused secondary targets on the next draw. A scene image requested
with ``self.get_image()`` includes all currently registered and visible views; there is
no need to copy camera pixels or refresh each inset yourself.
