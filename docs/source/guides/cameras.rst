Working with cameras and scene images
=====================================

A camera represents a view of the current scene: in short, the camera describes
where the scene is being viewed from and how much of the scene is visible. The
renderer "draws" the scene from this view and turns it into an image.

Most often, interaction with the camera happens inside a scene class using
``self.camera``.

The camera frame
----------------

In 2D scenes, the camera's view is described by its *frame* (not to be confused
with a frame of a video). This frame is roughly equivalent to a picture frame
that is laid on top of the scene, with the camera "seeing" everything that lies
inside the frame. When the camera pans you can think of it as the frame sliding
across the "surface" of the scene, and when the camera zooms in or out, you can
think of it as the frame getting smaller or bigger (since less or more of the
scene, respectively, will fit into the picture frame).

Moving the Cairo camera
-----------------------

In the ordinary Cairo :class:`.Camera`, the frame is an actual mobject called
``frame``. You can modify or animate this frame like any other mobject::

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

Moving the OpenGL camera
------------------------

With the OpenGL renderer, the camera itself is a mobject. Animate
``self.camera`` directly to pan or zoom::

    class OpenGLCameraExample(Scene):
        def construct(self):
            square = Square().shift(2 * RIGHT)
            self.add(square)
            self.play(self.camera.animate.move_to(square))
            self.play(self.camera.animate.scale(0.5))

Run this example with ``--renderer=opengl``. For orientation controls, see
:class:`.OpenGLCamera` and :class:`.ThreeDScene`.

Choosing a camera class
-----------------------

To select a different Cairo camera, pass ``camera_class`` to the scene's
constructor. For example, :class:`.MultiCamera` supports picture-in-picture views::

    class CustomCameraScene(Scene):
        def __init__(self, **kwargs):
            super().__init__(camera_class=MultiCamera, **kwargs)

Use the same pattern with your own :class:`.Camera` subclass to customize its
settings or projection. The renderer creates the camera during scene
initialization, before ``setup()`` and ``construct()`` are called.

Camera view and image resolution
--------------------------------

The dimensions of the camera's frame and of the images output by the renderer
are specified separately. Camera frame dimensions are defined in scene units,
while output dimensions are defined in pixels. The output image's rectangular
pixel area is called the *viewport*. Configure its pixel dimensions before
constructing a camera, scene, or renderer::

    with tempconfig({"pixel_width": 640, "pixel_height": 360}):
        scene = Scene()
        scene.add(Square())
        image = scene.get_image()

A default Cairo camera uses ``config.frame_width`` for its width and derives its
height from the viewport's aspect ratio. This keeps circles circular and squares
square, including in square or portrait output.

Passing only ``frame_width`` or ``frame_height`` to :class:`.Camera` derives the
other dimension from that same aspect ratio. Passing both dimensions or a custom
``frame`` uses the dimensions you specify::

    camera = Camera(frame_width=8, frame_height=4)
    camera.frame.move_to([2, 1, 0])

When both the width and height of the camera frame are explicitly provided, you
should ensure that frame dimensions and pixel dimensions have the same aspect
ratio; otherwise, the camera's output will be distorted when it is rendered.

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

Use ``scene.show()`` to open a fresh image in PIL's external image viewer. In a
notebook, call ``display(scene.get_image())`` or put ``scene.get_image()`` as the
cell's final expression. Saving the image to disk is explicit, as in the example.

Request snapshots between animations or at an idle prompt to inspect the mobjects
as they currently stand. Animation playback and updaters run separately, so a
snapshot after ``self.play()`` shows the state after the animation has finished.
See :meth:`.Scene.get_image` for details on snapshot timing.

.. note::

    For OpenGL, request snapshots on the thread that created the rendering context.

Inspecting individual mobjects
------------------------------

For ordinary Cairo mobjects, use :meth:`.Mobject.get_image` or :meth:`.Mobject.show`::

    Square().show()
    Group(Square().shift(LEFT), Circle().shift(RIGHT)).get_image().save("objects.png")
    image = square.get_image(camera=self.camera)

These methods render an image from the view of a camera such that only the
chosen mobject and its submobjects are drawn; anything else in the scene is
ignored.

The ``camera`` parameter allows for a different camera to be used to generate
the image. Without it, a new default :class:`.Camera` is created. To use the
view of the current camera, pass ``camera=self.camera``.

These standalone helpers are Cairo-specific; use ``scene.get_image()`` for an
OpenGL scene, including its meshes.

Three-dimensional and nested views
----------------------------------

Use :class:`.ThreeDScene` and its camera orientation methods for three-dimensional
scenes. Image inspection uses the current projection and fixed-object declarations,
just like ordinary drawing.

:class:`.ZoomedScene` sets up a secondary camera and an inset display to show a
magnified region of the scene::

    class DetailExample(ZoomedScene):
        def construct(self):
            self.add(Square())
            self.activate_zooming(animate=False)
            self.get_image().save("detail.png")

Multiple camera views
---------------------

The Cairo backend supports several camera views within one scene through
:class:`.MultiCamera`. The primary camera draws the overall scene; each
secondary camera supplies an image displayed by an
:class:`.ImageMobjectFromCamera` mobject.
During the execution of the scene, the renderer draws each camera's view into its
display mobject. This API is not supported by the OpenGL backend.

There are two independent controls:

* The secondary camera's ``frame`` selects the region to look at. Move it to pan,
  or shrink it to zoom in.
* The display mobject selects where that view appears in the primary scene, as a
  "picture-in-picture" display. This mobject can be manipulated like any other.

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

Run this example with ``--renderer=cairo``.

Call ``self.camera.add_image_mobject_from_camera(view)`` to refresh the display's
image from its source camera on each draw, then ``self.add(view)`` to show it in
the scene.
``view.add_display_frame()`` adds a visible border around the display.
To show the region which the secondary camera is currently looking at, give its
``frame`` a visible stroke and add it to the scene::

    left_camera.frame.set_stroke(YELLOW, width=2)
    self.add(left_camera.frame)

A display initially matches the aspect ratio of its source camera. When resizing
this display, make sure it is scaled uniformly to preserve its aspect ratio;
stretching only its width or height can distort the image.
Each inset's pixel resolution follows its display size relative to the primary
camera frame.

All cameras view the same scene contents. Each display and its border are excluded
from their own camera's view. In the example, both detail cameras look below the
insets, keeping the insets out of each other's views.

For nested insets, use a :class:`.MultiCamera` as a secondary camera and register
its displays there. Keep this hierarchy acyclic: camera registrations that form
a cycle raise an error. Cameras registered at the same level are drawn in order;
place their displays outside each other's views, as above, for independent insets.

To remove a view entirely, remove both its visible mobject and its registration::

    self.remove(left_view)
    self.camera.image_mobjects_from_cameras.remove(left_view)

``self.get_image()`` captures the scene together with its current inset views.
