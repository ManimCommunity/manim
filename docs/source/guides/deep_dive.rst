A deep dive into Manim's internals
==================================

**Author:** `Benjamin Hackl <https://benjamin-hackl.at>`__

.. admonition:: Version

    This guide describes Manim ``v0.30.0``, whose render loop was reworked
    completely: a :class:`.Manager` runs the scene, and renderers draw frames from
    snapshots of the mobjects. Both the Cairo and the OpenGL renderer are covered.

Introduction
------------

Manim can be a wonderful library, if it behaves the way you would like it to,
and/or the way you expect it to. Unfortunately, this is not always the case
(as you probably know if you have played with some manimations yourself already).
To understand where things *go wrong*, digging through the library's source code
is sometimes the only option -- but in order to do that, you need to know where
to start digging.

This article is intended as some sort of life line through the render process.
We aim to give an appropriate amount of detail describing what happens when
Manim reads your scene code and produces the corresponding animation. Throughout
this article, we will focus on the following toy example::

    from manim import *

    class ToyExample(Scene):
        def construct(self):
            orange_square = Square(color=ORANGE, fill_opacity=0.5)
            blue_circle = Circle(color=BLUE, fill_opacity=0.5)
            self.add(orange_square)
            self.play(ReplacementTransform(orange_square, blue_circle, run_time=3))
            small_dot = Dot()
            small_dot.add_updater(lambda mob: mob.next_to(blue_circle, DOWN))
            self.play(Create(small_dot))
            self.play(blue_circle.animate.shift(RIGHT))
            self.wait()
            self.play(FadeOut(blue_circle, small_dot))

Before we go into details or even look at the rendered output of this scene,
let us first describe verbally what happens in this *manimation*. In the first
three lines of the ``construct`` method, a :class:`.Square` and a :class:`.Circle`
are initialized, then the square is added to the scene. The first frame of the
rendered output should thus show an orange square.

Then the actual animations happen: the square first transforms into a circle,
then a :class:`.Dot` is created (Where do you guess the dot is located when
it is first added to the scene? Answering this already requires detailed
knowledge about the render process.). The dot has an updater attached to it, and
as the circle moves right, the dot moves with it. In the end, all mobjects are
faded out.

Actually rendering the code yields the following video:

.. manim:: ToyExample
    :hide_source:

    class ToyExample(Scene):
        def construct(self):
            orange_square = Square(color=ORANGE, fill_opacity=0.5)
            blue_circle = Circle(color=BLUE, fill_opacity=0.5)
            self.add(orange_square)
            self.play(ReplacementTransform(orange_square, blue_circle, run_time=3))
            small_dot = Dot()
            small_dot.add_updater(lambda mob: mob.next_to(blue_circle, DOWN))
            self.play(Create(small_dot))
            self.play(blue_circle.animate.shift(RIGHT))
            self.wait()
            self.play(FadeOut(blue_circle, small_dot))


For this example, the output (fortunately) coincides with our expectations.

Overview
--------

Because there is a lot of information in this article, here is a brief overview
discussing the contents of the following chapters on a very high level.

- `Preliminaries`_: In this chapter we unravel all the steps that take place
  to prepare a scene for rendering; right until the point where the user-overridden
  ``construct`` method is run. This includes a brief discussion on using Manim's CLI
  versus other means of rendering (e.g., via Jupyter notebooks, or in your Python
  script by calling the :meth:`.Scene.render` method yourself), and the role of
  the :class:`.Manager`, which runs a scene from start to finish.
- `Mobject Initialization`_: For the second chapter we dive into creating and handling
  Mobjects, the basic elements that should be displayed in our scene.
  We discuss the :class:`.Mobject` base class, the kinds of mobjects that can be
  drawn, and then the most important of them, vectorized Mobjects. In particular,
  we describe the point data structure from which the Bézier curves of a
  vectorized Mobject are drawn. We conclude the chapter with a tour into
  :meth:`.Scene.add`, the bookkeeping mechanism controlling which mobjects should
  be rendered.
- `Animations and the Render Loop`_: In this chapter we walk through the
  instantiation of :class:`.Animation` objects (the blueprints that hold
  information on how Mobjects should be modified when the render loop runs),
  followed by an investigation of the :meth:`.Scene.play` call: how the manager
  prepares the animations, steps through the frames of the animation, and seals
  the encoding job for a short video segment ("partial movie file"). Depending on
  the encoding configuration, the partial movie file can continue encoding while
  the next animation is rendered. In the end, after all of
  :meth:`.Scene.construct` has been run, the manager waits for any remaining
  encoding jobs and combines the partial movie files into one video.
- `From mobjects to pixels`_: The last chapter follows one frame from the scene's
  mobjects to its pixels: how each mobject becomes a *render snapshot*, what a
  renderer is given and what it returns, and what the Cairo and the OpenGL
  renderer do with the snapshots.

In short, a frame travels like this:

.. code-block:: text

    Scene.mobjects, Scene.camera, Scene.light
        │  FrameBuilder: one render snapshot (RNode) per mobject, in drawing
        │  order, and a RenderConfig with the camera's state and the lights
        ▼
    renderer.render(nodes, config)
        │  Cairo or OpenGL draws the snapshots
        ▼
    Frame ──► encoder ──► SceneFileWriter (partial movie files, final video)

And with that, let us get *in medias res*.

Preliminaries
-------------

Importing the library
^^^^^^^^^^^^^^^^^^^^^

You can ask Manim to render in several ways. For example, you can run
``manim -qm -p file_name.py ToyExample`` or render directly from a Python script
with a snippet like

::

    with tempconfig({"quality": "medium_quality", "preview": True}):
        scene = ToyExample()
        scene.render()

Whether you are rendering this way or from a Jupyter notebook, you are still
telling your Python interpreter to import the library. The usual pattern used to
do this is

::

    from manim import *

which (while being a debatable strategy in general) imports a lot of classes and
functions shipped with the library and makes them available in your global name space.
I explicitly avoided stating that it imports **all** classes and functions of the
library, because it does not do that: Manim makes use of the practice described
in `Section 6.4.1 of the Python tutorial <https://docs.python.org/3/tutorial/modules.html#importing-from-a-package>`__,
and all module members that should be exposed to the user upon running the ``*``-import
are explicitly declared in the ``__all__`` variable of the module.

Manim also uses this strategy internally: taking a peek at the file that is run when
the import is called, ``__init__.py`` (see
`here <https://github.com/ManimCommunity/manim/blob/main/manim/__init__.py>`__),
you will notice that most of the code in that module is concerned with importing
members from various different submodules, again using ``*``-imports.

.. hint::

    If you would ever contribute a new submodule to Manim, the main
    ``__init__.py`` is where it would have to be listed in order to make its
    members accessible to users after importing the library.

In that file, there is one particular import at the beginning of the file however,
namely::

    from ._config import *

This initializes Manim's global configuration system, which is used in various places
throughout the library. After the library runs this line, the current configuration
options are set. The code in there takes care of reading the options in your ``.cfg``
files (all users have at least the global one that is shipped with the library)
as well as correctly handling command line arguments (if you used the CLI to render).

You can read more about the config system in the
:doc:`corresponding thematic guide </guides/configuration>`, and if you are interested in learning
more about the internals of the configuration system and how it is initialized,
follow the code flow starting in `the config module's init file
<https://github.com/ManimCommunity/manim/blob/main/manim/_config/__init__.py>`__.

Now that the library is imported, we can turn our attention to the next step:
reading your scene code (which is not particularly exciting, Python just creates
a new class ``ToyExample`` based on our code; Manim is virtually not involved
in that step, with the exception that ``ToyExample`` inherits from ``Scene``).

However, with the ``ToyExample`` class created and ready to go, there is a new
excellent question to answer: how is the code in our ``construct`` method
actually executed?

Scene instantiation and rendering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The answer to this question depends on how exactly you are running the code.
To make things a bit clearer, let us first consider the case that you
have created a file ``toy_example.py`` which looks like this::

    from manim import *

    class ToyExample(Scene):
        def construct(self):
            orange_square = Square(color=ORANGE, fill_opacity=0.5)
            blue_circle = Circle(color=BLUE, fill_opacity=0.5)
            self.add(orange_square)
            self.play(ReplacementTransform(orange_square, blue_circle, run_time=3))
            small_dot = Dot()
            small_dot.add_updater(lambda mob: mob.next_to(blue_circle, DOWN))
            self.play(Create(small_dot))
            self.play(blue_circle.animate.shift(RIGHT))
            self.wait()
            self.play(FadeOut(blue_circle, small_dot))

    with tempconfig({"quality": "medium_quality", "preview": True}):
        scene = ToyExample()
        scene.render()

With such a file, the desired scene is rendered by simply running this Python
script via ``python toy_example.py``. Then, as described above, the library
is imported and Python has read and defined the ``ToyExample`` class (but,
read carefully: *no instance of this class has been created yet*).

At this point, the interpreter is about to enter the ``tempconfig`` context
manager. Even if you have not seen Manim's ``tempconfig`` before, its name
already suggests what it does: it creates a copy of the current state of the
configuration, applies the changes to the key-value pairs in the passed
dictionary, and upon leaving the context the original version of the
configuration is restored. TL;DR: it provides a fancy way of temporarily setting
configuration options.

Inside the context manager, two things happen: an actual ``ToyExample`` scene
object is instantiated, and its ``render`` method is called. The render lifecycle
is coordinated by a :class:`.Manager` attached to that scene. The CLI attaches a
manager explicitly, while direct calls to :meth:`.Scene.render` create one lazily
if necessary. To illustrate this, let us briefly look at the two most common ways
of rendering scenes:

**Command Line Interface.** When using the CLI and running the command
``manim -qm -p toy_example.py ToyExample`` in your terminal, the actual
entry point is Manim's ``__main__.py`` file (located
`here <https://github.com/ManimCommunity/manim/blob/main/manim/__main__.py>`__.
Manim uses `Click <https://click.palletsprojects.com/en/8.0.x/>`__ to implement
the command line interface, and the corresponding code is located in Manim's
``cli`` module (https://github.com/ManimCommunity/manim/tree/main/manim/cli).
The corresponding code instantiates the scene, attaches a :class:`.Manager`, and
then calls the scene's render method in the
`render command <https://github.com/ManimCommunity/manim/blob/main/manim/cli/render/commands.py>`__.

**Jupyter notebooks.** In Jupyter notebooks, the communication with the library
is handled by the ``%%manim`` magic command, which is implemented in the
``manim.utils.ipython_magic`` module. There is
:meth:`some documentation <.ManimMagic.manim>` available for the magic command.
The implementation instantiates the requested scene and calls its render method;
the scene attaches a manager lazily through that entry point.


Now that we know that either way, a :class:`.Scene` object is created, let us investigate
what Manim does when that happens. When instantiating our scene object

::

    scene = ToyExample()

the ``Scene.__init__`` method is called, given that we did not implement our own initialization
method. Inspecting the corresponding code (see
`here <https://github.com/ManimCommunity/manim/blob/main/manim/scene/scene.py>`__)
reveals that ``Scene.__init__`` first sets several attributes of the scene objects that do not
depend on any configuration options set in ``config``. It creates the scene's camera
(``camera_class``, :class:`.Camera` by default), which the scene keeps and exposes as
``scene.camera``. Then it selects a renderer: the ``renderer`` argument of the scene,
or ``config.renderer``, names a *renderer factory* from
:mod:`manim.renderer.backends` (:class:`~.backends.Cairo` or
:class:`~.backends.OpenGL`).
A factory is an immutable value that knows what its renderers can do; it is stored as
``scene.backend``, and its :meth:`~.RendererFactory.create` method makes the scene's
renderer, exposed by the read-only ``renderer`` property::

    Scene(renderer="opengl")
    Scene(renderer=OpenGL(anti_alias_width=2))

After selecting the renderer, the scene resolves the mutable configuration into
an immutable *render-session specification*. In abbreviated form, initialization
continues as follows:

::

    self.session_spec = resolve_render_session(
        config,
        self.backend.capabilities,
        renderer_name=self.backend.name,
    )
    self.output_plan = resolve_output_plan(
        resolve_media_layout(...),
        self.session_spec.output,
        scene_name=type(self).__name__,
        requested_output_name=...,
    )
    self.file_writer_settings = _SceneFileWriterSettings(
        plan=self.output_plan,
        video_encoder=self.session_spec.video_encoder,
        max_inflight_encoders=config.max_inflight_encoders,
        encoder_queue_size=config.encoder_queue_size,
        max_files_cached=config.max_files_cached,
        assets_dir=...,
    )

The session specification separates *primary artifact* intent (an ``OutputSpec``
which defines the type of file we wish Manim to create based on our scene) from
*presentation requests* such as displaying a live preview or opening the
artifact after rendering finishes.
When the primary artifact is a video, the session specification also contains
information about the video codec, pixel format, dimension, frame rate, etc.
The specification also separately records requests for *dry-run execution*.
A dry run runs the scene's animations without drawing frames or using media and
cache resources. ``format = none`` suppresses the primary artifact only: an
OpenGL live preview with automatic output still draws and displays frames without
writing a file. Both requests have an effective output format of ``none``; the
session's ``dry_run`` field tells them apart. Resolution also validates requests
against the capabilities of the selected renderer factory. For example, Cairo
rejects live preview, while OpenGL supports it. A concrete format records the
live preview as well.

The scene then resolves existing directory templates once into an immutable
*output plan* containing scene-specific output paths for the artifact, sections,
image-sequences, and cached files. If any of these paths contain directories
which do not exist, those directories will be created when a file writer needs
to use them and not at this point in time.
The resolved output format determines the file extension; ``output_file`` supplies
a name rather than overriding that format.

The scene stores the output plan, video encoding settings, cache limits, and
sound-asset directory in ``_SceneFileWriterSettings``. The :class:`.Manager` uses
these settings to create a :class:`.SceneFileWriter` when rendering starts, before
``setup()``. Accessing ``manager.file_writer`` also creates this writer if
needed; it is the same writer for the rest of the scene.

The writer uses the saved settings to encode and assemble media through ``libav``.
It creates directories as it writes files.

**Creating graphics resources.** A renderer allocates its drawing resources when it
draws its first frame, at the size that frame asks for:

- Cairo allocates an image buffer and drawing context.
- OpenGL creates its context and an offscreen framebuffer. With a live-preview
  request, the manager opens the preview window when rendering starts, before
  ``setup()``, and the window shows the frames drawn offscreen, scaled to fit.
  Accessing ``renderer.context`` creates the context sooner.

``scene.get_image()`` draws with a second renderer from the scene's factory,
which keeps the drawing state of the scene's renderer as it is and draws
offscreen.

The ``-p`` / ``--preview`` option opens the completed output file after rendering;
it is separate from OpenGL's live-preview window.

Scene initialization also prepares ``mobjects``, the list of mobjects added to the
scene, and sets ``manager`` to ``None``. The manager is lazily created: the
first time a call is made which requires a manager, such as ``render()`` or
``get_image()``, a new manager is created and attached to the scene as
``scene.manager``. This manager is then reused for any subsequent calls.

**Who is responsible for what.** Each object in this setup has one job:

- The **scene** is responsible for what is shown: its mobjects, its camera, and
  its light (:attr:`.Scene.light`, set by :class:`.ThreeDScene`), and the hooks
  ``setup``, ``construct`` and ``tear_down``.
- The **manager** runs the scene. It is responsible for the animation clock
  (``manager.time``), the play counter (``manager.num_plays``), whether plays are
  left out of the output (``manager.skip_animations``), the
  :class:`.SceneFileWriter`, the preview window handle (``manager.preview``), and a
  :class:`.FrameBuilder` that turns the scene's state into what a renderer draws.
- The **renderer** draws. It is given render snapshots and a
  :class:`.RenderConfig` and returns a :class:`.Frame`. It keeps its drawing
  resources and what it can reuse between frames.

A scene keeps the renderer selected at construction; successive plays reuse that
renderer and the manager's writer. Each new scene creates its own renderer from
its factory.

The CLI creates a fresh scene for each selected class and each interactive rerun,
on both Cairo and OpenGL. It closes the previous run's resources before creating
the next scene, which starts with a new camera and animation clock.

The rest of this article is concerned with the last line in our toy example script::

    scene.render()

This is where the actual magic happens. :meth:`.Scene.render` obtains the scene's
manager, creating and attaching a :class:`.Manager` if necessary, and delegates to
:meth:`.Manager.render`. The manager describes the full *render cycle* of a scene
through four methods: :meth:`.Manager.setup`, :meth:`.Manager.construct`,
:meth:`.Manager.tear_down`, and :meth:`.Manager.post_construct`.

The first three call the corresponding customizable scene hooks:

- :meth:`.Scene.setup`, which is intended for preparing and, well, *setting up*
  the scene for your animation (e.g., adding initial mobjects, assigning custom
  attributes to your scene class, etc.),
- :meth:`.Scene.construct`, which is the *script* for your screen play and
  contains programmatic descriptions of your animations, and
- :meth:`.Scene.tear_down`, which is intended for any operations you might
  want to run on the scene after the last frame has already been rendered
  (for example, this could run some code that generates a custom thumbnail
  for the video based on the state of the objects in the scene -- this
  hook is more relevant for situations where Manim is used within other
  Python scripts).

After these hooks have run, :meth:`.Manager.post_construct` checks whether the
resolved output can be finalized. An explicit video format for a scene without
play calls raises an error instead of silently changing the requested artifact.
Otherwise, the manager tells the :class:`.SceneFileWriter` to finish any
time-based output. For video output, the
file writer waits for partial movie files that are still being encoded and
combines them into the final movie. For last-frame PNG output, the animations
run without drawing (see below), and a single image is drawn at the end.
When automatic output selected a video for a scene without play calls, the
writer saves that image as a fallback PNG and
the manager logs a warning. The writer records the completed artifact as
``final_file_path``. After finalization, the manager uses the presentation
specification to open the artifact or reveal it in the file browser.

**Closing resources and inspecting results.** By default, rendering closes the
renderer once the output is complete. To keep it open, for instance to show more
frames in a preview window, use a ``with manager:`` block::

    scene = ToyExample()
    manager = Manager(scene)
    with manager:
        manager.render()
        last_frame = scene.get_image()

The renderer stays open after successful rendering until the end of the ``with``
block. The scene and its mobjects remain available afterward:
``scene.get_image()`` draws the scene's current state, with temporary rendering
resources if the renderer is closed.
OpenGL meshes stored directly on the GPU require their original context, so these
must be inspected inside the block.

When file logging is enabled, the manager opens the scene's log at the start of
rendering or the first direct ``play()`` call. It removes and closes that handler
when rendering ends or the manager closes, leaving other log handlers in place.
If rendering fails, the manager stops unfinished encoding work and closes the
renderer before raising the original error. Errors from cleanup are logged
separately. A failed ``play()`` call only discards its own unfinished output and
restores the scene time, so code that handles the error, such as an interactive
session, can keep playing animations. Put animation code in ``setup()`` or
``construct()`` so it runs within the normal render lifecycle; custom file or
process operations need their own cleanup.

**Back in our toy example,** the call to :meth:`.Scene.render` creates a manager,
then :meth:`.Manager.render` triggers :meth:`.Scene.setup` (which only consists of
``pass``), followed by :meth:`.Scene.construct`. At this point, our *animation
script* is run, starting with the initialization of ``orange_square``.


Mobject Initialization
----------------------

Mobjects are, in a nutshell, the Python objects that represent all the
*things* we want to display in our scene. Before we follow our debugger
into the depths of mobject initialization code, it makes sense to
discuss Manim's different types of Mobjects and their basic data
structure.

What even is a Mobject?
^^^^^^^^^^^^^^^^^^^^^^^

:class:`.Mobject` stands for *mathematical object* or *Manim object*
(depends on who you ask 😄). The Python class :class:`.Mobject` is
the base class for all objects that should be displayed on screen.
Looking at the `initialization method
<https://github.com/ManimCommunity/manim/blob/main/manim/mobject/mobject.py>`__
of :class:`.Mobject`, you will find that not too much happens in there:

- some initial attribute values are assigned, like ``uid`` (a number that
  identifies the mobject within the process), ``name`` (which makes the
  render logs mention the name of the mobject instead of its type),
  ``submobjects`` (initially an empty list), ``color``, and some others.
- Then, two methods related to *points* are called: ``reset_points``
  followed by ``generate_points``,
- and finally, ``init_colors`` is called.

Digging deeper, you will find that :meth:`.Mobject.reset_points` simply
sets the ``points`` attribute of the mobject to an empty NumPy vector,
while the other two methods, :meth:`.Mobject.generate_points` and
:meth:`.Mobject.init_colors` are just implemented as ``pass``.

This makes sense: :class:`.Mobject` is the base class of everything a
scene holds, and a plain :class:`.Mobject` serves as a container for other
mobjects, its ``submobjects``, as a :class:`.Group` does. When a frame
is drawn, each mobject in the scene is turned into a *render snapshot* that
holds the values a renderer needs (see `From mobjects to pixels`_), and there
are five kinds of mobjects with such a snapshot:

- :class:`.VMobject`, which are *vectorized mobjects*, that is, mobjects
  that consist of points that are connected via curves. These are pretty
  much everywhere, and we will discuss them in detail in the next section.
- :class:`.MeshMobject`, which are surfaces made of triangles, such as
  :class:`.Sphere`, :class:`.Surface` and the faces of :class:`.Cube`.
- :class:`.ImageMobject`, which represent images that you can display
  in your scene.
- :class:`.ImageMobjectFromCamera`, which shows what another camera sees.
- :class:`.PMobject`, which are mobjects used to represent point clouds; we
  will not discuss them further in this guide.

**Fields.** The attributes that a mobject is drawn from, such as ``points``,
``z_index`` or the colors of a vectorized mobject, are *versioned fields*
(:mod:`manim.utils.versioned_fields`). Every assignment to a field gives it a new
version number, and the arrays a field holds are read-only: changing a mobject
means assigning a new value. Looking at these version numbers is how Manim tells
what changed between two frames, which we will come back to when drawing frames.

... and what are VMobjects?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As just mentioned, :class:`VMobjects <.VMobject>` represent vectorized
mobjects. The shape of a :class:`.VMobject` is given by its ``points``
attribute, which is divided into sets of four points each. Each of these sets
describes a cubic Bézier curve, with the first and last entry describing the
end points of the curve ("anchors"), and the second and third entry
describing the control points in between ("handles").

.. hint::
  To learn more about Bézier curves, take a look at the excellent
  online textbook `A Primer on Bézier curves <https://pomax.github.io/bezierinfo/>`__
  by `Pomax <https://twitter.com/TheRealPomax>`__ -- there is a playground representing
  cubic Bézier curves `in §1 <https://pomax.github.io/bezierinfo/#introduction>`__,
  the red and yellow points are "anchors", and the green and blue
  points are "handles".

In contrast to :class:`.Mobject`, :class:`.VMobject` can be displayed
on screen (even though, technically, it is still considered a base class).
To illustrate how points are processed, consider the following short example
of a :class:`.VMobject` with 8 points (and thus made out of 8/4 = 2 cubic
Bézier curves). The resulting :class:`.VMobject` is drawn in green.
The handles are drawn as red dots with a line to their closest anchor.

.. manim:: VMobjectDemo
    :save_last_frame:

    class VMobjectDemo(Scene):
        def construct(self):
            plane = NumberPlane()
            my_vmobject = VMobject(color=GREEN).set_points(
                [
                    [-2, -1, 0],  # start of first curve
                    [-3, 1, 0],
                    [0, 3, 0],
                    [1, 3, 0],  # end of first curve
                    [1, 3, 0],  # start of second curve
                    [0, 1, 0],
                    [4, 3, 0],
                    [4, -2, 0],  # end of second curve
                ]
            )
            handles = [
                Dot(point, color=RED) for point in
                [[-3, 1, 0], [0, 3, 0], [0, 1, 0], [4, 3, 0]]
            ]
            handle_lines = [
                Line(
                    my_vmobject.points[ind],
                    my_vmobject.points[ind+1],
                    color=RED,
                    stroke_width=2
                ) for ind in range(0, len(my_vmobject.points), 2)
            ]
            self.add(plane, *handles, *handle_lines, my_vmobject)


.. warning::
  Manually setting the points of your :class:`.VMobject` is usually
  discouraged; there are specialized methods that can take care of
  that for you -- but it might be relevant when implementing your own,
  custom :class:`.VMobject`.



Squares and Circles: back to our Toy Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With a basic understanding of different types of mobjects,
and an idea of how vectorized mobjects are built we can now
come back to our toy example and the execution of the
:meth:`.Scene.construct` method. In the first two lines
of our animation script, the ``orange_square`` and the
``blue_circle`` are initialized.

When creating the orange square by running

::

  Square(color=ORANGE, fill_opacity=0.5)

the initialization method of :class:`.Square`,
``Square.__init__``, is called. `Looking at the
implementation <https://github.com/ManimCommunity/manim/blob/5d72d9cfa2e3dd21c844b1da807576f5a7194fda/manim/mobject/geometry/polygram.py#L607>`__,
we can see that the ``side_length`` attribute of the square is set,
and then

::

  super().__init__(height=side_length, width=side_length, **kwargs)

is called. This ``super`` call is the Python way of calling the
initialization function of the parent class. As :class:`.Square`
inherits from :class:`.Rectangle`, the next method called
is ``Rectangle.__init__``. There, only the first three lines
are really relevant for us::

  super().__init__(UR, UL, DL, DR, color=color, **kwargs)
  self.stretch_to_fit_width(width)
  self.stretch_to_fit_height(height)

First, the initialization function of the parent class of
:class:`.Rectangle` -- :class:`.Polygon` -- is called. The
four positional arguments passed are the four corners of
the polygon: ``UR`` is up right (and equal to ``UP + RIGHT``),
``UL`` is up left (and equal to ``UP + LEFT``), and so forth.
Before we follow our debugger deeper, let us observe what
happens with the constructed polygon: the remaining two lines
stretch the polygon to fit the specified width and height
such that a rectangle with the desired measurements is created.

The initialization function of :class:`.Polygon` is particularly
simple, it only calls the initialization function of its parent
class, :class:`.Polygram`. There, we have almost reached the end
of the chain: :class:`.Polygram` inherits from :class:`.VMobject`,
whose initialization function mainly sets the values of some
attributes (quite similar to ``Mobject.__init__``, but more specific
to the Bézier curves that make up the mobject).

After calling the initialization function of :class:`.VMobject`,
the constructor of :class:`.Polygram` also does something somewhat
odd: it sets the points (which, you might remember above, should
actually be set in a corresponding ``generate_points`` method
of :class:`.Polygram`).

.. warning::
  In several instances, the implementation of mobjects does
  not really stick to all aspects of Manim's interface. This
  is unfortunate, and increasing consistency is something
  that we actively work on. Help is welcome!

Without going too much into detail, :class:`.Polygram` sets its
``points`` attribute via :meth:`.VMobject.start_new_path`,
:meth:`.VMobject.add_points_as_corners`, which take care of
setting the quadruples of anchors and handles appropriately.
After the points are set, Python continues to process the
call stack until it reaches the method that was first called;
the initialization method of :class:`.Square`. After this,
the square is initialized and assigned to the ``orange_square``
variable.

The initialization of ``blue_circle`` is similar to the one of
``orange_square``, with the main difference being that the inheritance
chain of :class:`.Circle` is different. Let us briefly follow the trace
of the debugger:

The implementation of ``Circle.__init__`` immediately calls
the initialization method of :class:`.Arc`, as a circle in Manim
is simply an arc with an angle of :math:`\tau = 2\pi`. When
initializing the arc, some basic attributes are set (like
``Arc.radius``, ``Arc.arc_center``, ``Arc.start_angle``, and
``Arc.angle``), and then the initialization method of its
parent class, :class:`.TipableVMobject`, is called (which is
a rather abstract base class for mobjects which a arrow tip can
be attached to). Note that in contrast to :class:`.Polygram`,
this class does **not** preemptively generate the points of the circle.

After that, things are less exciting: :class:`.TipableVMobject` again
sets some attributes relevant for adding arrow tips, and afterwards
passes to the initialization method of :class:`.VMobject`. From there,
:class:`.Mobject` is initialized and :meth:`.Mobject.generate_points`
is called, which actually runs the method implemented in
:meth:`.Arc.generate_points`.

After both our ``orange_square`` and the ``blue_circle`` are initialized,
the square is actually added to the scene. The :meth:`.Scene.add` method
is actually doing a few interesting things, so it is worth to dig a bit
deeper in the next section.


Adding Mobjects to the Scene
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code in our ``construct`` method that is run next is

::

  self.add(orange_square)

From a high-level point of view, :meth:`.Scene.add` adds the
``orange_square`` to the list of mobjects that should be rendered,
which is stored in the ``mobjects`` attribute of the scene. However,
it does so in a very careful way to avoid the situation that a mobject
is being added to the scene more than once. At a first glance, this
sounds like a simple task -- the problem is that ``Scene.mobjects``
is not a "flat" list of mobjects, but a list of mobjects which
might contain mobjects themselves, and so on.

Stepping through the code in :meth:`.Scene.add`, we see that the list of so-called
foreground mobjects (which are rendered on top of all other mobjects)
is added to the list of passed mobjects. This is to ensure that the
foreground mobjects will stay above of the other mobjects, even after
adding the new ones. In our case, the list of foreground mobjects
is actually empty, and nothing changes.

Next, :meth:`.Scene.restructure_mobjects` is called with the list
of mobjects to be added as the ``to_remove`` argument, which might
sound odd at first. Practically, this ensures that mobjects are not
added twice, as mentioned above: if they were present in the scene
``Scene.mobjects`` list before (even if they were contained as a
child of some other mobject), they are first removed from the list.
The way :meth:`.Scene.restructure_mobjects` works is rather aggressive:
It operates on a given list of mobjects (by default ``Scene.mobjects``).
It iterates through all of the members of
the list, and checks whether any of the mobjects passed in ``to_remove``
are contained as children (in any nesting level). If so, **their parent
mobject is deconstructed** and their siblings are inserted directly
one level higher. Consider the following example::

  >>> from manim import Scene, Square, Circle, Group
  >>> test_scene = Scene()
  >>> mob1 = Square()
  >>> mob2 = Circle()
  >>> mob_group = Group(mob1, mob2)
  >>> test_scene.add(mob_group)
  <manim.scene.scene.Scene object at ...>
  >>> test_scene.mobjects
  [Group]
  >>> test_scene.restructure_mobjects(to_remove=[mob1])
  <manim.scene.scene.Scene object at ...>
  >>> test_scene.mobjects
  [Circle]

Note that the group is disbanded and the circle moves into the
root layer of mobjects in ``test_scene.mobjects``.

After the mobject list is "restructured", the mobject to be added
are simply appended to ``Scene.mobjects``. In our toy example,
the ``Scene.mobjects`` list is actually empty, so the
``restructure_mobjects`` method does not actually do anything. The
``orange_square`` is simply added to ``Scene.mobjects``, and
:meth:`.Scene.add` returns.

Let us look at the next line of code in our toy example, which includes
the initialization of an animation class,
::

  ReplacementTransform(orange_square, blue_circle, run_time=3)

Hence it is time to talk about :class:`.Animation`.


Animations and the Render Loop
------------------------------

Initializing animations
^^^^^^^^^^^^^^^^^^^^^^^

Before we follow the trace of the debugger, let us briefly discuss
the general structure of the (abstract) base class :class:`.Animation`.
An animation object holds all the information necessary for the renderer
to generate the corresponding frames. Animations (in the sense of
animation objects) in Manim are *always* tied to a specific mobject;
even in the case of :class:`.AnimationGroup` (which you should actually
think of as an animation on a group of mobjects rather than a group
of animations). Moreover, except for in a particular special case,
the run time of animations is also fixed and known beforehand.

The initialization of animations actually is not very exciting,
``Animation.__init__`` merely sets some attributes derived
from the passed keyword arguments and additionally ensures that
the ``Animation.starting_mobject`` and ``Animation.mobject``
attributes are populated. Once the animation is played, the
``starting_mobject`` attribute holds an unmodified copy of the
mobject the animation is attached to; during the initialization
it is set to a placeholder mobject. The ``mobject`` attribute
is set to the mobject the animation is attached to.

Animations have a few special methods which are called during the
render loop:

- :meth:`.Animation.begin`, which is called (as hinted by its name)
  at the beginning of every animation, so before the first frame
  is rendered. In it, all the required setup for the animation happens.
- :meth:`.Animation.finish` is the counterpart to the ``begin`` method
  which is called at the end of the life cycle of the animation (after
  the last frame has been rendered).
- :meth:`.Animation.interpolate` is the method that updates the mobject
  attached to the animation to the corresponding animation completion
  percentage. For example, if in the render loop,
  ``some_animation.interpolate(0.5)`` is called, the attached mobject
  will be updated to the state where 50% of the animation are completed.

We will discuss details about these and some further animation methods
once we walk through the actual render loop. For now, we continue with
our toy example and the code that is run when initializing the
:class:`.ReplacementTransform` animation.

The initialization method of :class:`.ReplacementTransform` only
consists of a call to the constructor of its parent class,
:class:`.Transform`, with the additional keyword argument
``replace_mobject_with_target_in_scene`` set to ``True``.
:class:`.Transform` then sets attributes that control how the
points of the starting mobject are deformed into the points of
the target mobject, and then passes on to the initialization
method of :class:`.Animation`. Other basic properties of the
animation (like its ``run_time``, the ``rate_func``, etc.) are
processed there -- and then the animation object is fully
initialized and ready to be played.

The ``play`` call: preparing to enter Manim's render loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are finally there, the render loop is in our reach. Let us
walk through the code that is run when :meth:`.Scene.play` is called.

The :class:`.Manager` runs every play through its *run loop*, whichever
renderer the scene uses. The scene prepares animations and updates mobjects. The
run loop steps through every frame of the animation and reports each step to a
few *listeners*: the *encoder* draws frames and sends them to the file writer, or
reuses a cached partial movie file; the *preview* shows frames in OpenGL's
window; and a progress bar shows how far the play has come. A frame is drawn
when a listener needs its image, and once per step.

:meth:`.Scene.play` passes the request to :meth:`.Manager.play`. During OpenGL
interaction, calls from another thread are queued for the rendering thread first.
The manager records the start time so that, after playback, it can place an
optional subcaption using the elapsed animation time (see
:meth:`.Scene.add_subcaption`).

The frame rate is saved when the scene is created. Create and render the scene
under the same frame-rate configuration, for example::

    with tempconfig({"frame_rate": 30}):
        scene = ToyExample()
        scene.render()

Changing ``config.frame_rate`` between construction and playback raises an error:
the animation steps must use the same rate as the saved video encoding settings.

To run the animation steps without drawing frames or producing media, see
:doc:`evaluation`.

First, the manager decides whether this play call is part of the output. For
example, ``-s`` requests only the final image, ``-n`` selects a range of play
calls, and a section with ``skip_animations=True`` leaves its plays out. A play
that is not part of the output is not drawn, but later plays start from the same
state as in a full render: if updaters or a stop condition run per frame, the play
steps through every frame; otherwise nothing can tell the frames in between apart,
and it goes to its last frame in one step. While such a play runs,
``manager.skip_animations`` is ``True``.

Sounds are placed at scene time, independently of the plays. All sounds of a
scene are mixed into one track, and when the movie is assembled, only the parts
of the track that the movie shows are kept. A play left out of the output
therefore cuts sound like it cuts the video, and reusing a cached segment does
not change the audio. Subcaptions and section videos are placed the same way.

Next, the manager calls :meth:`.Scene.compile_animation_data` to prepare the
animations. This happens once per play call and includes the following steps:

- The method processes all animations and the keyword arguments passed
  to the initial :meth:`.Scene.play` call. In particular, this means
  that it makes sure all arguments passed to the play call are actually
  animations (or ``.animate`` syntax calls, which are also assembled to
  be actual :class:`.Animation`-objects at that point). It also propagates
  any animation-related keyword arguments (like ``run_time``,
  or ``rate_func``) passed to :meth:`.Scene.play` to each individual
  animation. The processed animations are then stored in the ``animations``
  attribute of the scene for the manager to use during playback.
- It adds all mobjects to which the animations that are played are
  bound to the scene (provided the animation is not an mobject-introducing
  animation -- for these, the addition to the scene happens later).
- In case the played animation is a :class:`.Wait` animation (this is the
  case in a :meth:`.Scene.wait` call), the method checks whether a static
  image should be rendered, or whether the render loop should be processed
  as usual (see :meth:`.Scene.should_update_mobjects` for the exact conditions,
  basically it checks whether there are any time-dependent updater functions
  and so on).
- Finally, the method determines the total run time of the play call (which
  at this point is computed as the maximum of the run times of the passed
  animations). This is stored in the ``duration`` attribute of the scene.


**Caching.** If caching is enabled, the encoder computes a key with
:func:`.get_hash_from_play_call` and asks the file writer whether a matching
partial movie file exists. A match lets Manim reuse that file: the play runs
like a play left out of the output, without drawing or encoding its frames.

The key includes the play call's start time and the frame rate as well as its
visual inputs. The start time is there because animations and updaters can read
the scene time, so the same inputs at another time can give other frames. A
change that moves later plays in time, such as an inserted wait, therefore renders
them again; another render of the same scene can still reuse its segments. Waits with a stop condition are always rendered again: when such a wait
stops can depend on state that the key does not include, so a cached movie file
could have the wrong length.

.. note::

   Cache keys do not include external state that the scene's code reads, such as
   files or the current date. Use ``--disable_caching`` to render without reusing
   partial movie files.

In the event that the animation has to be rendered, the encoder gives the
:class:`.SceneFileWriter` the current animation index and asks it to start a
segment job. The writer creates a ``VideoSegmentEncoder`` from the resolved
profile and wraps it in a ``_PartialMovieEncodeJob``. The synchronous segment
encoder is responsible for its container, video stream, sequential presentation
timestamps, and target cleanup; the job is responsible for the frame queue and
worker thread. During the render loop, concrete top-left-origin ``uint8`` RGBA
arrays are added to the queue and encoded by the worker. With the writing process in place, the manager
then asks the scene to "begin" the animations.

By default, Manim finishes encoding each partial movie file before rendering the
next animation. If ``max_inflight_encoders`` is set to a value greater than 1,
encoding can instead overlap with rendering. The setting bounds the number of
partial movie encoders active at the same time, while ``encoder_queue_size``
bounds the number of pending frame buffers held by each encoder (the renderer
waits when the queue is full, so memory use does not grow with scene length).

First, it literally *begins* all of the animations by calling their
setup methods (:meth:`.Animation._setup_scene`, :meth:`.Animation.begin`).
In doing so, the mobjects that are newly introduced by an animation
(like via :class:`.Create` etc.) are added to the scene. Furthermore, the
animation suspends updater functions being called on its mobject, and
it sets its mobject to the state that corresponds to the first frame
of the animation.

Up to this very point, we did not actually render any (partial)
image or movie files from the scene yet. This is, however, about to change.
Before we enter the render loop, let us briefly revisit our toy
example and discuss how the generic :meth:`.Scene.play` call
setup looks like there.

For the :class:`.ReplacementTransform` call, the manager has no subcaption to
schedule and no request to skip rendering. It asks the scene to prepare the
animation: the argument is already an animation, no extra keyword arguments were
passed to ``play``, and ``orange_square`` is already in the scene. The scene saves
the animation's run time, 3 seconds, in ``duration``. Assuming this is the first
render, there is no cached movie to reuse. The manager passes the computed cache
key to the file writer, which opens an encoding job for the partial movie and
waits for frames.

The scene then ``begin``\ s the animation: for the
:class:`.ReplacementTransform` this means that the animation populates
all of its relevant animation attributes (i.e., compatible copies
of the starting and the target mobject so that it can safely interpolate
between the two).

Time to render some frames.


The render loop (for real this time)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The manager first checks whether this is a frozen wait. If so, the run loop
reports a single frame that is shown ``int(duration * frame_rate)`` times. The
encoder draws it once and passes it to the writer with that repeat count, and the
clock advances by that many frames without updating the mobjects again. OpenGL
also holds the frame in its preview window for the requested wait duration.

.. NOTE::

  An animation is considered a "frozen frame" animation if only a
  static :class:`.Wait` animation is played. See the description
  of :meth:`.Scene.compile_animation_data` above, or the
  implementation of :meth:`.Scene.should_update_mobjects` for
  more details.

For a non-frozen animation, such as our transformation, the manager steps through
the animation and requests a frame at each time stamp. Here is how it does that:

- The manager determines the run time of the animations by calling
  :meth:`.Scene.get_run_time`. This method basically takes the maximum
  ``run_time`` attribute of all of the animations passed to the
  :meth:`.Scene.play` call.
- The time stamps of the frames are ``i / frame_rate`` for ``i = 0, 1, ...``,
  relative to the start of the play, for every frame that starts before the total
  run time (see ``manim._run.frames.frame_count``, which allows for rounding, so
  that 0.28 seconds at 25 fps are exactly 7 frames). The progress bar listener
  shows a ``tqdm`` `progress bar <https://tqdm.github.io>`__ that counts them,
  stored in ``scene.time_progression``.
- Then the run loop goes through these time stamps: for each time stamp ``t``,
  ``Scene.update_to_time`` is called, which ...

  - ... first computes the time passed since the last update (which might be 0,
    especially for the initial call) and references it as ``dt``,
  - then (in the order in which the animations are passed to :meth:`.Scene.play`)
    calls :meth:`.Animation.update_mobjects` to trigger all updater functions that
    are attached to the respective animation except for the "main mobject" of
    the animation (that is, for example, for :class:`.Transform` the unmodified
    copies of start and target mobject -- see :meth:`.Animation.get_all_mobjects_to_update`
    for more details),
  - then the relative time progression with respect to the current animation
    is computed (``alpha = t / animation.run_time``), which is then used to
    update the state of the animation with a call to :meth:`.Animation.interpolate`.
  - After all of the passed animations have been processed, the updater functions
    of all mobjects in the scene, and finally those attached to
    the scene itself are run.

At this point, the internal (Python) state of all mobjects has been updated
to match the currently processed timestamp. If the play is part of the output,
it is now time to *take a picture*!

To take it, the manager calls ``renderer.render(*manager._frame())``: its
:class:`.FrameBuilder` turns the scene's mobjects, camera and light into render
snapshots and a :class:`.RenderConfig`, and the renderer draws them into a
:class:`.Frame`. The last chapter, `From mobjects to pixels`_, follows this step
in detail.

After drawing, the frame's ``pixels()`` are a copy of the rendered image as a
top-left-origin, row-major (C-contiguous) ``uint8`` RGBA array, which the encoder
passes to the :class:`.SceneFileWriter`. Then the manager advances the clock by one frame.
If the animation has a stop condition, the manager checks it at this point,
before starting the next iteration.

**Which time does user code see?** Within each normal animation step,
interpolation and updaters see the time at the start of that step. The stop
condition sees the time after its frame interval. For example, a
non-frozen 0.3-second animation starting at time 0, at 4 fps, takes two steps:

.. list-table::
   :header-rows: 1

   * - Animation offset
     - ``Scene.time`` during interpolation and updaters
     - ``Scene.time`` when checking the stop condition
   * - 0 seconds
     - 0 seconds
     - 0.25 seconds
   * - 0.25 seconds
     - 0.25 seconds
     - 0.5 seconds

The two frames occupy 0.5 seconds, so the next play starts at time 0.5. A frozen
wait of the same requested duration instead repeats one frame and advances by
0.25 seconds. These rounding rules apply to both backends. The clock counts
frames: the time is the number of frames played so far divided by the frame
rate.

After the last step, ``finish()`` and ``clean_up_from_scene()`` see the time at
the end of the frames processed so far. This also applies when a stop condition
ends a non-frozen wait early. We will return to those methods below.

A TL;DR for the render loop, in the context of our toy example, reads as follows:

- The scene finds that a 3 second long animation (the :class:`.ReplacementTransform`
  changing the orange square to the blue circle) should be played. Given the requested
  medium render quality, the frame rate is 30 frames per second, and so the time
  progression with steps ``[0, 1/30, 2/30, ..., 89/30]`` is created.
- In the internal render loop, each of these time stamps is processed:
  there are no updater functions, so effectively the scene updates the
  state of the transformation animation to the desired time stamp (for example,
  at time stamp ``t = 45/30``, the animation is completed to a rate of
  ``alpha = 0.5``).
- Then the encoder asks for the frame: the frame builder makes the snapshot of the
  only mobject in the scene, the main mobject attached to the transformation, and
  the renderer draws it. The encoder passes the frame's RGBA pixels to the file
  writer, and the manager advances the clock by one frame.
- At the end of the loop, 90 frames have been passed to the file writer.

Completing the render loop
^^^^^^^^^^^^^^^^^^^^^^^^^^

The last few steps of the run loop are not too exciting: for every animation,
the corresponding :meth:`.Animation.finish` and
:meth:`.Animation.clean_up_from_scene` methods are called.

.. NOTE::

  Note that as part of :meth:`.Animation.finish`, the :meth:`.Animation.interpolate`
  method is called with an argument of 1.0 -- you might have noticed already that
  the last frame of an animation can sometimes be a bit off or incomplete.
  The render loop samples times strictly before the requested run time. When that
  run time is a whole number of frame intervals, the last image shows the animation
  one interval before its end. Appending the final state as another frame would
  lengthen the video by ``1 / frame_rate`` seconds. The final interpolation updates
  the mobject for the next animation without adding that extra frame.

Finally, the manager closes the time progression, completing the terminal's
progress bar, and asks the :class:`.SceneFileWriter` to close the partial movie
stream.
This seals the corresponding encoding job so that it accepts no more frames. With
the default ``max_inflight_encoders = 1``, the file writer immediately waits for
the job and the partial movie file has been written when the call returns. With a
higher value, the job can remain in flight while the next animation is rendered;
the oldest jobs are joined as the configured limit is reached.

This pretty much concludes the walkthrough of a :meth:`.Scene.play` call,
and actually there is not too much more to say for our toy example either: at
this point, the encoding job for a partial movie file representing the
:class:`.ReplacementTransform` has been sealed (and, with the default encoding
configuration, the file has been written). The initialization of the
:class:`.Dot` happens analogous to the initialization of ``blue_circle``, which
has been discussed above. The :meth:`.Mobject.add_updater` call literally just
attaches a function to the ``updaters`` attribute of the ``small_dot``. And the
remaining :meth:`.Scene.play` and :meth:`.Scene.wait` calls follow the exact same
procedure as discussed in the render loop section above; each such call produces
a corresponding partial movie file.

Once the :meth:`.Scene.construct` method has been fully processed, the manager
calls the scene's cleanup method :meth:`.Scene.tear_down`, followed by
:meth:`.Manager.post_construct`, which asks the file writer to wrap things up by
calling :meth:`.SceneFileWriter.finish`. The file writer first waits for all
remaining encoding jobs, then combines the completed partial movie files into the
final product and records its path. If rendering aborts during a play instead, the
manager asks the file writer to abort its encoding jobs; the incomplete current
partial movie file is removed so that it cannot be mistaken for a valid cached
result on a later render. Once output is finalized, the manager carries out any
post-render presentation request from the immutable session specification.

From mobjects to pixels
-----------------------

We have glossed over one step of the render loop: *taking a picture*. This
chapter follows a single frame of our toy example, at ``t = 45/30`` in the
:class:`.ReplacementTransform`, from the scene's mobjects to the pixels that the
encoder hands to the file writer.

Render snapshots
^^^^^^^^^^^^^^^^

Renderers draw *render snapshots* (:mod:`manim.renderer.snapshot`): immutable
records of the values needed to draw one mobject. A snapshot is an :class:`.RNode`
with these parts:

- ``uid``: the mobject's :attr:`~.Mobject.uid`.
- ``payload``: the values to draw, one record type per kind of mobject:
  :class:`.RPath` for a :class:`.VMobject`, :class:`.RMesh` for a
  :class:`.MeshMobject`, :class:`.RPoints` for a :class:`.PMobject`,
  :class:`.RImage` for an :class:`.ImageMobject` and :class:`.RCameraView` for an
  :class:`.ImageMobjectFromCamera`. The ``R`` in these names stands for
  *render snapshot*.
- ``placement``: how the node is placed in the view: ``None`` for a mobject in the
  scene, :class:`.FixedInFrame` for one fixed in the frame
  (:meth:`.Mobject.fix_in_frame`), :class:`.FacingCamera` for one with a fixed
  orientation (:meth:`.Mobject.fix_orientation`).
- ``key`` and ``versions``: what a renderer uses to reuse work. The key changes
  whenever any field of the mobject changes; the versions are kept per part of
  the payload (a *slot*, such as ``"geometry"`` or ``"fill"`` of a path), for
  renderers that keep parts of a node separately, such as uploaded geometry.

The payload of our transformed square is an :class:`.RPath`:

- ``points``: the cubic Bézier points of the path, the mobject's own array (field
  values are read-only, so the snapshot shares them);
- ``fill`` and ``stroke``: what colors the area and the outline. A mobject's
  ``fill`` and ``stroke`` are *paints* (:mod:`manim.appearance.paint`): a
  :class:`.SolidColor`, a :class:`.LinearGradient`, or a :class:`.ColorField` that
  places colors in the scene. Each paint resolves itself into a snapshot paint
  (:meth:`.Paint.resolve`): :class:`.RSolidColor`, :class:`.RLinearGradient`,
  whose stops are spaced between two points in the scene, or :class:`.RColorField`.
  The stroke snapshot, :class:`.RStroke`, adds the width, joint and cap;
- ``background_stroke``: an outline drawn below the fill, when its width is above
  zero;
- ``material``: how the path responds to light (:mod:`manim.appearance.material`):
  :class:`.UnlitMaterial` shows its colors, :class:`.PbrMaterial` shades them by
  the scene's lights;
- ``depth_reference``: the point by which a perspective camera orders the path,
  for mobjects with ``sort_by_depth``.

Snapshots are made by :func:`~manim.renderer.extract.extract`, which walks the
families of the scene's mobjects in drawing order: family order, stably sorted by
``z_index``. An :class:`~manim.renderer.extract.Extractor` keeps the snapshot of
each mobject together with the versions of its fields, and returns it again while
they stay the same: a mobject that did not change since the previous frame costs a
look at its version numbers. With ``MANIM_CHECK_SNAPSHOTS=1``, as in Manim's
tests, every reused snapshot is compared with a fresh one.

What a renderer is given
^^^^^^^^^^^^^^^^^^^^^^^^

The manager's :class:`.FrameBuilder` makes everything a renderer needs for one
frame:

- the snapshots of the scene's mobjects, from its extractor, with mobjects fixed in
  the frame moved to the end;
- a :class:`.RenderConfig`: the pixel size, the background, the scene's lights and
  ambient light, and the camera's state as a :class:`.CameraState` (the frame's
  center and size, its rotation, and for a :class:`.ThreeDCamera` the focal
  distance and zoom). With a :class:`.MultiCamera`, the configuration also holds
  one :class:`.CameraView` per camera view, with the snapshots and configuration
  of what that camera sees.

The renderer protocol (:class:`.Renderer`) is small: ``render(nodes, config)``
draws a frame and returns a :class:`.Frame`, whose ``pixels()`` method gives a
top-left-origin, row-major ``uint8`` RGBA array; ``close()`` releases the drawing
resources. For anything it cannot draw, a renderer raises ``CapabilityError``
(from :mod:`manim.renderer.protocol`). The snapshots and the configuration are all
the input: the same nodes and configuration give the same frame. The guide
:doc:`writing_a_renderer` describes how to write one.

Some work is the same for every renderer and lives in shared helpers:

- ``manim.renderer._view`` places points in the view (projection under a
  :class:`.ThreeDCamera`, mobjects fixed in the frame or facing the camera) and
  orders snapshots far to near under a perspective camera;
- ``manim.renderer._paint`` turns snapshot paints into color stops and computes
  the color of a lit path;
- ``manim.renderer._lighting`` holds the shading function, the same for every
  renderer: glTF's metallic-roughness parameters under point lights;
- ``manim.renderer._reuse`` decides how much of the previous frame can be reused.
  The snapshots at the start of the list that have the same keys as in the previous
  frame, seen from the same view with the same lights, give the same pixels. A
  renderer saves the drawing of such a prefix and copies it in the following frames
  while the prefix stays the same; only the rest is drawn.

In our toy example, the transforming square is the only mobject and changes in
every frame, so every frame draws it.

The Cairo renderer
^^^^^^^^^^^^^^^^^^

:class:`.CairoRenderer` draws on the CPU into an image buffer with
`pycairo <https://pycairo.readthedocs.io>`__. For each frame it:

1. draws each camera view of the configuration into an image buffer of its own;
2. clears the frame's buffer to the background color or image;
3. replaces every mesh by one snapshot per face: each face becomes a filled
   polygon, colored once with the shading function at its middle;
4. orders the snapshots far to near under a perspective camera;
5. copies the saved drawing of the unchanged prefix, if there is one, and draws
   the rest in order:

   - paths become Cairo paths: the background stroke, then the fill, then the
     stroke. A solid paint becomes a color, a gradient a ``cairo.LinearGradient``
     between its two scene points, and a color field an image pattern placed in
     the scene (under a flat camera);
   - point clouds are drawn as squares of pixels;
   - images and camera views are transformed onto their corner points and
     composited.

Under a flat camera, the Cairo paths of unchanged mobjects are kept from one
frame to the next and reused.

The OpenGL renderer
^^^^^^^^^^^^^^^^^^^

:class:`.OpenGLRenderer` draws on the GPU with `ModernGL
<https://moderngl.readthedocs.io>`__, into an offscreen framebuffer of the frame's
size; a preview window shows that framebuffer scaled to the window. For each
frame it:

1. draws each camera view into a texture;
2. clears the framebuffer to the background color;
3. orders the snapshots far to near under a perspective camera;
4. copies the saved drawing of the unchanged prefix, if there is one, and draws
   the rest in order:

   - paths: OpenGL's shaders draw quadratic Bézier curves, so each cubic path is
     split on the CPU into quadratic pieces that stay within a fraction of a pixel
     of it. The pieces are kept per mobject by the version of its points.
     Consecutive paths that cannot change each other's pixels are drawn together,
     in three steps: one draw call for their background strokes, one *winding
     fill*, and one draw call for their strokes. The winding fill adds up the signed
     coverage of each curve's triangles in a float buffer, which leaves the winding
     number of the path at every pixel, and then colors the pixels with a nonzero
     winding number. The stroke shader draws each quadratic piece as a ribbon of the
     stroke's width. Paints become colors per point, and a color field a texture
     sampled where the path is drawn;
   - meshes: their vertices, normals and colors go to the mesh shader, which
     shades them with the same lighting as the shared shading function, written in
     GLSL. Meshes with ``depth_test`` hide what lies behind them;
   - point clouds are drawn as point sprites, and images and camera views as
     textured quads.

Both renderers draw a frame from the same input and the same shared helpers, so a
scene looks the same on either; where one renderer cannot draw something, it
raises ``CapabilityError``.

Back in our toy example, at ``t = 45/30``: the frame builder's extractor makes a
fresh snapshot of the transforming square, whose points changed, and the
renderer draws its fill and stroke over the cleared background. The encoder reads
the frame's pixels and queues them for the partial movie file.

And there you go! This is a more or less detailed description of how Manim works
under the hood. While we did not discuss every single line of code in detail
in this walkthrough, it should still give you a fairly good idea of how the
library is structured, from the scene and its manager through the snapshots of
its mobjects to the pixels that the renderers draw.
