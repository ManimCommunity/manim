A deep dive into Manim's internals
==================================

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
Now let us explore the code flow behind Manim's rendering logic.

Overview
--------

Because there is a lot of information in this article, here is a brief overview
discussing the contents of the following sections on a very high level.

- preliminaries (import, up to scene.render): TODO
- initializing mobjects (already within construct): TODO
- the actual render loop: TODO


Preliminaries
-------------

Importing the library
^^^^^^^^^^^^^^^^^^^^^

Now let us get *in medias res*. Independent of how exactly you are telling your system
to render the scene, i.e., whether you run ``manim -qm -p file_name.py ToyExample``, or
whether you are rendering the scene directly from the Python script via a snippet
like

::

    with tempconfig({"quality": "medium_quality", "preview": True}):
        scene = ToyExample()
        scene.render()

or whether you are rendering the code in a Jupyter notebook, you are still telling your
python interpreter to import the library. The usual pattern used to do this is

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
:doc:`corresponding tutorial <configuration>`, and if you are interested in learning
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
manager. Even if you have not seen Manim's ``tempconfig`` before, it's name
already suggests what it does: it creates a copy of the current state of the
configuration, applies the changes to the key-value pairs in the passed
dictionary, and upon leaving the context the original version of the
configuration is restored. TL;DR: it provides a fancy way of temporarily setting
configuration options.

Inside the context manager, two things happen: an actual ``ToyExample``-scene
object is instantiated, and the ``render`` method is called. Every way of using
Manim ultimately does something along of these lines, the library always instantiates
the scene object and then calls its ``render`` method. To illustrate that this
really is the case, let us briefly look at the two most common ways of rendering
scenes:

**Command Line Interface.** When using the CLI and running the command
``manim -qm -p toy_example.py ToyExample`` in your terminal, the actual
entry point is Manim's ``__main__.py`` file (located
`here <https://github.com/ManimCommunity/manim/blob/main/manim/__main__.py>`__.
Manim uses `Click <https://click.palletsprojects.com/en/8.0.x/>`__ to implement
the command line interface, and the corresponding code is located in Manim's
``cli`` module (https://github.com/ManimCommunity/manim/tree/main/manim/cli).
The corresponding code creating the scene class and calling its render method
is located `here <https://github.com/ManimCommunity/manim/blob/ac1ee9a683ce8b92233407351c681f7d71a4f2db/manim/cli/render/commands.py#L139-L141>`__.

**Jupyter notebooks.** In Jupyter notebooks, the communication with the library
is handled by the ``%%manim`` magic command, which is implemented in the
``manim.utils.ipython_magic`` module. There is
:meth:`some documentation <.ManimMagic.manim>` available for the magic command,
and the code creating the scene class and calling its render method is located
`here <https://github.com/ManimCommunity/manim/blob/ac1ee9a683ce8b92233407351c681f7d71a4f2db/manim/utils/ipython_magic.py#L137-L138>`__.


Now that we know that either way, a :class:`.Scene` object is created, let us investigate
what Manim does when that happens. When instantiating our scene object

::

    scene = ToyExample()

the ``Scene.__init__`` method is called, given that we did not implement our own initialization
method. Inspecting the corresponding code (see
`here <https://github.com/ManimCommunity/manim/blob/main/manim/scene/scene.py>`__)
reveals that ``Scene.__init__`` first sets several attributes of the scene objects that do not
depend on any configuration options set in ``config``. Then the scene inspects the value of
``config.renderer``, and based on its value, either instantiates a ``CairoRenderer`` or an
``OpenGLRenderer`` object and assigns it to its ``renderer`` attribute.

The scene then asks its renderer to initialize the scene by calling

::

    self.renderer.init_scene(self)

Inspecting both the default Cairo renderer and the OpenGL renderer shows that the ``init_scene``
method effectively makes the renderer instantiate a :class:`.SceneFileWriter` object, which
basically is Manim's interface to ``ffmpeg`` and actually writes the movie file. The Cairo
renderer (see the implementation `here <https://github.com/ManimCommunity/manim/blob/main/manim/renderer/cairo_renderer.py>`__) does not require any further initialization. The OpenGL renderer
does some additional setup to enable the realtime rendering preview window, which we do not go
into detail further here.

.. warning::

    Currently, there is a lot of interplay between a scene and its renderer. This is a flaw
    in Manim's current architecture, and we are working on reducing this interdependency to
    achieve a less convoluted code flow.

After the renderer has been instantiated and initialized its file writer, the scene populates
further initial attributes (notable mention: the ``mobjects`` attribute which keeps track
of the mobjects that have been added to the scene). It is then done with its instantiation
and ready to be rendered.

The rest of this article is concerned with the last line in our toy example script::

    scene.render()

This is where the actual magic happens.


Mobject Initialization
----------------------

- custom scene.setup method is called (NOP in our case)
- custom scene.construct method is called (!)
- mobject initialization in depth: ``orange_square``

  - Step through initialization hierarchy (Square, Rectangle, Polygon, Polygram, VMobject, Mobject)
  - Talk a bit about color init?
  - Travelling back up, until Polygram is reached again; points are actually initialized

    - digression: internal representation of points (explain that
      self.points holds anchors + handles of bezier curves)
    - first point added to points list (start new path)
    - remaining vertices: new line segments in between

- mobject initialization of circle: more or less same as ``orange_square``,
  different inheritance structure obviously.

- adding a mobject to the scene!

  - TL;DR: families (= recursive list of all mobjects + submobjects
    of mobjects that are in the scene already are expanded, mobject that
    should be newly added are removed from it, and then added "safely"
    (to avoid duplicates) to the list separately.



Producing Frames: The Render Loop
---------------------------------

- constructing the ``ReplacementTransform``:

  - ``ReplacementTransform`` only sets the flag for replacing the
    starting mobject with the target mobject in the scene
  - ``Transform`` (base class) has information about how points
    from starting mobject move to points of target mobject
  - ``Animation`` (base class) has all other info
 
- entering the play call!

  - minor preprocessing regarding animation time for subcaption feature (not important at all)
  - enter renderer.play!

    - ask scene to compile animation data (static mobjects / moving mobjects + animation run time)
      static mobjects are mobjects that can be rendered once and then remain in the background
      throughout the entire animation. in terms of layers: all mobjects
      that are below the first "moving" / animated mobject.
    - manim's caching mechanism (no comment, just say that it is there and
      allows reusing already rendered animations that did not "change"
    - "background image" consisting of static mobjects is rendered.
    - ffmpeg pipeline opens, awaiting frames from file writer.
    - scene.begin_animations: introducers actually add mobjects to scene,
      starting mobjects are assigned properly, animations are set to
      initial interpolation state.
    - check whether current animation is a frozen frame, not in our case
    - scene.play_internal:

      - construct time_progression (i.e., the progress bar; t-values for
        which frames are rendered)
      - step through time progression. scene.update_to_time(t)

        - updates animation mobjects
        - runs interpolate for correct alpha value
        - runs mobject updaters
        - runs scene updaters
        - self.renderer.render(self, t, self.moving_mobjects), actually
          rendering the frame

      - finish animations
      - ffmpeg movie pipeline closes; partial movie file is written

- after all animations: combination of all partial movie files to one
  rendered video.
