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

Inspecting the `implementation of the render method <https://github.com/ManimCommunity/manim/blob/df1a60421ea1119cbbbd143ef288d294851baaac/manim/scene/scene.py#L211>`__
reveals that there are several hooks that can be used for pre- or postprocessing
a scene. Unsurprisingly, :meth:`.Scene.render` describes the full *render cycle*
of a scene. During this life cycle, there are three custom methods whose base
implementation is empty and that can be overwritten to suit your purposes. In
the order they are called, these customizable methods are:

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

After these three methods are run, the animations have been fully rendered,
and Manim calls :meth:`.CairoRenderer.scene_finished` to gracefully
complete the rendering process. This checks whether any animations have been
played -- and if so, it tells the :class:`.SceneFileWriter` to close the pipe
to ``ffmpeg``. If not, Manim assumes that a static image should be output
which it then renders using the same strategy by calling the render loop
(see below) once.

**Back in our toy example,** the call to :meth:`.Scene.render` first
triggers :meth:`.Scene.setup` (which only consists of ``pass``), followed by
a call of :meth:`.Scene.construct`. At this point, our *animation script*
is run, starting with the initialization of ``orange_square``.


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
<https://github.com/ManimCommunity/manim/blob/5d72d9cfa2e3dd21c844b1da807576f5a7194fda/manim/mobject/mobject.py#L94>`__
of :class:`.Mobject`, you will find that not too much happens in there:

- some initial attribute values are assigned, like ``name`` (which makes the
  render logs mention the name of the mobject instead of its type),
  ``submobjects`` (initially an empty list), ``color``, and some others.
- Then, two methods related to *points* are called: ``reset_points``
  followed by ``generate_points``,
- and finally, ``init_colors`` is called.

Digging deeper, you will find that :meth:`.Mobject.reset_points` simply
sets the ``points`` attribute of the mobject to an empty NumPy vector,
while the other two methods, :meth:`.Mobject.generate_points` and
:meth:`.Mobject.init_colors` are just implemented as ``pass``.

This makes sense: :class:`.Mobject` is not supposed to be used as
an *actual* object that is displayed on screen; in fact the camera
(which we will discuss later in more detail; it is the class that is,
for the Cairo renderer, responsible for "taking a picture" of the
current scene) does not process "pure" :class:`Mobjects <.Mobject>`
in any way, they *cannot* even appear in the rendered output.

This is where different types of mobjects come into play. Roughly
speaking, the Cairo renderer setup knows three different types of
mobjects that can be rendered:

- :class:`.ImageMobject`, which represent images that you can display
  in your scene,
- :class:`.PMobject`, which are very special mobjects used to represent
  point clouds; we will not discuss them further in this tutorial,
- :class:`.VMobject`, which are *vectorized mobjects*, that is, mobjects
  that consist of points that are connected via curves. These are pretty
  much everywhere, and we will discuss them in detail in the next section.

... and what are VMobjects?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As just mentioned, :class:`VMobjects <.VMobject>` represent vectorized
mobjects. To render a :class:`.VMobject`, the camera looks at the
``points`` attribute of a :class:`.VMobject` and divides it into sets
of four points each. Each of these sets is then used to construct a
cubic Bézier curve with the first and last entry describing the
end points of the curve ("anchors"), and the second and third entry
describing the control points in between ("handles").

.. hint::
  To learn more about Bézier curves, take a look at the excellent
  online textbook `A Primer on Bézier curves <https://pomax.github.io/bezierinfo/>`__
  by `Pomax <https://twitter.com/TheRealPomax>`__ -- there is an playground representing
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
            my_vmobject = VMobject(color=GREEN)
            my_vmobject.points = [
                np.array([-2, -1, 0]),  # start of first curve
                np.array([-3, 1, 0]),
                np.array([0, 3, 0]),
                np.array([1, 3, 0]),  # end of first curve
                np.array([1, 3, 0]),  # start of second curve
                np.array([0, 1, 0]),
                np.array([4, 3, 0]),
                np.array([4, -2, 0]),  # end of second curve
            ]
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

The implementation of :meth:`.Circle.__init__` immediately calls
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

Stepping through the code in :meth:`.Scene.add`, we see that first
it is checked whether we are currently using the OpenGL renderer
(which we are not) -- adding mobjects to the scene works slightly
different (and actually easier!) for the OpenGL renderer. Then, the
code branch for the Cairo renderer is entered and the list of so-called
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
The way :meth:`.Scene.restrucutre_mobjects` works is rather aggressive:
It always operates on a given list of mobjects; in the ``add`` method
two different lists occur: the default one, ``Scene.mobjects`` (no extra
keyword arugment is passed), and ``Scene.moving_mobjects`` (which we will 
discuss later in more detail). It iterates through all of the members of
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
  >>> test_scene.mobjects
  [Group]
  >>> test_scene.restructure_mobjects(to_remove=[mob1])
  >>> test_scene.mobjects
  [Circle]

Note that the group is disbanded and the circle moves into the
root layer of mobjects in ``test_scene.mobjects``.

After the mobject list is "restructured", the mobject to be added
are simply appended to ``Scene.mobjects``. In our toy example,
the ``Scene.mobjects`` list is actually empty, so the
``restructure_mobjects`` method does not actually do anything. The
``orange_square`` is simply added to ``Scene.mobjects``, and as
the aforementioned ``Scene.moving_mobjects`` list is, at this point,
also still empty, nothing happens and :meth:`.Scene.add` returns.

We will hear more about the ``moving_mobject`` list when we discuss
the render loop. Before we do that, let us look at the next line
of code in our toy example, which includes the initialization of
an animation class, 
::
  
  ReplacementTransform(orange_square, blue_circle, run_time=3)

Hence it is time to talk about :class:`.Animation`.


Animations and the Render Loop
------------------------------

Initializing animations
^^^^^^^^^^^^^^^^^^^^^^^

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
