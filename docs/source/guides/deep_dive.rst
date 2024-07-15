A deep dive into Manim's internals
==================================

**Authors:** `Benjamin Hackl <https://benjamin-hackl.at>`__ and `Aarush Deshpande <https://github.com/JasonGrace2282>`__

.. admonition:: Disclaimer

    This guide reflects the state of the library as of version ``v0.20.0``
    and primarily treats the Cairo renderer. The situation in the latest
    version of Manim might be different; in case of substantial deviations
    we will add a note below.

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
  ``construct`` method is ran. This includes a brief discussion on using Manim's CLI
  versus other means of rendering (e.g., via Jupyter notebooks, or in your Python
  script by calling the :meth:`.Manager.render` method yourself).
- `Mobject Initialization`_: For the second chapter we dive into creating and handling
  Mobjects, the basic elements that should be displayed in our scene.
  We discuss the :class:`.Mobject` base class, how there are essentially
  three different types of Mobjects, and then discuss the most important of them,
  vectorized Mobjects. In particular, we describe the internal point data structure
  that governs how the mechanism responsible for drawing the vectorized Mobject
  to the screen sets the corresponding Bézier curves. We conclude the chapter
  with a tour into :meth:`.Scene.add`, the bookkeeping mechanism controlling which
  mobjects should be rendered.
- `Animations and the Render Loop`_: And finally, in the last chapter we walk
  through the instantiation of :class:`.Animation` objects (the blueprints that
  hold information on how Mobjects should be modified when the render loop runs),
  followed by a investigation of the infamous :meth:`.Scene.play` call. We will
  see that there are three relevant parts in a :meth:`.Scene.play` call;
  a part in which the passed animations and keyword arguments are processed
  and prepared, followed by the actual "render loop" in which the library
  steps through a time line and renders frame by frame. The final part
  does some post-processing to save a short video segment ("partial movie file")
  and cleanup for the next call to :meth:`.Scene.play`. In the end, after all of
  :meth:`.Scene.construct` has been run, the library combines the partial movie
  files to one video.

.. hint::

   As we move forward, try to keep in mind the responsibilities of every
   class we introduce. We'll talk more about them in detail, but here's a brief
   overview

   * :class:`.Scene` is responsible for managing the classes :class:`Mobject`, :class:`.Animation`,
     and :class:`.Camera`.

   * :class:`.Manager` is responsible for coordinating the :class:`.Scene`, :class:`.Renderer`,
     and :class:`.FileWriter`.

   * :class:`.FileWriter` is responsible for writing frames and partial movie files, as well
     as combining them all into a final movie file.

   * :class:`.Renderer` is an abstract class which has to be subclassed.
     It's job is to take information related to the :class:`.Camera`, and the mobjects
     on the :class:`.Scene` at a certain frame, and to return the pixels in a frame.

And with that, let us get *in medias res*.

Preliminaries
-------------

Importing the library
^^^^^^^^^^^^^^^^^^^^^

Independent of how exactly you are telling your system
to render the scene, i.e., whether you run ``manim -qm -p file_name.py ToyExample``, or
whether you are rendering the scene directly from the Python script via a snippet
like

::

    with tempconfig({"quality": "medium_quality", "preview": True}):
        manager = Manager(ToyExample)
        manager.render()

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
        manager = Manager(ToyExample)
        manager.render()

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

Inside the context manager, two things happen: a :class:`.Manager` is created for
the ``ToyExample``-scene, and the ``render`` method is called. Every way of using
Manim ultimately does something along of these lines, the library always instantiates
the manager of the scene object and then calls its ``render`` method. To illustrate that this
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


Now that we know that either way, a :class:`.Manager` for a :class:`.Scene` object is created, let us investigate
what Manim does when that happens. When instantiating our manager

::

    manager = Manager(ToyExample)

The :meth:`.Manager.__init__` method is called. Looking at the source code (`here <https://github.com/ManimCommunity/manim/blob/experimental/manim/manager.py>`__),
we see that the :meth:`.Scene.__init__` method is called,
given that we did not implement our own initialization
method. Inspecting the corresponding code (see `here <https://github.com/ManimCommunity/manim/blob/main/manim/scene/scene.py>`__)
reveals that :class:`Scene.__init__` first sets several attributes of the scene objects that do not
depend on any configuration options set in ``config``. It then initializes it's :class:`.Camera`.
The purpose of a :class:`.Camera` is to keep track of what you can see in the scene. Think of it
as a pair of eyes, that limit how far you can look sideways and vertically.

The :class:`.Scene` also sets up :attr:`.Scene.mobjects`. This attribute keeps track of all the :class:`.Mobject`
that have been added to the scene.

The :class:`.Manager` then continues on to create a :class:`.Window`, which is the popopen interactive window,
and creates the renderer::

    self.renderer = self.create_renderer()
    self.renderer.use_window()

If you hover over :attr:`.Manager.renderer`, you might see that the type is a :class:`.RendererProtocol`.
A :class:`~typing.Protocol` is a contract for a class. It says that whatever the class is, it will implement
the methods defined inside the protocol. In this case, it means that the renderer will have all the methods
defined in :class:`.RendererProtocol`.

.. note::

   The point of using :class:`~typing.Protocol` is so that in the future, plugins
   can swap out the renderer with their own version - either for speed, or for a different
   behavior.


For the rest of this article to take a concrete example, we'll use :class:`.OpenGLRenderer`.

Finally, the :class:`.Manager` creates a :class:`.FileWriter`. This is the object that actually
writes the partial movie files.

The rest of this article is concerned with the last line in our toy example script::

    manager.render()

This is where the actual magic happens.

.. note::

   TODO TO REVIEWERS - Replace this link with the proper permanent link

Inspecting the `implementation of the render method <https://github.com/ManimCommunity/manim/blob/df1a60421ea1119cbbbd143ef288d294851baaac/manim/scene/scene.py#L211>`__
we see that there are two passes of rendering.

.. note::

   As of the experimental branch at June 30th, 2024, two pass rendering
   does not exist. This will proceed to explain the single pass rendering system.

Looking around, we find that there are several hooks that can be used for pre- or postprocessing
a scene (check out :meth:`.Manager._setup`, and :meth:`.Manager._tear_down`).

.. note::

   You might notice :attr:`.Manager.virtual_animation_start_time` and :attr:`.Manager.real_animation_start_time`
   when looking through :meth:`.Manager._setup`. These will be explained later.

Unsurprisingly, :meth:`.Manager.render` describes the full *render cycle*
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
and Manim calls :meth:`.Manager.tear_down` to gracefully
complete the rendering process. This checks whether any animations have been
played -- and if so, it tells the :class:`.SceneFileWriter` to close the output
file. If not, Manim assumes that a static image should be output
which it then renders using the same strategy by calling the render loop
(see below) once.

**Back in our toy example,** the call to :meth:`.Manager.render` first
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
sets the ``points`` attribute of the mobject to an empty NumPy array,
while the other two methods, :meth:`.Mobject.generate_points` and
:meth:`.Mobject.init_colors` are just implemented as ``pass``.

This makes sense: :class:`.Mobject` is not supposed to be used as
an *actual* object that is displayed on screen.

This is where different types of mobjects come into play. Roughly
speaking, the Cairo renderer setup knows three different types of
mobjects that can be rendered:

- :class:`.ImageMobject`, which represent images that you can display
  in your scene,
- :class:`.PMobject`, which are very special mobjects used to represent
  point clouds; we will not discuss them further in this guide,
- :class:`.VMobject`, which are *vectorized mobjects*, that is, mobjects
  that consist of points that are connected via curves. These are pretty
  much everywhere, and we will discuss them in detail in the next section.

... and what are VMobjects?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As just mentioned, :class:`VMobjects <.VMobject>` represent vectorized
mobjects. To render a :class:`.VMobject`, the camera looks at the
:attr:`~.VMobject.points` attribute of a :class:`.VMobject` and divides it into sets
of three points each. Each of these sets is then used to construct a
quadratic Bézier curve with the first and last entry describing the
end points of the curve ("anchors"), and the second entry
describing the control points in between ("handle").

.. hint::
  To learn more about Bézier curves, take a look at the excellent
  online textbook `A Primer on Bézier curves <https://pomax.github.io/bezierinfo/>`__
  by `Pomax <https://twitter.com/TheRealPomax>`__ -- there is a playground representing
  quadratic Bézier curves `in §1 <https://pomax.github.io/bezierinfo/#introduction>`__,
  the red and yellow points are "anchors", and the green and blue
  points are "handles".

In contrast to :class:`.Mobject`, :class:`.VMobject` can be displayed
on screen (even though, technically, it is still considered a base class).
To illustrate how points are processed, consider the following short example
of a :class:`.VMobject` with 6 points (and thus made out of 6/3 = 2 cubic
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
we remove all the mobjects that are being added -- this is to make
sure we don't add a :class:`.Mobject` twice! After that, we can safely
add it to :attr:`.Scene.mobjects`.

We will hear more from :class:`.Scene` soon.
Before we do that, let us look at the next line
of code in our toy example, which includes the initialization of
an animation class,
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
:meth:`.Animation.__init__` merely sets some attributes derived
from the passed keyword arguments and additionally ensures that
the :attr:`~Animation.starting_mobject` and :attr:`~.Animation.mobject`
attributes are populated. Once the animation is played, the
:attr:`~.Animation.starting_mobject` attribute holds an unmodified copy of the
mobject the animation is attached to; during the initialization
it is set to a placeholder mobject. The :attr:`~.Animation.mobject` attribute
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

The Animation Buffer
^^^^^^^^^^^^^^^^^^^^
There's an attribute of animations that we have glossed
over, and that is :attr:`.Animation.buffer`, of type :class:`.SceneBuffer`.
The :attr:`~.Animation.buffer` is the animations way of communicating
with what happens on the scene. If you want to modify
the scene during the interpolation stage (outside of :meth:`~.Animation.begin` or :meth:`~.Animation.finish`),
the attribute :attr:`.Animation.apply_buffer` is what tells the scene that the buffer
should be processed.

For example, an animation that adds a circle to the scene every frame might look like this

.. code-block:: python

   class CircleAnimation(Animation):
      def begin(self) -> None:
          self.circles = VGroup()

      def interpolate(self, alpha: float) -> None:
          # create and arrange the circles
          self.circles.add(Circle())
          self.circles().arrange()
          # add the new circle to the scene
          self.buffer.add(self.circles[-1])
          # make sure the scene actually realizes something changed
          self.apply_buffer = True

Every time the :class:`.Scene` applies the buffer, it gets emptied out
for use the next time.

The ``play`` call: preparing to enter Manim's render loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are finally there, the render loop is in our reach. Let us
walk through the code that is run when :meth:`.Scene.play` is called.

.. note::

   In the future, control will not be passed to the Manager.
   Instead, the Scene will keep track of every animation and
   at the very end, the Manager will render everything.

As you will see when inspecting the method, :meth:`.Scene.play` almost
immediately passes over to the :class:`~.Manager._play` method of the :class:`.Manager`.
The one thing :meth:`.Scene.play` does before that is preparing the animations.
Whenever :attr:`.Mobject.animate` is called, it creates a new object called a
:class:`._AnimationBuilder`. We have to make sure to convert that into an actual
animation by calling it's :meth:`._AnimationBuilder.build` method.
We also have to update the animations with the correct rate functions, lag ratios,
and run time.

.. note::

   Methods in :class:`.Manager` starting with an underscore ``_`` are intended to be
   private, and are not guaranteed to be stable across versions of Manim. The :class:`.Manager`
   class provides some "public" methods (methods not prefixed with ``_``) that can be overridden to
   change the behavior of the program.

.. warning::

   Subcaptions and audio is still in progress


After the :class:`.Scene` has done all the processing of animations,
it hands out control to the :class:`.Manager`. The :class:`.Manager`
then updates the skipping status of the :class:`.Scene`. This makes sure
that if ``-s`` or ``-n`` is used for sections, the scene does the correct
thing.

The next important line is::

    self._write_hashed_movie_file()

Here, the :class:`.Manager` checks whether or not Manim's caching system should
be used. The idea of the caching system is simple: for every play call, a
hash value is computed, which is then stored and upon re-rendering the scene,
the hash is generated again and checked against the stored value. If it is the
same, the cached output is reused, otherwise it is fully rerendered again.
We will not go into details of the caching system here; if you would like
to learn more, the :func:`.get_hash_from_play_call` function in the
:mod:`.utils.hashing` module is essentially the entry point to the caching
mechanism.

In the event that the animation has to be rendered, the manager asks
its :class:`.FileWriter` to open an output container. The process
is started by a call to ``libav`` and opens a container to which rendered
raw frames can be written. As long as the output is open, the container
can be accessed via the ``output_container`` attribute of the file writer.

With the writing process in place, the renderer then asks the scene
to "begin" the animations.

First, it literally *begins* all of the animations by calling their
setup methods (:meth:`.Animation.begin`).
In doing so, the mobjects that are newly introduced by an animation
(like via :class:`.Create` etc.) are added to the scene. Furthermore, the
animation suspends updater functions being called on its mobject, and
it sets its mobject to the state that corresponds to the first frame
of the animation.

.. note::

    Implementation of figuring out which mobjects have to be redrawn
    is still in progress.


Up to this very point, we did not actually render any (partial)
image or movie files from the scene yet. This is, however, about to change.
Before we enter the render loop, let us briefly revisit our toy
example and discuss how the generic :meth:`.Scene.play` call
setup looks like there.

For the call that plays the :class:`.ReplacementTransform`, there
is no subcaption to be taken care of. The renderer then asks
the scene to compile the animation data: the passed argument
already is an animation (no additional preparations needed),
there is no need for processing any keyword arguments (as
we did not specify any additional ones to ``play``). The
mobject bound to the animation, ``orange_square``, is already
part of the scene (so again, no action taken). Finally, the run
time is extracted (3 seconds long) and stored in
``Scene.duration``. The renderer then checks whether it should
skip (it should not), then whether the animation is already
cached (it is not). The corresponding animation hash value is
determined and passed to the file writer, which then also calls
``libav`` to start the writing process which waits for rendered
frames from the library.

The scene then ``begin``\ s the animation: for the
:class:`.ReplacementTransform` this means that the animation populates
all of its relevant animation attributes (i.e., compatible copies
of the starting and the target mobject so that it can safely interpolate
between the two).

The mechanism determining static and moving mobjects considers
all of the scenes mobjects (at this point only the
``orange_square``), and determines that the ``orange_square`` is
bound to an animation that is currently played. As a result,
the square is classified as a "moving mobject".

Time to render some frames.


The render loop (for real this time)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now we get to the meat of rendering, which happens in :meth:`.Manager._progress_through_animations`.

- The manager determines the run time of the animations by calling
  :meth:`.Manager._calc_run_time`. This method basically takes the maximum
  ``run_time`` attribute of all of the animations passed to the
  :meth:`.Scene.play` call.
- Then, the progressbar is created by :meth:`.Manager._create_progressbar`,
  which returns a ``tqdm`` `progress bar object <https://tqdm.github.io>`__
  object (from the ``tqdm`` library), or a fake progressbar if
  :attr:`.ManimConfig.write_to_movie` is ``False``.
- Then the *time progression* is constructed via
  :meth:`.Manager._calc_time_progression` method, which returns
  ``np.arange(0, run_time, 1 / config.frame_rate)``. In
  other words, the time progression holds the time stamps (relative to the
  current animations, so starting at 0 and ending at the total animation run time,
  with the step size determined by the render frame rate) of the timeline where
  a new animation frame should be rendered.
- Then the scene iterates over the time progression: for each time stamp ``t``,
  we find the time difference between the current and previous frame (AKA ``dt``).
  We then update the animations in the scene using ``dt`` by
  - iterating over each animation
  - next, we update the animations mobjects
  - then the relative time progression with respect to the current animation
    is computed (``alpha = t / animation.run_time``), which is then used to
    update the state of the animation with a call to :meth:`.Animation.interpolate`.
  - After all of the passed animations have been processed, the updater functions
    of all mobjects in the scene, all meshes, and finally those attached to
    the scene itself are run.

  After updating the animations, we pass ``dt`` to :meth:`.Manager._update_frame` which...

  - ... updates the total time passed
  - Updates all the mobjects by calling :meth:`.Scene._update_mobjects`. This in turn
    iterates over all the mobjects on the screen and updates them.
  - After that, the current state of the scene is computed by :meth:`.Scene.get_state`,
    which returns a :class:`.SceneState`.
  - The state is then passed into :meth:`.Manager._render_frame`, which gets
    the renderer to create the pixels. With :class:`.OpenGLRenderer`, this
    also updates the window. :meth:`~.Manager._render_frame` also checks if it should write a frame,
    and if so, writes a frame via the :class:`.FileWriter`.
  - Finally, it uses a concept of virtual time vs real time to see
    if the right amount of time has passed in the window. The virtual
    time is the amount of time that is supposed to have passed (that is, ``t``).
    The real time is how much time has actually passed in the window
    (current time - start time of play). If the animations are progressing
    faster than they would in real life, it will slow down the window by calling
    :meth:`~.Manager._update_frame` with ``dt=0`` until that's no longer the case.
    This is to make sure that animations never go too fast: it doesn't do anything if
    animations are too slow!

At this point, the internal (Python) state of all mobjects has been updated
to match the currently processed timestamp.

A TL;DR for the render loop, in the context of our toy example, reads as follows:

- The scene finds that a 3 second long animation (the :class:`.ReplacementTransform`
  changing the orange square to the blue circle) should be played. Given the requested
  medium render quality, the frame rate is 30 frames per second, and so the time
  progression with steps ``[0, 1/30, 2/30, ..., 89/30]`` is created.
- In the internal render loop, each of these time stamps is processed:
  there are no updater functions, so effectively the manager updates the
  state of the transformation animation to the desired time stamp (for example,
  at time stamp ``t = 45/30``, the animation is completed to a rate of
  ``alpha = 0.5``).
- Then the manager asks the renderer to do its job. The renderer then produces
  the pixels, which are then fed into the :class:`.FileWriter`.
- At the end of the loop, 90 frames have been passed to the file writer.

Completing the render loop
^^^^^^^^^^^^^^^^^^^^^^^^^^

The last few steps in the :meth:`.Manager._play` call are not too
exciting: for every animation, the corresponding :meth:`.Animation.finish`
method is called.

.. NOTE::

  Note that as part of :meth:`.Animation.finish`, the :meth:`.Animation.interpolate`
  method is called with an argument of 1.0 -- you might have noticed already that
  the last frame of an animation can sometimes be a bit off or incomplete.
  This is by current design! The last frame rendered in the render loop (and displayed
  for a duration of ``1 / frame_rate`` seconds in the rendered video) corresponds to
  the state of the animation ``1 / frame_rate`` seconds before it ends. To display
  the final frame as well in the video, we would need to append another ``1 / frame_rate``
  seconds to the video -- which would then mean that a 1 second rendered Manim video
  would be slightly longer than 1 second. We decided against this at some point.

In the end, the time progression is closed (which completes the displayed progress bar)
in the terminal.

This pretty much concludes the walkthrough of a :class:`.Scene.play` call,
and actually there is not too much more to say for our toy example either: at
this point, a partial movie file that represents playing the
:class:`.ReplacementTransform` has been written. The initialization of
the :class:`.Dot` happens analogous to the initialization of ``blue_circle``,
which has been discussed above. The :meth:`.Mobject.add_updater` call literally
just attaches a function to the ``updaters`` attribute of the ``small_dot``. And
the remaining :meth:`.Scene.play` and :meth:`.Scene.wait` calls follow the
exact same procedure as discussed in the render loop section above; each such call
produces a corresponding partial movie file.

Once the :meth:`.Scene.construct` method has been fully processed (and thus all
of the corresponding partial movie files have been written), the
scene calls its cleanup method :meth:`.Scene.tear_down`, and then
asks its renderer to finish the scene. The renderer, in turn, asks
its scene file writer to wrap things up by calling :meth:`.SceneFileWriter.finish`,
which triggers the combination of the partial movie files into the final product.

And there you go! This is a more or less detailed description of how Manim works
under the hood. While we did not discuss every single line of code in detail
in this walkthrough, it should still give you a fairly good idea of how the general
structural design of the library looks like.
