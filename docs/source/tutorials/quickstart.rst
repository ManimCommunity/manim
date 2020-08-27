Quickstart
==========

This document will lead you step by step through the necessary procedure to get
started with manim for the first time as soon as possible.  This tutorial
assumes you have already installed manim following the steps in
:doc:`../installation`.


Start a new project
*******************

To start a new manim video project, all you need to do is choose a single
folder where all of the files related to the video will reside.  For this
example, this folder will be called ``project``,

.. code-block:: bash

   project/

Every file containing code that produces a video with manim will be stored
here, as well as any output files that manim produces and configuration files
that manim needs.


Your first Scene
****************

To produce your first scene, create a new file in your project folder called
``scene.py``,

.. code-block:: bash

   project/
   └─scene.py

and copy the following code in it.

.. code-block:: python

   from manim import *

   # all code must be contained inside the construct
   # method of a class that inherits from Scene
   class SquareToCircle(Scene):
       def construct(self):
           circle = Circle()                   # create a circle
           circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
           self.play(ShowCreation(circle))     # show the circle on screen

Then open your command line, navigate to your project directory, and execute
the following command:

.. code-block:: bash

   $ manim scene.py SquareToCircle -pl

After showing some output, manim should render the scene into a .mp4 file,
and open that file with the default movie player application.  You should see a
video playing the following animation.

.. image:: ../_static/quickstart/first_scene.gif
    :align: center
    :alt: first scene output

If you see the video and it looks correct, congrats! You just wrote your first
manim scene from scratch.  If you get an error message instead, or if do not
see a video, or if the video output does not look like this, it is likely that
manim has not been installed correctly. Please refer to the
:doc:`../installation/troubleshooting` page for more information.


Some bells and whistles
***********************

Our scene is a little basic, so let's add some bells and whistles.  Modify the
``scene.py`` file to contain the following:

.. code-block:: python

   from manim import *

   class SquareToCircle(Scene):
       def construct(self):
           circle = Circle()                    # create a circle
           circle.set_fill(PINK, opacity=0.5)   # set color and transparency

           square = Square()                    # create a square
           square.flip(RIGHT)                   # flip horizontally
           square.rotate(-3 * TAU / 8)          # rotate a certain amount

           self.play(ShowCreation(square))      # animate the creation of the square
           self.play(Transform(square, circle)) # interpolate the square into the circle
           self.play(FadeOut(square))           # fade out animation

And render it using the following command:

.. code-block:: bash

   $ manim scene.py SquareToCircle -pl

The output should look as follows.

.. image:: ../_static/quickstart/second_scene.gif
    :align: center
    :alt: second scene output

This example shows one of the most basic features of manim: the ability to
implement complicated and mathematically-intensive animations (such as cleanly
interpolating between two geometric shapes) in very few lines of code.


You're done!
************

With a working installation of manim, and the bare basics under your belt, it
is now time to start creating awesome mathematical animations.  For a look
under the hood at what manim is doing when rendering the ``SquareToCircle``
scene, see the next tutorial :doc:`a_deeper_look`.  For an extensive review of
manim's features, as well as its configuration and other settings, see the
other :doc:`../tutorials`.  For a list of all available features, see the
:doc:`../reference` page.
