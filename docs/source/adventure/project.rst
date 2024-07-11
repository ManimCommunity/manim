***************
A Basic Project
***************

.. image:: ../_static/AdventureManim.png
    :align: center


**Authors:** `Tristan Schulz <https://github.com/MrDiver>`__ and `Aarush Deshpande <https://github.com/JasonGrace2282>`__

.. note:: This is a work in progress guide and might not be complete at this point

############
Introduction
############
Throughout this guide, we'll walk you through how to create a simple 30 second video about vector addition. If you don't
already know what that is, it's recommended you watch `this<https://youtu.be/fNk_zzaMoSs?si=fQDML214IeNl0OZ1>`_ video
by the original creator of manim, 3Blue1Brown.

The next step is figuring out how the project should look: what content should it cover, in what order, etc. In this
tutorial, we'll focus on two parts of vector addition: the algebraic way, and the geometric way. For the algebraic way,
we'll show two vectors (as matrices) being added, and give a short explanation. After that we'll show the typical tip-to-tail
method for adding vectors graphically. Of course, choosing good examples is very important to help the viewer understand.
In our case, we'll use the two vectors :math:`v_1\equiv\langle 2, 1\rangle` and :math:`v_2\equiv\langle 0,-3 \rangle`.

#########################
Algebraic Vector Addition
#########################

We'll start with the basic setup needed for every manim video. Create
a new file called `vector_addition.py`, and paste the contents here.

.. code-block:: python

    from manim import *


    class VectorAddition(Scene):
        def construct(self) -> None:
            self.play(Create(Circle()))


    # if you're used to rendering from the terminal,
    # this way you can just run the python file instead
    # of executing manim -p vector_addition.py VectorAddition
    with tempconfig({"preview": True}):
        VectorAddition().render()

If you did it correctly, running the python file (either through your IDE
or through the terminal with ``python3 vector_addition.py``) should render a scene
with a circle being created:

.. manim:: CreateCircle
   :hide_source:

   class CreateCircle(Scene):
      def construct(self):
          self.play(Create(Circle()))

============
Introduction
============
First we need to introduce the viewer to what we're going to talk about. Ideally,
it would be an interesting hook, but for the sake of learning the library we will
stick with a simple text-based intro. Try to recreate the following:

.. manim:: AdventureIntro
    :hide_source:
    :ref_classes: Tex Text Write Unwrite Create

    class AdventureIntro(Scene):
        def construct(self):
            intro = Text("Let's try to add two vectors!")
            # put an r"" instead of a normal string so we don't have any special characters like \n
            vec_txts = Tex(r"We'll use $\boldsymbol{\vec{v}_1}=(2, 2)$ and $\boldsymbol{\vec{v}_2}=(0, -3)$")
            self.play(Create(intro))
            self.wait(1)
            # "grey out" the intro and shift it upwards as we write the second line
            self.play(intro.animate.shift(2*UP).set_opacity(0.5), Write(vec_txts))
            self.wait(1)
            self.play(Unwrite(intro), Unwrite(vec_txts), run_time=.5)
            self.wait(0.2)

.. dropdown:: Authors Solution

    .. code-block:: python

        class AdventureIntro(Scene):
            def construct(self):
                intro = Text("Let's try to add two vectors!")
                # put an r"" instead of a normal string so we don't have any special characters like \n
                vec_txts = Tex(
                    r"We'll use $\boldsymbol{\vec{v}_1}=(2, 2)$ and $\boldsymbol{\vec{v}_2}=(0, -3)$"
                )
                self.play(Create(intro))
                self.wait(1)
                # "grey out" the intro and shift it upwards as we write the second line
                self.play(intro.animate.shift(2 * UP).set_opacity(0.5), Write(vec_txts))
                self.wait(1)
                self.play(Unwrite(intro), Unwrite(vec_txts), run_time=0.5)
                self.wait(0.2)


################
The Final Result
################
Putting it all together, we can render the final result.

.. include:: vector_addition.rst
