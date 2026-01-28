from manim import *

class IntegralExample(Scene):
    def construct(self):
        # 1. Create Axes
        ax = Axes(x_range=[0, 3], y_range=[0, 9], axis_config={"include_tip": True})
        labels = ax.get_axis_labels(x_label="x", y_label="f(x)")

        # 2. Define the Function (x squared)
        curve = ax.plot(lambda x: x**2, x_range=[0, 3], color=BLUE)

        # 3. Create the Integral Area
        area = ax.get_area(curve, x_range=[0, 2], color=GREY, opacity=0.5)

        # 4. Add LaTeX Label
        integral_text = MathTex(r"\int_{0}^{2} x^2 \,dx", color=WHITE).to_corner(UL)

        # 5. Play Animations
        self.play(Create(ax), Write(labels))
        self.play(Create(curve))
        self.play(FadeIn(area), Write(integral_text))
        self.wait(2)

.. code-block:: python

   class DifferentRotations(Scene):
       def construct(self):
           left_square = Square(color=BLUE, fill_opacity=0.7).shift(2 * LEFT)
           right_square = Square(color=GREEN, fill_opacity=0.7).shift(2 * RIGHT)
           self.play(
               left_square.animate.rotate(PI), Rotate(right_square, angle=PI), run_time=2
           )
           self.wait()

4. Render ``DifferentRotations`` by running the following command in the command line:

.. code-block:: bash

   manim -pql scene.py DifferentRotations

The following animation will render:

.. manim:: DifferentRotations2
   :hide_source:

   class DifferentRotations2(Scene):
       def construct(self):
           left_square = Square(color=BLUE, fill_opacity=0.7).shift(2*LEFT)
           right_square = Square(color=GREEN, fill_opacity=0.7).shift(2*RIGHT)
           self.play(left_square.animate.rotate(PI), Rotate(right_square, angle=PI), run_time=2)
           self.wait()

This ``Scene`` illustrates the quirks of ``.animate``. When using ``.animate``, Manim
actually takes a ``Mobject``'s starting state and its ending state and interpolates the two.
In the ``AnimatedSquareToCircle`` class, you can observe this when the square rotates:
the corners of the square appear to contract slightly as they move into the positions required
for the first square to transform into the second one.

In ``DifferentRotations``, the difference between ``.animate``'s interpretation of rotation and the
``Rotate`` method is far more apparent. The starting and ending states of a ``Mobject`` rotated 180 degrees
are the same, so ``.animate`` tries to interpolate two identical objects and the result is the left square.
If you find that your own usage of ``.animate`` is causing similar unwanted behavior, consider
using conventional animation methods like the right square, which uses ``Rotate``.


``Transform`` vs ``ReplacementTransform``
*****************************************
The difference between ``Transform`` and ``ReplacementTransform`` is that ``Transform(mob1, mob2)`` transforms the points
(as well as other attributes like color) of ``mob1`` into the points/attributes of ``mob2``.

``ReplacementTransform(mob1, mob2)`` on the other hand literally replaces ``mob1`` on the scene with ``mob2``.

The use of ``ReplacementTransform`` or ``Transform`` is mostly up to personal preference. They can be used to accomplish the same effect, as shown below.

.. code-block:: python

    class TwoTransforms(Scene):
        def transform(self):
            a = Circle()
            b = Square()
            c = Triangle()
            self.play(Transform(a, b))
            self.play(Transform(a, c))
            self.play(FadeOut(a))

        def replacement_transform(self):
            a = Circle()
            b = Square()
            c = Triangle()
            self.play(ReplacementTransform(a, b))
            self.play(ReplacementTransform(b, c))
            self.play(FadeOut(c))

        def construct(self):
            self.transform()
            self.wait(0.5)  # wait for 0.5 seconds
            self.replacement_transform()


However, in some cases it is more beneficial to use ``Transform``, like when you are transforming several mobjects one after the other.
The code below avoids having to keep a reference to the last mobject that was transformed.

.. manim:: TransformCycle

    class TransformCycle(Scene):
        def construct(self):
            a = Circle()
            t1 = Square()
            t2 = Triangle()
            self.add(a)
            self.wait()
            for t in [t1,t2]:
                self.play(Transform(a,t))

************
You're done!
************

With a working installation of Manim and this sample project under your belt,
you're ready to start creating animations of your own.  To learn
more about what Manim is doing under the hood, move on to the next tutorial:
:doc:`output_and_config`.  For an overview of
Manim's features, as well as its configuration and other settings, check out the
other :doc:`Tutorials <../tutorials/index>`.  For a list of all available features, refer to the
:doc:`../reference` page.
