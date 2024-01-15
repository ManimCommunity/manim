******************************************************
An Adventure through Manim's Features and Capabilities
******************************************************

.. image:: ../_static/AdventureManim.png
    :align: center


**Authors:** `Tristan Schulz <https://github.com/MrDiver>`__ and `Aarush Deshpande <https://github.com/JasonGrace2282>`__

.. note:: This is a work in progress guide and might not be complete at this point

##############
What to expect
##############
This guide will take you on a Tour through the features and capabilities of Manim. The goal is to give you a good overview of what Manim can do and how to use it. It is not meant to be a complete reference, but rather a starting point for your own explorations.

The goal of this guide is to give you a clear path from the basics of Manim to a finished animation. It will not go into detail about the inner workings of Manim, but rather focus on the practical aspects of using it.
At the end of this guide you should be able to create your own animations and have a good understanding of how to use Manim.

.. warning::
    Please note that this guide is only for Manim and expects basic knowledge about programming with Python. If you are new to Python you should first learn the basics of Python before you start with Manim.
    You can find a full introduction to Python here: https://docs.python.org/3/tutorial/

    You can still follow this guide with basic knowledge of Python, but you will have to learn some Python basics along the way in order to understand the code examples.

#################
What are Mobjects
#################

Mobjects are the basic building blocks of Manim. They are the objects that are animated and displayed on the screen.
Mobjects can be anything from simple shapes to complex 3D objects. They can be animated, moved, rotated, scaled and much more.
In this guide we will focus on the 2D Mobjects, but the same principles apply to 3D Mobjects as well.

.. manim:: MobjectsFloating
    :hide_source:

    class MobjectsFloating(Scene):
        def construct(self):
            c = Circle()
            s = Square()
            t = Triangle()
            c.shift(UP)
            t.shift(LEFT*3+DOWN)
            s.shift(RIGHT*3+DOWN)
            self.add(c, s, t)
            timer = ValueTracker(0)
            c.add_updater(lambda m: m.move_to(UP+0.2*DOWN*np.sin(timer.get_value()+1)))
            s.add_updater(lambda m: m.move_to(RIGHT*3+DOWN+0.3*DOWN*np.sin(timer.get_value()+2)))
            t.add_updater(lambda m: m.move_to(LEFT*3+DOWN+0.3*DOWN*np.sin(timer.get_value()+4)))
            self.add(timer)
            self.play(timer.animate.set_value(2*np.pi), run_time=5, rate_func=linear)



For a list of all Mobjects you can look at the :doc:`/reference_index/mobjects` Documentation Page. There are many more to explore and you can even create your own Mobjects, which we will cover later.


.. manim:: PredefinedMobjects
    :save_last_frame:
    :hide_source:

    class PredefinedMobjects(Scene):
        def construct(self):
            c = Circle()
            s = Square().set_color(GREEN)
            t = Triangle()
            graph = FunctionGraph(lambda x: np.sin(x))
            axis = Axes()
            par = ParametricFunction(lambda t: np.array([np.cos(3*t), np.sin(2*t), 0]), [0, 2*np.pi])
            mat = Matrix([["\\pi", 0], [0, 1]])
            chart = BarChart(
                        values=[-5, 40, -10, 20, -3],
                        bar_names=["one", "two", "three", "four", "five"],
                        y_range=[-20, 50, 10],
                        y_length=6,
                        x_length=10,
                        x_axis_config={"font_size": 36},
                    ).shift(RIGHT/2)
            func = lambda pos: ((pos[0] * UR + pos[1] * LEFT) - pos) / 3
            vecfield = ArrowVectorField(func,x_range=[-3,3],y_range=[-3,3])
            cross = VGroup(
                Line(UP + LEFT, DOWN + RIGHT),
                Line(UP + RIGHT, DOWN + LEFT))
            a = Circle().set_color(RED).scale(0.5)
            b = cross.set_color(BLUE).scale(0.5)
            t3 = MobjectTable(
                [[a.copy(),b.copy(),a.copy()],
                [b.copy(),a.copy(),a.copy()],
                [a.copy(),b.copy(),b.copy()]])
            t3.add(Line(
                t3.get_corner(DL), t3.get_corner(UR)
            ).set_color(RED))

            group = [c, s, t, graph, axis, par, mat, chart, vecfield, t3]
            names = ["Circle", "Square", "Triangle", "FunctionGraph", "Axes", "ParametricFunction", "Matrix", "BarChart" ,"ArrowVectorField", "MobjectTable"]
            zipped = zip(group, names)
            combined = []
            for mob, name in zipped:
                square = Square()
                name = Text(name).scale(0.5)
                mob.scale_to_fit_width(square.get_width())
                square.scale(1.2)
                name.next_to(square, DOWN)
                group = VGroup(mob, name, square)
                combined.append(group)

            all = VGroup(*combined).arrange_in_grid(buff=1,rows=2).scale(0.8).to_edge(UP)
            dots = MathTex("\\dots").next_to(all, DOWN, buff=1)
            self.add(all, dots)

.. note::
    The type of Mobject that is used most of the time is the `VMobject`. This is a Mobject that is made up of `VectorizedPoints`. These are points that are defined by their coordinates and can be connected by lines or curves.
    Every time we talk about Mobjects in this guide we mean VMobjects, unless we state otherwise.

=============================
Mobjects and their Attributes
=============================

In order to display Mobjects in your animations you need to add them to the scene. You can do this by calling ``self.add(mobject)`` in the ``construct`` method of your scene.
This tells Manim that you want to display the Mobject in your scene.

.. manim:: CreatingMobjects
    :save_last_frame:

    class CreatingMobjects(Scene):
        def construct(self):
            c = Circle()
            self.add(c)

This will be the basic structure of all your animations. You will create Mobjects and add them to the scene. Then you can animate them and change their properties.
Try the "Make Interactive" Button and see if you can create a `Square` instead of a `Circle`.

The first line is the name of your scene, in this case it is ``CreatingMobjects``. It inherits from ``Scene``: as we explore later, you'll find examples where we inherit from
class other than ``Scene`` to gain access to more specialized methods. Your animation must take place in the ``construct`` method of your scene, otherwise it will not render.

You can run this scene on your local machine by saving it in a file called ``my_first_scene.py`` and running ``manim -pqm my_first_scene.py`` in the terminal.

------------------
Mobject Attributes
------------------

Mobjects also posses many attributes that you can change. For example you can change the color of a Mobject by calling ``mobject.set_color(color)`` or scale it by calling ``mobject.scale(factor)``.

The basic attributes are the ``points``, ``fill_color``, ``fill_opacity``, ``stroke_color``, ``stroke_opacity``, ``stroke_width``.
The ``points`` define the outline of the Mobject, whereas the color attributes define how this outline is displayed.

A full list of the attributes of :class:`VMobject` can be found in the :doc:`../reference/manim.mobject.types.vectorized_mobject.VMobject` Documentation Page. Please note that depending
on the type of Mobject you are using, there might be additional attributes, which are listed on the corresponding documentation page.

-------------------
Changing the Points
-------------------

Most of the function that you will use in Manim will be functions that change the points of a Mobject. For example ``mobject.shift(direction)`` will move the Mobject in the given direction.
On the other hand, ``mobject.rotate(angle)`` will rotate the Mobject by the given angle.

.. manim:: MobjectPoints
    :save_last_frame:

    class MobjectPoints(Scene):
        def construct(self):
            c = Circle()
            s = Square()
            t = Triangle()

            c.shift(3*LEFT)
            s.rotate(PI/4)
            t.shift(3*RIGHT)

            self.add(c, s, t)

------------------
Changing the Color
------------------

Changing the color works in the same way but instead of modifying it you can set it to a new value. For example ``mobject.set_fill(color=color)`` will set the fill color of the Mobject to the given color.

You can also pass in attributes through the constructor of the Mobject. For example ``Circle(fill_color=RED)`` will create a circle with a red fill color.
For a list of parameters that you can pass you can always visit the corresponding Documentation Page in the Reference Manual.

.. manim:: MobjectColor
    :save_last_frame:

    class MobjectColor(Scene):
        def construct(self):
            c = Circle(fill_color=YELLOW).shift(3*LEFT)
            s = Square()
            t = Triangle().shift(3*RIGHT)

            c.set_fill(color=RED).set_opacity(1)
            s.set_stroke(color=GREEN)
            t.set_color(color=BLUE).set_opacity(0.5)

            self.add(c, s, t)


-------------------
Test your Knowledge
-------------------

Now that you saw the basic ways to change Mobjects, try to reproduce the following Image. You can use the "Make Interactive" Button of the above Scene to get started.

.. manim:: TestYourKnowledge1
    :save_last_frame:
    :hide_source:

    class TestYourKnowledge1(Scene):
        def construct(self):
            c = Circle(fill_color=RED,stroke_color=GREEN).shift(3*LEFT)
            s = Square(fill_color=GREEN,stroke_color=BLUE).set_opacity(0.2)
            t = Triangle(fill_color=RED,stroke_opacity=0).shift(RIGHT)

            c.set_fill(color=RED).set_opacity(1)
            s.set_stroke(color=GREEN)
            t.set_color(color=BLUE).set_opacity(0.5)

            self.add(c, s, t)


###################
Animations in Manim
###################

Now that we looked long enough at static Images, let's get to the fun part of Manim. Animations!
Animations are at the core of Manim and are what makes it so powerful. You can animate almost anything in Manim and you can do it in many different ways.
In this section we will look at the different ways to animate Mobjects and how to control the animations.

.. manim:: Manimations1
    :hide_source:

    class Manimations1(Scene):
        def construct(self):
            c = Circle().shift(UP).set_color(RED)
            s = Square().shift(LEFT*3)
            t = Triangle().shift(RIGHT*3)
            l = MathTex(r"\mathbf{M}").shift(DOWN).set_fill(opacity=0).set_stroke(color=WHITE, opacity=1, width=5).scale(4)
            self.play(AnimationGroup(Create(c), GrowFromCenter(s), Write(l), FadeIn(t), lag_ratio=0.2))
            group = VGroup(l,c, s, t)
            self.play(group.animate.arrange(RIGHT))
            self.play(group.animate.arrange(DOWN))
            self.play(group.animate.arrange_in_grid(buff=1,rows=2))
            self.play(Unwrite(group))



================================
Introduction to Basic Animations
================================

There are multiple ways to animate the addition and removal of mobjects from the scene. The most common ways to introduce mobjects is with ``FadeIn`` or ``Create``,
and the most common ways to remove objects from the scene are their counterparts: ``FadeOut`` and ``Uncreate``.

.. manim:: BasicAnimations

   class BasicAnimations(Scene):
      def construct(self):
          c1 = Circle().shift(2*LEFT)
          c2 = Circle().shift(2*RIGHT)
          self.play(FadeIn(c1), Create(c2))
          self.play(FadeOut(c1), Uncreate(c2))

--------
Runtimes
--------

You can adjust the duration of each animation individually, or you can set a duration for all in animations in a ``Scene.play`` call.

.. manim:: AnimationRuntimes

   class AnimationRuntimes(Scene):
      def construct(self):
          c = Circle().shift(2*LEFT)
          s = Square().shift(2*RIGHT)
          # set animation runtimes individually
          self.play(Create(c, run_time=2), Create(s, run_time=1))
          # in this call, the individual runtimes of each animation
          # are overridden by the runtime in the self.play call
          self.play(FadeOut(c, run_time=2), FadeOut(s, run_time=1), run_time=1.5)

--------------
Rate Functions
--------------
A rate function allows you to adjust the speed at an animation proceeds.

.. manim:: RateFunctionsExample

   class RateFunctionsExample(Scene):
      def construct(self):
          c1 = Circle().shift(2*LEFT)
          c2 = Circle().shift(2*RIGHT)
          self.play(
              Create(c1, rate_func=rate_functions.linear),
              Create(c2, rate_func=rate_functions.ease_in_sine),
              run_time=5
          )

You can see all of the current ones below:

.. manim:: AllRateFunctions
    :hide_source:

    class AllRateFunctions(Scene):
        def construct(self):
            time_progress = ValueTracker(0)
            func_grid = VGroup()
            exclude = ["wraps", "bezier", "sigmoid", "unit_interval", "zero", "not_quite_there", "squish_rate_func"]
            rate_funcs = list(filter(
                lambda t: str(t[1])[:10] == "<function " and all(t[0] != s for s in exclude),
                rate_functions.__dict__.items(),
            ))
            for name, rate_func in rate_funcs:
                plot_bg = Rectangle(height=1.5, width=2.0)
                y_zero = DashedLine(stroke_width=1.5, stroke_color=YELLOW)
                y_one = DashedLine(stroke_width=0.5, stroke_color=BLUE).shift(0.5*UP)
                y_minus_one = y_one.copy().shift(DOWN)
                plot_title = (
                    Text(name, weight=SEMIBOLD, font="Open Sans")
                    .scale(0.4)
                    .next_to(plot_bg, UP, buff=0.1)
                )
                func_grid.add(VGroup(plot_bg, y_zero, y_one, y_minus_one, plot_title))

            func_grid.arrange_in_grid(cols=8)
            func_grid.stretch_to_fit_height(0.9 * config.frame_height)
            func_grid.stretch_to_fit_width(0.9 * config.frame_width)
            func_grid.move_to(ORIGIN)

            y_zero, y_one = func_grid.submobjects[0].submobjects[1:3]
            origin = y_zero.get_start()
            height = (y_one.get_start() - origin)[1]
            width = (y_zero.get_end() - origin)[0]

            funcs = []
            dots = VGroup()
            for plot_group, (_, rate_func) in zip(func_grid.submobjects, rate_funcs):
                origin = plot_group.submobjects[1].get_start()
                func = lambda t, o=origin, rf=rate_func: o + np.array([width*t, height*rf(t), 0])
                funcs.append(func)
                plot = (
                    ParametricFunction(
                        func,
                        t_range=[0, 1, 0.01],
                        use_smoothing=False,
                        color=YELLOW,
                    )
                )
                plot_group.add(plot)

                dot = Dot().scale(0.5).move_to(func(0))
                dots.add(dot)

            def dot_updater(dots):
                t = time_progress.get_value()
                for dot, func in zip(dots.submobjects, funcs):
                    dot.move_to(func(t))

            self.add(func_grid, dots)
            dots.add_updater(dot_updater)
            # there is some wacky rate function giving out-of-bounds results...
            self.play(
                time_progress.animate.set_value(1),
                run_time=3,
            )

Alternatively, you can create your own. A rate function takes in a value between 0 and 1 representing the "progress" of the animation. You can think of this as the
ratio of the time passed since the animation started, to the runtime of the animation. It should return how much of the animation should have been completed by that time.

As an example, check out the rate function below.

.. manim:: CustomRateFunctions

   class CustomRateFunctions(Scene):
      def construct(self):
          def there_and_back_three(alpha: float):
              if alpha <= 1/3:
                  return 3*alpha
              elif alpha <= 2/3:
                  return 1-3*(alpha-1/3)
              else:
                  return 3*(alpha-2/3)

          self.play(Create(Circle(), rate_func=there_and_back_three), run_time=4)

----------------------
The ``Wait`` Animation
----------------------

Now all these animations seem a bit rushed. Luckily, Manim allows us to create periods of time where nothing is happening.
Let's look at an example:

.. manim:: BasicAnimationWithWait

   class BasicAnimationWithWait(Scene):
      def construct(self):
          c = Circle()
          self.play(Create(c))
          self.wait() # wait for one second by default
          self.play(FadeOut(c))
          self.wait(0.5) # wait half a second

A little bit later on, we will learn how to leverage the ``stop_condition`` parameter to stop after a certain event happens.

=====================
Transforming Mobjects
=====================

Manim allows us to smoothly transform one ``Mobject`` into another using ``Transform`` (and in just a second, we'll talk about ``ReplacementTransform``).
``Transform(mob1, mob2)`` turns the attributes of ``mob1`` into the attributes of ``mob2``.

.. manim:: TransformAnimation

   class TransformAnimation(Scene):
      def construct(self):
          c = Circle()
          self.add(c)
          self.play(Transform(c, Square()))
          self.play(FadeOut(c)) # fadeout c

-----------------------------------------
``Transform`` vs ``ReplacementTransform``
-----------------------------------------

While ``Transform(mob1, mob2)`` changes the attributes of ``mob1`` to ``mob2``, ``ReplacementTransform(mob1, mob2)`` literally replaces ``mob1`` on the
scene with ``mob2``.

Here is the same scene in the last section, but using ``ReplacementTransform``:

.. manim:: ReplacementTransformAnimation

   class ReplacementTransformAnimation(Scene):
      def construct(self):
          c = Circle()
          s = Square()
          self.add(c)
          self.play(ReplacementTransform(c, s))
          self.play(FadeOut(s)) # fadeout s

Ultimately, the choice of which to use is up to the programmer. However, some examples like the one below make the code simpler when using one over the other.

.. manim:: CyclingShapesAnimation

   class CyclingShapesAnimation(Scene):
      def construct(self):
          mob = Circle()
          shapes = (Square(), Triangle(), Circle().set_fill(color=RED, opacity=0.5))
          self.add(mob)
          for shape in shapes:
              # if we use transform, we avoid having to
              # keep track of the previously transformed
              # shape
              self.play(Transform(mob, shape))
              self.wait(0.3)


-------------------
``.animate`` Syntax
-------------------

One of the most powerful features of Manim is it's ``.animate`` syntax. It allows you to animate the changing of an attribute of a mobject. You can see an example below:

.. manim:: AnimateSyntaxExample

   class AnimateSyntaxExample(Scene):
      def construct(self):
          c = Circle()
          self.add(c)
          self.play(c.animate.shift(RIGHT))
          self.play(c.animate.to_corner(DL).set_fill(color=RED, opacity=0.4))



.. note::

   ``.animate`` works by interpolating between the initial and the final mobject. As such, beware when using ``.animate.rotate`` with angles greater than pi radians
   as it may not produce the intended animation.


-------------------
Test Your Knowledge
-------------------

Try to create the following animation!

.. manim:: TestBasicAnimationKnowledge
    :hide_source:

    class TestBasicAnimationKnowledge(Scene):
        def construct(self):
            c = Circle().set_fill(color=RED, opacity=0.5)
            s = Star().set_stroke(color=YELLOW).set_fill(color=YELLOW, opacity=0.3)
            t = Triangle().set_fill(color=BLUE, opacity=0.1)
            VGroup(c, s, t).arrange(RIGHT).move_to(ORIGIN) # users will arrange manually
            self.play(
                DrawBorderThenFill(c),
                GrowFromPoint(s, ORIGIN),
                SpinInFromNothing(t),
                run_time=2
            )
            self.wait()
            for mob in (s, t):
                self.play(Transform(c, mob))
                self.remove(mob)
                self.wait(0.2)
            self.play(c.animate.move_to(ORIGIN))

Hint: you might need to look at different :doc:`/reference_index/animations`!

=================
Grouping Mobjects
=================

Oftentimes it is convenient to animate the movement of several mobjects at once. To help accomplish this goal, manim provides two classes: ``Group`` and ``VGroup``.
99% of the time, ``VGroup``'s are used, but if you're dealing with some form of an ``ImageMobject`` you will have to use ``Group``. Here's an example of how groups can be useful:

.. manim:: GroupingExample

    class GroupingExample(Scene):
        def construct(self):
            tri = Triangle()
            sq = Square()
            circ = Circle()
            grp1 = VGroup(tri,sq,circ).arrange(RIGHT)
            grp2 = VGroup(tri,circ)
            self.add(tri,sq,circ)
            self.play(grp1.animate.shift(UP))
            self.play(grp2.animate.shift(2*DOWN))
            self.play(tri.animate.next_to(circ,RIGHT))
            self.play(grp1.animate.shift(UP))
            self.wait()

.. note::
   From now onwards, if we refer to a group we are referring to a ``VGroup``, unless specifically stated otherwise.

Groups also have a bunch of methods to make your life easier. Take a look at some in the example below:

.. manim:: GroupingMethodsExample

   class GroupingMethodsExample(Scene):
        def construct(self):
            group = VGroup(
                Square(),
                Star(color=YELLOW).set_fill(color=YELLOW, opacity=0.5),
                Triangle(),
                Circle().set_fill(color=RED, opacity=0.5)
            )
            self.play(group.animate.arrange(DOWN), run_time=2)
            self.play(group.animate.arrange_in_grid(cols=2), run_time=2)
            for mob in group:
                self.play(Uncreate(mob))
            self.wait(0.2)

##################
Syncing Animations
##################

In many animations it makes sense to have things moving together at the same rate.
However, Manim gives you better ways to accomplish this task then by copying the same parameters
everywhere.

=========
Updaters
=========
Manim allows you to "update" the attributes of a mobject every frame of an animation
via something called updaters. There are two types: normal updaters, and time-based updaters.

.. note::
    The way manim works with time based updaters is going to be reworked at some point. Stay
    up to date with the changelogs to make sure your code will work.

---------------
Normal Updaters
---------------
You can attach an updater to a mobject via the `.add_updater` method. It takes a function whose
first parameter is the mobject itself, and you can modify the mobject however you want.

For example, here we used ``lambda m: m.next_to(d, RIGHT)``. In this case, ``m`` is the Mobject ``Text("Hi!")``.

.. manim:: UpdaterExample
    :ref_classes: MoveAlongPath

    class UpdaterExample(Scene):
        def construct(self):
            t = Text("Hi!")
            d = Dot(color=ORANGE)
            trace = TracedPath(d.get_center, dissipating_time=1, stroke_color=RED)
            t.add_updater(lambda m: m.next_to(d, RIGHT))
            self.add(t, trace)
            self.play(MoveAlongPath(d, Square(), rate_func=linear, run_time=3))
            self.wait()

-------------------
Time Based Updaters
-------------------
Time based updaters are just like normal updaters, but take an extra parameter ``dt``.
This represents how much time has passed between the last call of your updater.

.. manim:: TimeBasedUpdater

    class TimeBasedUpdater(Scene):
        def construct(self):
            time = 0
            d = DecimalNumber(0)
            def updater(m: VMobject, dt: float):
                # access the time defined outside this function
                nonlocal time
                time+=dt
                d.set_value(time)
            d.add_updater(updater)
            self.add(d)
            self.wait(1.1)



=============
ValueTrackers
=============

``ValueTracker``s are the real things that allow you to synchronize multiple animations at once.
They are basically just stored values, but you can animate their ``.set_value`` to produce animations.

.. manim:: ValueTrackerShowcase

    class ValueTrackerShowcase(Scene):
        def construct(self):
            line = Rectangle(height=1, width=4).set_stroke(color=WHITE, opacity=1).move_to(ORIGIN)
            vt = ValueTracker(1e-2) # setting to zero creates bugs with stretch_to_fit_width
            progress = Rectangle(height=1, width=vt.get_value()).set_stroke(color=RED,opacity=1)
            progress.add_updater(lambda p: p.stretch_to_fit_width(vt.get_value()).align_to(line, LEFT))
            d = DecimalNumber(0).to_edge(UP)
            d.add_updater(lambda d: d.set_value(vt.get_value()))
            self.add(d,line,progress)
            self.play(vt.animate.set_value(4), rate_func=linear, run_time=1.5)
            self.wait(0.1)

-------------
always_redraw
-------------
``always_redraw`` is a simple function that allows you to recreate a mobject at
every frame of the animation. As an example, check out this animation:

.. manim:: AlwaysRedrawTangentAnimation

    class AlwaysRedrawTangentAnimation(Scene):
        def construct(self):
            ax = Axes()
            sine = ax.plot(np.sin, color=RED)
            alpha = ValueTracker(0)
            point = always_redraw(
                lambda: Dot(
                    sine.point_from_proportion(alpha.get_value()),
                    color=BLUE
                )
            )
            tangent = always_redraw(
                lambda: TangentLine(
                    sine,
                    alpha=alpha.get_value(),
                    color=YELLOW,
                    length=4
                )
            )
            self.add(ax, sine, point, tangent)
            self.play(alpha.animate.set_value(1), rate_func=linear, run_time=2)

-------------------
Test Your Knowledge
-------------------
Try to recreate the following animation!

.. manim:: KnowledgeCheckUpdaters
    :hide_source:

    class KnowledgeCheckUpdaters(Scene):
        def construct(self):
            l1 = Line(6*LEFT,6*RIGHT)
            l2 = Line(4*DL,3*UR)
            vt = ValueTracker(0)
            d1, d2 = Dot(color=RED), Dot(color=ORANGE)
            txt = MathTex(r"\Delta", color=RED).add_updater(lambda t: t.next_to(d2, LEFT)).scale(2)
            bt = TracedPath(d1.get_center, stroke_color=RED)
            tt = TracedPath(d2.get_center, stroke_color=ORANGE)
            d1.add_updater(lambda d: d.move_to(l1.point_from_proportion(vt.get_value())))
            d2.add_updater(lambda d: d.move_to(l2.point_from_proportion(vt.get_value())))
            self.add(d1, d2, bt, tt, txt)
            self.play(vt.animate.set_value(1), run_time=1.5)
            self.play(vt.animate.set_value(0.8))
            self.play(Create(Line(d1.get_center(), d2.get_center(), color=YELLOW)))
            vmob = VMobject(color=ORANGE).set_points_as_corners([ORIGIN, d1.get_center(), d2.get_center(), ORIGIN]).set_fill(color=[RED,ORANGE,YELLOW], opacity=1).set_z_index(-50)
            txt.clear_updaters()
            self.play(Create(vmob), txt.animate.move_to(vmob.get_center()).set_z_index(50).set_color(BLUE))
