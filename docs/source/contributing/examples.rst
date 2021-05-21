===============
Adding Examples
===============

This is a page for adding examples to the documentation. 
Here are some guidelines you should follow before you publish your examples.

Guidelines for examples
-----------------------

Everybody is welcome to contribute examples to the documentation. Since straightforward 
examples are a great resource for quickly learning manim, here are some guidelines.

What makes a great example
--------------------------

.. note:: 

   As soon as a new version of manim is released, the documentation will be a snapshot of that 
   version. Examples contributed after the release will only be shown in the latest documentation.
   
* Examples should be ready to copy and paste for use.

* Examples should be brief yet still easy to understand.

* Examples don't require the ``from manim import *`` statement, this will be added automatically when the docs are built.

* There should be a balance of animated and non-animated examples.

- As manim makes animations, we can include lots of animated examples; however, our RTD has a maximum 20 minutes to build. Animated examples should only be used when necessary, as last frame examples render faster.

- Lots of examples (e.g. size of a plot-axis, setting opacities, making texts, etc.) will also work as images. It is a lot more convenient to see the end product immediately instead of waiting for an animation to reveal it.

* Please ensure the examples run on the current master when you contribute an example.\

* If the functions used are confusing for people, make sure to add comments in the example to explain what they do.

How examples are structured
---------------------------

* Examples can be organized into chapters and subchapters.

- When you create examples, the beginning example chapter should focus on only one functionality. When the functionality is simple, multiple ideas can be illustrated under a single example.

- As soon as simple functionalities are explained, the chapter may include more complex examples which build on the simpler ideas.

Writing examples
~~~~~~~~~~~~~~~~

When you want to add/edit examples, they can be found in the ``docs/source/`` directory, or directly in the manim source code (e.g. ``manim/mobject/mobject.py``). The examples are written in 
``rst`` format and use the manim directive (see :mod:`~.manim_directive` ), ``.. manim::``. Every example is in its own block, and looks like this:

.. code:: rst

    Formulas
    ========

    .. manim:: Formula1
        :save_last_frame:

        class Formula1(Scene):
            def construct(self):
                t = MathTex(r"\int_a^b f'(x) dx = f(b) - f(a)")
                self.add(t)
                self.wait(1)

In the building process of the docs, all ``rst`` files are scanned, and the 
manim directive (``.. manim::``) blocks are identified as scenes that will be run 
by the current version of manim.
Here is the syntax:

* ``.. manim:: [SCENE_NAME]`` has no indentation and ``SCENE_NAME`` refers to the name of the class below.

* The flags are followed in the next line (no blank line here!), with the indentation level of one tab.

All possible flags can be found at :mod:`~.manim_directive`.

In the example above, the ``Formula1`` following ``.. manim::`` is the scene
that the directive expects to render; thus, in the python code, the class
has the same name: ``class Formula1(Scene)``.

.. note::

   Sometimes, when you reload an example in your browser, it has still the old
   website somewhere in its cache. If this is the case, delete the website cache,
   or open a new incognito tab in your browser, then the latest docs
   should be shown. 
   **Only for locally built documentation:** If this still doesn't work, you may need
   to delete the contents of ``docs/source/references`` before rebuilding
   the documentation.