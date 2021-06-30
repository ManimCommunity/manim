##########
Using Text
##########

There are two different ways by which you can render **Text** in videos:

1. Using Pango (:mod:`~.text_mobject`)
2. Using LaTeX (:mod:`~.tex_mobject`)

If you want to render simple text, you should use either :class:`~.Text` or 
:class:`~.MarkupText`, or one of its derivatives like :class:`~.Paragraph`.
See :ref:`using-text-objects` for more information.

LaTeX should be used when you need mathematical typesetting. See 
:ref:`rendering-with-latex` for more information.

.. _using-text-objects:

Text Without LaTeX
******************

The simplest way to add text to your animations is to use the :class:`~.Text`
class. It uses the `Pango library`_ to render text. With Pango, you can also
render non-English alphabets like 你好 or  こんにちは or 안녕하세요 or
مرحبا بالعالم.

Here is a simple *Hello World* animation.

.. manim:: HelloWorld 
    :save_last_frame:
    :ref_classes: Text

    class HelloWorld(Scene):
        def construct(self):
            text = Text("Hello world").scale(3)
            self.add(text)

You can also use :class:`~.MarkupText` which allows the use of PangoMarkup
(see the documentation of :class:`~.MarkupText` for details) to render text.
For example:

.. manim:: SingleLineColor 
    :save_last_frame:
    :ref_classes: MarkupText

    class SingleLineColor(Scene):
        def construct(self):
            text = MarkupText(f'all in red <span fgcolor="{YELLOW}">except this</span>', color=RED)
            self.add(text)

.. _Pango library: https://pango.gnome.org

Working with :class:`~.Text`
============================

This section explains the properties of :class:`~.Text` and how can it be used
in your animations.

Using Fonts
-----------

You can set a different font using :attr:`~.Text.font`.

.. note:: 

    The font used must be installed in your system, and Pango should know
    about it. You can get a list of fonts using :func:`manimpango.list_fonts`.

    >>> import manimpango
    >>> manimpango.list_fonts()
    [...]


.. manim:: FontsExample 
    :save_last_frame:
    
    class FontsExample(Scene):
        def construct(self):
            ft = Text("Noto Sans", font="Noto Sans")
            self.add(ft)

Setting Slant and Weight
------------------------
Slant is the style of the Text, and it can be ``NORMAL`` (the default), 
``ITALIC`` or ``OBLIQUE``. Usually, for many fonts both ``ITALIC`` and
``OBLIQUE`` look similar, but ``ITALIC`` uses **Roman Style**, whereas
``OBLIQUE`` uses **Italic Style**.

Weight specifies the boldness of a font. You can see a list of weights in
:class:`manimpango.Weight`.

.. manim:: SlantsExample
    :save_last_frame:

    class SlantsExample(Scene):
        def construct(self):
            a = Text("Italic", slant=ITALIC)
            self.add(a)

.. manim:: DifferentWeight 
    :save_last_frame:

    class DifferentWeight(Scene):
        def construct(self):
            import manimpango

            g = VGroup()
            weight_list = dict(sorted({weight: manimpango.Weight(weight).value for weight in manimpango.Weight}.items(), key=lambda x: x[1]))
            for weight in weight_list:
                g += Text(weight.name, weight=weight.name, font="Open Sans")
            self.add(g.arrange(DOWN).scale(0.5))

Using Colors
------------

You can set the color of the text using :attr:`~.Text.color`:

.. manim:: SimpleColor
    :save_last_frame:

    class SimpleColor(Scene):
        def construct(self):
            col = Text("RED COLOR", color=RED)
            self.add(col)

You can use utilities like :attr:`~.Text.t2c` for coloring specific characters.
This may be problematic if your text contains ligatures
as explained in :ref:`iterating-text`.

:attr:`~Text.t2c` accepts two types of dictionaries,

* The keys can contain indices like ``[2:-1]`` or ``[4:8]``, 
  this works similar to how `slicing <https://realpython.com/python-strings/#string-slicing>`_
  works in Python. The values should be the color of the Text from :class:`~.Color`.
  

* The keys contain words or characters which should be colored separately
  and the values should be the color from :class:`~.Color`:

.. manim:: Textt2cExample
    :save_last_frame:

    class Textt2cExample(Scene):
        def construct(self):
            t2cindices = Text('Hello', t2c={'[1:-1]': BLUE}).move_to(LEFT)
            t2cwords = Text('World',t2c={'rl':RED}).next_to(t2cindices, RIGHT)
            self.add(t2cindices, t2cwords)

If you want to avoid problems when using colors (due to ligatures), consider using
:class:`MarkupText`.


Using Gradients
---------------

You can add a gradient using :attr:`~.Text.gradient`. The value must
be an iterable of any length:

.. manim:: GradientExample
    :save_last_frame:

    class GradientExample(Scene):
        def construct(self):
            t = Text("Hello", gradient=(RED, BLUE, GREEN)).scale(2)
            self.add(t)

You can also use :attr:`~.Text.t2g` for gradients with specific 
characters of the text. It shares a similar syntax to :ref:`Using Colors`:

.. manim:: t2gExample
    :save_last_frame:

    class t2gExample(Scene):
        def construct(self):
            t2gindices = Text(
                'Hello',
                t2g={
                    '[1:-1]': (RED,GREEN),
                },
            ).move_to(LEFT)
            t2gwords = Text(
                'World',
                t2g={
                    'World':(RED,BLUE),
                },
            ).next_to(t2gindices, RIGHT)
            self.add(t2gindices, t2gwords)

Setting Line Spacing
--------------------

You can set the line spacing using :attr:`~.Text.line_spacing`:

.. manim:: LineSpacing
    :save_last_frame:

    class LineSpacing(Scene):
        def construct(self):
            a = Text("Hello\nWorld", line_spacing=1)
            b = Text("Hello\nWorld", line_spacing=4)
            self.add(Group(a,b).arrange(LEFT, buff=5))


.. _disable-ligatures:

Disabling Ligatures
-------------------

By disabling ligatures you would get a one-to-one mapping between characters and
submobjects. This fixes the issues with coloring text. 


.. warning::

    Be aware that using this method with text that heavily depends on
    ligatures (Arabic text) may yield unexpected results.

You can disable ligatures by passing ``disable_ligatures`` to 
:class:`Text`. For example:

.. manim:: DisableLigature
    :save_last_frame:

    class DisableLigature(Scene):
        def construct(self):
            li = Text("fl ligature").scale(2)
            nli = Text("fl ligature", disable_ligatures=True).scale(2)
            self.add(Group(li, nli).arrange(DOWN, buff=.8))

.. _iterating-text:

Iterating :class:`~.Text`
-------------------------

Text objects behave like :class:`VGroups <.VGroup>`. Therefore, you can slice and index
the text.

For example, you can set each letter to different color by iterating it.

.. manim:: IterateColor
    :save_last_frame:

    class IterateColor(Scene):
        def construct(self):
            text = Text("Colors").scale(2)
            for letter in text:
                letter.set_color(random_bright_color())
            self.add(text)

.. warning::

    Please note that `Ligature`_ can cause problems here. If you need a
    one-to-one mapping of characters to submobjects you should pass
    the ``disable_ligatures`` parameter to :class:`~.Text`.
    See :ref:`disable-ligatures`.

.. _Ligature: https://en.wikipedia.org/wiki/Ligature_(writing)

Working with :class:`~.MarkupText`
==================================

MarkupText is similar to :class:`~.Text`, the only difference between them is 
that this accepts and processes PangoMarkup (which is similar to
html), instead of just rendering plain text.

Consult the documentation of :class:`~.MarkupText` for more details
and further references about PangoMarkup.

.. manim:: MarkupTest 
    :save_last_frame:

    class MarkupTest(Scene):
        def construct(self):
            text = MarkupText(
                f'<span underline="double" underline_color="green">double green underline</span> in red text<span fgcolor="{YELLOW}"> except this</span>',
                color=RED,
            ).scale(0.7)
            self.add(text)

.. _rendering-with-latex:

Text With LaTeX
***************

Just as you can use :class:`~.Text` to add text to your videos, you can
use :class:`~.Tex` to insert LaTeX.

For example,

.. manim:: HelloLaTeX 
    :save_last_frame:

    class HelloLaTeX(Scene):
        def construct(self):
            tex = Tex(r"\LaTeX").scale(3)
            self.add(tex)

.. note::

    Note that we are using a raw string (``r'...'``) instead of a regular string (``r'...'``).
    This is because TeX code uses a lot of special characters - like ``\`` for example - that
    have special meaning within a regular python string. An alternative would have been to
    write ``\\`` to escape the backslash: ``Tex('\\LaTeX')``.

Working with :class:`~.MathTex`
===============================

Everything passed to :class:`~.MathTex` is in math mode by default. To be more precise,
:class:`~.MathTex` is processed within an ``align*`` environment. You can achieve a
similar effect with :class:`~.Tex` by enclosing your formula with ``$`` symbols:
``$\xrightarrow{x^6y^8}$``:

.. manim:: MathTeXDemo 
    :save_last_frame:

    class MathTeXDemo(Scene):
        def construct(self):
            rtarrow0 = MathTex(r"\xrightarrow{x^6y^8}").scale(2)
            rtarrow1 = Tex(r"$\xrightarrow{x^6y^8}$").scale(2)
            
            self.add(VGroup(rtarrow0, rtarrow1).arrange(DOWN))


LaTeX commands and keyword arguments
====================================

We can use any standard LaTeX commands in the AMS maths packages. Such
as the ``mathtt`` math-text type or the ``looparrowright`` arrow.

.. manim:: AMSLaTeX
    :save_last_frame:

    class AMSLaTeX(Scene):
        def construct(self):
            tex = Tex(r'$\mathtt{H} \looparrowright$ \LaTeX').scale(3)
            self.add(tex)

On the Manim side, the :class:`~.Tex` class also accepts attributes to 
change the appearance of the output. This is very similar to the 
:class:`~.Text` class. For example, the ``color`` keyword changes the
color of the TeX mobject.

.. manim:: LaTeXAttributes
    :save_last_frame:

    class LaTeXAttributes(Scene):
        def construct(self):
            tex = Tex(r'Hello \LaTeX', color=BLUE).scale(3)
            self.add(tex)

Extra LaTeX Packages
====================

Some commands require special packages to be loaded into the TeX template. 
For example, to use the ``mathscr`` script, we need to add the ``mathrsfs``
package. Since this package isn't loaded into Manim's tex template by default,
we have to add it manually.

.. manim:: AddPackageLatex
    :save_last_frame:

    class AddPackageLatex(Scene):
        def construct(self):
            myTemplate = TexTemplate()
            myTemplate.add_to_preamble(r"\usepackage{mathrsfs}")
            tex = Tex(r'$\mathscr{H} \rightarrow \mathbb{H}$}', tex_template=myTemplate).scale(3)
            self.add(tex)

Substrings and parts
====================

The TeX mobject can accept multiple strings as arguments. Afterwards you can
refer to the individual parts either by their index (like ``tex[1]``), or by
selecting parts of the tex code. In this example, we set the color
of the ``\bigstar`` using :func:`~.set_color_by_tex`:

.. manim:: LaTeXSubstrings
    :save_last_frame:

    class LaTeXSubstrings(Scene):
        def construct(self):
            tex = Tex('Hello', r'$\bigstar$', r'\LaTeX').scale(3)
            tex.set_color_by_tex('igsta', RED)
            self.add(tex)

Note that :func:`~.set_color_by_tex` colors the entire substring containing
the Tex, not just the specific symbol or Tex expression. Consider the following example:

.. manim:: IncorrectLaTeXSubstringColoring
    :save_last_frame:

    class IncorrectLaTeXSubstringColoring(Scene):
        def construct(self):
            equation = MathTex(
                r"e^x = x^0 + x^1 + \frac{1}{2} x^2 + \frac{1}{6} x^3 + \cdots + \frac{1}{n!} x^n + \cdots"
            )
            equation.set_color_by_tex("x", YELLOW)
            self.add(equation)

As you can see, this colors the entire equation yellow, contrary to what 
may be expected. To color only ``x`` yellow, we have to do the following:

.. manim:: CorrectLaTeXSubstringColoring
    :save_last_frame:

    class CorrectLaTeXSubstringColoring(Scene):
        def construct(self):
            equation = MathTex(
                r"e^x = x^0 + x^1 + \frac{1}{2} x^2 + \frac{1}{6} x^3 + \cdots + \frac{1}{n!} x^n + \cdots",
                substrings_to_isolate="x"
            )
            equation.set_color_by_tex("x", YELLOW)
            self.add(equation)

By setting ``substring_to_isolate`` to ``x``, we split up the
:class:`~.MathTex` into substrings automatically and isolate the ``x`` components 
into individual substrings. Only then can :meth:`~.set_color_by_tex` be used 
to achieve the desired result.

Note that Manim also supports a custom syntax that allows splitting
a TeX string into substrings easily: simply enclose parts of your formula
that you want to isolate with double braces. In the string
``MathTex(r"{{ a^2 }} + {{ b^2 }} = {{ c^2 }}")``, the rendered mobject
will consist of the substrings ``a^2``, ``+``, ``b^2``, ``=``, and ``c^2``.
This makes transformations between similar text fragments easy
to write using :class:`~.TransformMatchingTex`.

LaTeX Maths Fonts - The Template Library
========================================

Changing fonts in LaTeX when typesetting mathematical formulae is 
tricker than regular text. It requires changing the template that is used
to compile the TeX. Manim comes with acollection of :class:`~.TexFontTemplates` 
ready for you to use. These templates will all work in math mode:

.. manim:: LaTeXMathFonts
    :save_last_frame:

    class LaTeXMathFonts(Scene):
        def construct(self):
            tex = Tex(r'$x^2 + y^2 = z^2$', tex_template=TexFontTemplates.french_cursive).scale(3)
            self.add(tex)

Manim also has a :class:`~.TexTemplateLibrary` containing the TeX 
templates used by 3Blue1Brown. One example is the ctex template,
used for typesetting Chinese script. For this to work, the ctex LaTeX package
must be installed on your system. Furthermore, if you are only 
typesetting Text, you probably do not need :class:`~.Tex` at all, and 
should use :class:`~.Text` instead.

.. manim:: LaTeXTemplateLibrary
    :save_last_frame:

    class LaTeXTemplateLibrary(Scene):
        def construct(self):
            tex = Tex('Hello 你好 \\LaTeX', tex_template=TexTemplateLibrary.ctex).scale(3)
            self.add(tex)


Aligning formulae
=================

:class:`~.MathTex` mobject is typeset in the LaTeX  ``align*``
environment. This means you can use the ``&`` alignment character 
when typesetting multiline formulae:

.. manim:: LaTeXAlignEnvironment
    :save_last_frame:

    class LaTeXAlignEnvironment(Scene):
        def construct(self):
            tex = MathTex(r'f(x) &= 3 + 2 + 1\\ &= 5 + 1 \\ &= 6').scale(2)
            self.add(tex)
