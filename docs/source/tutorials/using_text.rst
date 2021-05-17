##########
Using Text
##########

There are two different way by which you can render **Text** in videos.

1. Using Pango (:mod:`~.text_mobject`)
2. Using LaTeX (:mod:`~.tex_mobject`)

If you want to render simple Text, you should use either :class:`~.Text` or 
:class:`~.MarkupText` or one of it's detrivaties like :class:`~.Paragraph`.
See :ref:`using-text-objects` for more information.

LaTeX should only be used when you need Mathematical Typesetting.


.. _using-text-objects:

Rendering Text without using LaTeX 
**********************************

The simplest way to add text to your animations is to use the :class:`~.Text`
class. It uses the `Pango library`_ to render text. With Pango, you can also
render non-English alphabets like `你好` or  `こんにちは` or `안녕하세요` or
`مرحبا بالعالم`.

Here, is a simple Hello World animation.

.. manim:: HelloWorld
    :save_last_frame:

    class HelloWorld(Scene):
        def construct(self):
            text = Text('Hello world').scale(3)
            self.add(text)

You can also use :class:`~.MarkupText` where PangoMarkup (:class:`MarkupText`)
can be used to render text. For example,

.. manim:: SingleLineColor
    :save_last_frame:

    class SingleLineColor(Scene):
        def construct(self):
            text = MarkupText(
               f'all in red <span fgcolor="{YELLOW}">except this</span>',
               color=RED
            )
            self.add(text)

.. _Pango library: https://pango.gnome.org

Working with :class:`~.Text`
============================

This section explains the properties of :class:`~.Text` and how can it be used
in your Animations.

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
``ITALIC``, ``OBLIQUE``. Usually, for many fonts both ``ITALIC`` and
``OBLIQUE`` looks similar, but ``ITALIC`` uses **Roman Style**, which 
``OBLIQUE`` uses **Italic Style**.

Weight specifies the boldness of a font. You can see a list in
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
            for i in manimpango.Weight:
                g += Text(i.name, weight=i.name, font="Open Sans")
            self.add(g.arrange(DOWN).scale(0.5))

Using Colors
------------

You can use Colors using :attr:`~.Text.color`. This would color the whole text.

For example,

.. manim:: SimpleColor
    :save_last_frame:

    class SimpleColor(Scene):
        def construct(self):
            col = Text("RED COLOR", color=RED)
            self.add(col)

You can use utilities like :attr:`~.Text.t2c` for coloring characters 
different from others. This may be problematic if your text contain ligatures
as explained in :ref:`iterating-text`.

:attr:`~Text.t2c` accepts two types of dictionaries,

* The keys can contain indices like ``[2:-1]`` or ``[4:8]``, 
  this works similar to how `slicing <https://realpython.com/python-strings/#string-slicing>`_
  works in Python. The values should be the color of the Text from :class:`~.Color`.
  
  .. note:: Negative indices are also supported.

* The keys contain words or character which should be coloured seperately
  and the values should be the color from :class:`~.Color`.

For example,

.. manim:: Textt2cExample
    :save_last_frame:

    class Textt2cExample(Scene):
        def construct(self):
            t2cindices = Text('Hello', t2c={'[1:-1]': BLUE}).move_to(LEFT)
            t2cwords = Text('World',t2c={'rl':RED}).next_to(t2cindices, RIGHT)
            self.add(t2cindices, t2cwords)

If you want avoid problems when colours(due to ligatures), consider using
:class:`MarkupText`.


Using Gradients
---------------

You can use Gradient using :attr:`~.Text.gradient`. The value must
be a Iterable of any Length.

For example,

.. manim:: GradientExample
    :save_last_frame:

    class GradientExample(Scene):
        def construct(self):
            t = Text("Hello", gradient=(RED, BLUE, GREEN)).scale(2)
            self.add(t)

You can also use :attr:`~.Text.t2g` for using gradients with specific 
characters of the Text. It has a very similar syntax like 
:ref:`Using Colors`.

For example,

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

You can set line spacing using :attr:`~.Text.line_spacing`.
For example,

.. manim:: LineSpacing
   :save_last_frame:

    class LineWidth(Scene):
        def construct(self):
            a = Text("Hello\nWorld", line_spacing=1)
            b = Text("Hello\nWorld", line_spacing=4)
            self.add(Group(a,b).arrange(LEFT, buff=5))


.. _disable-ligatures:

Disabling Ligatures
-------------------

By disabling ligatures you would get a 1-1 mapping between characters and
submobjects. This would fix coloring issue's. 


.. warning::

    Be aware that using this method with a text which heavily needs
    ligatures may not work as expected. For example, when disabling
    ligatures with Arabic text the output doesn't looks as expected.

You can disable ligatures by passing ``disable_ligatures`` parameter to 
:class:`Text`. For example,

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

Text objects behave like a VGroup-like iterable of all characters in the given
text. In particular, slicing is possible.

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
    one-one mapping of characters to submobjects you should use 
    ``disable_ligatures`` parameter in :class:`~.Text` while rendering.
    See :ref:`disable-ligatures`

.. _Ligature: https://en.wikipedia.org/wiki/Ligature_(writing)
