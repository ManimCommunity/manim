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

    class HelloWorld(Scene):
        def construct(self):
            text = Text('Hello world').scale(3)
            self.play(FadeIn(text))

You can also use :class:`~.MarkupText` where PangoMarkup (:class:`MarkupText`)
can be used to render text. For example,

.. manim:: SingleLineColour
    :save_last_frame:

    class SingleLineColour(Scene):
        def construct(self):
            text = MarkupText(
               f'all in red <span fgcolor="{YELLOW}">except this</span>',
               color=RED
            )
            self.add(text)

.. _Pango library: https://pango.gnome.org

Working with :class:`~.Text`
===========================

This section explains the properties of :class:`~.Text` and how can it be used
in your Animations.

Iterating :class:`~.Text`
-------------------------

Text objects behave like a VGroup-like iterable of all characters in the given
text. In particular, slicing is possible.

For example, you can set each letter to different colour by iterating it.

.. manim:: IterateColor

    class IterateColor(Scene):
        def construct(self):
            text = Text("Colours").scale(2)
            for letter in text:
                letter.set_color(random_bright_color())
                self.play(Write(letter))

.. warning::

    Please note that `Ligature`_ can cause problems here. If you need a
    one-one mapping of characters to submobjects you should use 
    ``disable_ligatures`` parameter in :class:`~.Text` while rendering.
    See :ref:`disable-ligatures`

.. _Ligature: https://en.wikipedia.org/wiki/Ligature_(writing)

Using Gradients
---------------

You can use Gradient using the :attr:`~.Text.gradient`. The value must
be a Iterable or any Length.

For example,

.. manim:: GradientExample
    :save_last_frame:

    class GradientExample(Scene):
        def construct(self):
            t = Text("Hello", gradient=(RED, BLUE, GREEN)).scale(2)
            self.add(t)


.. _disable-ligatures

Disabling Ligatures
-------------------

By disabling ligatures you would get a 1-1 mapping between characters and
submobjects. This would fix colouring issue's. 


.. warning::

    Be aware that using this method with a text which heavily needs ligatures
    may not work as expected. For example, when disabling ligatures with Arabic
    text the output doesn't looks as expected.

You can disable ligatures by passing ``disable_ligatures`` parameter to 
:class:`Text`. For example,

.. manim:: DisableLigature

    class DisableLigature(Scene):
        def construct(self):
            li = Text("fl ligature").scale(2)
            nli = Text("fl ligature", disable_ligatures=True).scale(2)
            self.add(Group(li, nli).arrange(DOWN, buff=.8))
