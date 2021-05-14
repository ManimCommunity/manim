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
----------------------------------

The simplest way to add text to your animations is to use the :class:`~.Text`
class. It uses the `Pango library`_ to render text. With Pango, you can also
render non-English alphabets like `你好` or  `こんにちは` or `안녕하세요` or
`مرحبا بالعالم`.

Here, is a simple Hello World animation.

.. manim:: HelloWorld

    class HelloWorld(Scene):
        def construct(self):
            text = Text('Hello world').scale(3)
            self.add(FadeIn(text))

You can also use :class:`~.MarkupText` where PangoMarkup (:ref:`pangomarkup`)
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
==========================

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




.. _Ligature: https://en.wikipedia.org/wiki/Ligature_(writing)

.. _pangomarkup:

What is PangoMarkup?
====================

PangoMarkup is a small markup language like html and it helps you avoid using
"range of characters" while colouring or styling a piece a Text. You can use
this language with :class:`~.MarkupText`.

A simple example of a marked-up string might be::

    <span foreground="blue" size="x-large">Blue text</span> is <i>cool</i>!"

and it can be used with :class:`~.MarkupText` as 

.. manim:: MarkupExample
    :save_last_frame:

    class MarkupExample(Scene):
        def construct(self):
            text = MarkupText('<span foreground="blue" size="x-large">Blue text</span> is <i>cool</i>!"')
            self.add(text)

A more elaborate example would be

.. manim:: MarkupElaborateExample
    :save_last_frame:

    class MarkupElaborateExample(Scene):
        def construct(self):
            text = MarkupText('<span foreground="purple">ا</span><span foreground="red">َ</span>ل<span foreground="blue">ْ</span>ع<span foreground="red">َ</span>ر<span foreground="red">َ</span>ب<span foreground="red">ِ</span>ي<span foreground="green">ّ</span><span foreground="red">َ</span>ة<span foreground="blue">ُ</span>')
            self.add(text)

PangoMarkup can also contain XML features such as numeric character
entities such as ``&#169;`` for © can be used too.

The most general markup tag is ``<span>``, then there are some 
convenience tags.

Here is a list of supported tags:

- ``<b>bold</b>``, ``<i>italic</i>`` and ``<b><i>bold+italic</i></b>``
- ``<ul>underline</ul>`` and ``<s>strike through</s>``
- ``<tt>typewriter font</tt>``
- ``<big>bigger font</big>`` and ``<small>smaller font</small>``
- ``<sup>superscript</sup>`` and ``<sub>subscript</sub>``
- ``<span underline="double" underline_color="green">double underline</span>``
- ``<span underline="error">error underline</span>``
- ``<span overline="single" overline_color="green">overline</span>``
- ``<span strikethrough="true" strikethrough_color="red">strikethrough</span>``
- ``<span font_family="sans">temporary change of font</span>``
- ``<span foreground="red">temporary change of color</span>``
- ``<span fgcolor="red">temporary change of color</span>``
- ``<gradient from="YELLOW" to="RED">temporary gradient</gradient>``

For ``<span>`` markup, colors can be specified either as 
hex triples like ``#aabbcc`` or as named CSS colors like 
``AliceBlue``.
The ``<gradient>`` tag being handled by Manim rather than 
Pango, supports hex triplets or Manim constants like 
``RED`` or ``RED_A``.
If you want to use Manim constants like ``RED_A`` together 
with ``<span>``, you will need to use Python's f-String 
syntax as follows::
    
    MarkupText(f'<span foreground="{RED_A}">here you go</span>')

If your text contains ligatures, the :class:`MarkupText` class may 
incorrectly determine the first and last letter when creating the 
gradient. This is due to the fact that e.g. ``fl`` are two characters,
but might be set as one single glyph, a ligature. If your language 
does not depend on ligatures, consider setting ``disable_ligatures``
to ``True`` of :class:`~.MarkupText`. If you cannot or do not want
to do without ligatures, the ``gradient`` tag supports an optional
attribute ``offset`` which can be used to compensate for that error.
Usage is as follows:

- ``<gradient from="RED" to="YELLOW" offset="1">example</gradient>`` to *start* the gradient one letter earlier
- ``<gradient from="RED" to="YELLOW" offset=",1">example</gradient>`` to *end* the gradient one letter earlier
- ``<gradient from="RED" to="YELLOW" offset="2,1">example</gradient>`` to *start* the gradient two letters earlier and *end* it one letter earlier

Specifying a second offset may be necessary if the text to be colored does
itself contain ligatures. The same can happen when using HTML entities for
special chars.

When using ``underline``, ``overline`` or ``strikethrough`` together with 
``<gradient>`` tags, you will also need to use the offset, because
underlines are additional paths in the final :class:`SVGMobject`, 
check out the corresponding example.

Escaping of special characters: ``>`` **should** be written as ``&gt;`` 
whereas ``<`` and ``&`` *must* be written as ``&lt;`` and 
``&amp;``.

You can find more information about Pango markup formatting at the
corresponding documentation page:
`Pango Markup <https://developer.gnome.org/pango/stable/pango-Markup.html>`_.
Please be aware that not all features are supported by this class and that
the ``<gradient>`` tag mentioned above is not supported by Pango.

