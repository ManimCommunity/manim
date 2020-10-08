Text
=================================

.. manim:: Example1Text
    :save_last_frame:

    class Example1Text(Scene):
        def construct(self):
            text = Text('Hello world').scale(3)
            self.add(text)


`Text` works also with other languages like `你好` or  `こんにちは` or `안녕하세요` or `مرحبا بالعالم`.
Be sure you have the font that supports those languages!

