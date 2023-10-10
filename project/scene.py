from manim import *
from pprint import pprint
# from utils import highlight_lines

class TextCreateScene(Scene):

    def construct(self):
        # text = Text("")
        text = MarkupText('')
        self.play(Create(text))
        # self.play(Create(text))
        # self.play(AddTextLetterByLetter(text))


class MarkupTextAddTextLetterByLetterScene(Scene):

    def construct(self):
        blank_markup_text = Text("")
        # blank_markup_text = MarkupText('<span bgcolor="#777777">Hello World</span>')
        self.play(AddTextLetterByLetter(blank_markup_text))
        


class ExampleScene(Scene):

    def construct(self):
        with open("scene.py", encoding="utf-8") as f:
            code = f.read()
        lines = highlight_lines(code)
        lines = lines[:7]
        print(repr(lines))
        for line in lines[:]:
            line = MarkupText(line,
                            font="Consolas",
                            font_size=16)
            line.align_to(ORIGIN, direction=LEFT)
            self.play(AddTextLetterByLetter(line))
            self.wait(1)
            self.clear()
        
        