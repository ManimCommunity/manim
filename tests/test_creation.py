from manim import *

class TestAddWordByWord(Scene): 
    def construct(self): 
        text = Text('Hello world')
        self.play(AddTextWordByWord(text))

class TestAddLetterByLetter(Scene): 
    def construct(self):
        text = Text('Hello World')
        self.play(AddTextLetterByLetter(text))


def test_scenes(): 
    TestAddWordByWord()
    TestAddLetterByLetter()

