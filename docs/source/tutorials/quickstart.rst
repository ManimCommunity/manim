from manim import *

class PillIntro(Scene):
    def construct(self):
        # एक कैप्सूल जैसी आकृति बनाना (Pill)
        pill = Capsule(color=RED, fill_opacity=0.8).scale(1.5)
        text = Text("The Human Body & Drug Work", font_size=24).next_to(pill, DOWN)
        
        # एनीमेशन: गोली का आना और नाम का लिखना
        self.play(Create(pill))
        self.play(Write(text))
        self.wait(1)
        
        # एनीमेशन: गोली का घूमना (जैसे शरीर में काम कर रही हो)
        self.play(pill.animate.rotate(PI/2).set_color(BLUE))
        self.wait(1)
        
        # सब कुछ गायब होना
        self.play(FadeOut(pill), FadeOut(text))

