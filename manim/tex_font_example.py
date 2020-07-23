from manim import (
    BLACK, WHITE, UP, UL, ORIGIN, DOWN, LEFT, RIGHT, TextMobject,
    ApplyMethod, AnimationGroup, VGroup, FadeIn, FadeOut, Transform,
    FadeOutAndShift, FadeInFromDown, FadeOutAndShiftDown, Scene
    )
from manim.tex_font_templates import (TexTemplateFromFontConfig, TEX_FONT_CONFIGS, FontMobject)

TEXT_COLOUR = WHITE
BACKGROUND_COLOUR = BLACK


class simple(Scene):
    def makelabel(self, font):
        return FontMobject(
            "FontMobject(text, font=", font+ ")").to_edge(UL)
    def maketext(self, font):
        return FontMobject(
            TEX_FONT_CONFIGS[font]["description"],
            font=font
            )
    def construct(self):
        self.label = TextMobject("Tex Font Profile Showcase").to_edge(UL)
        first = True
        font = "font"
        self.text = TextMobject("Tex font Sample")
        self.add(self.label)
        self.add(self.text)
        self.wait(1)
        for font in TEX_FONT_CONFIGS:
            self.nexttext = self.maketext(font)
            transform = Transform(self.label, self.makelabel(font)) if first is True else \
                        Transform(self.label[1], self.makelabel(font)[1])
            self.play(FadeOutAndShift(self.text, direction=UP),
                      transform,
                      FadeInFromDown(self.nexttext)
                      )
            self.text = self.nexttext
            first = False
            self.wait(0.5)
        self.wait(2)
        self.nexttext = TextMobject("Thanks to\\\\ http://jf.burnol.free.fr/showcase.html")
        self.play(FadeOutAndShift(self.text, direction=UP),
                  FadeOut(self.label),
                  FadeInFromDown(self.nexttext))
        self.text = self.nexttext
        self.wait(4)
        self.nexttext = FontMobject("Code at \\\\gitlab.com/co-bordism/manim/-/tree/all\_tex\_fonts\_community", font="ecfaugieeg")
        self.play(FadeOutAndShift(self.text, direction=UP),
                  FadeInFromDown(self.nexttext))
        self.wait(6)
