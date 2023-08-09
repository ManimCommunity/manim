from manim import *


class T(Scene):
    def construct(self):
        banner = ManimBanner()
        self.play(banner.create(), run_time=0.5)
        self.play(banner.expand(), run_time=0.5)


if __name__ == "__main__":
    with tempconfig({}):
        T().render()
