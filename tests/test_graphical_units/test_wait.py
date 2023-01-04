from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "wait"


@frames_comparison(last_frame=False)
def test_WaitPause_DoesNotDrawMobjectsTwice(scene):
    scene.camera.background_color = WHITE

    # These boxes should be the same color, all throughout the animation.
    # If they look different, it could mean the mobjects are being drawn
    # multiple times.
    grey1 = Square(fill_color="#888888", fill_opacity=1.0)
    grey2 = Square(fill_color="#000000", fill_opacity=0.5)

    grey2.next_to(grey1, RIGHT)

    scene.add(grey1, grey2)
    scene.pause()

    text = Text("Animating", font_size=18, color=BLACK).scale(2)
    text.move_to([0, 3, 0])
    scene.play(FadeIn(text), run_time=0.5)

    scene.remove(text)
    scene.wait(0.5)
    scene.add(text)

    scene.play(FadeOut(text), run_time=0.5)

    scene.pause()
