from __future__ import annotations

from manim import *


class MyScene(Scene):
    def construct(self):
        t = ValueTracker(0)
        for y in range(2, -3, -2):
            for x in range(-4, 5, 2):
                updater = (
                    lambda x, y: (
                        lambda s: s.move_to(
                            [
                                x + np.cos(t.get_value()),
                                y + (-1 if x in (-2, 2) else 1) * np.sin(t.get_value()),
                                0,
                            ]
                        )
                    )
                )(x, y)
                self.add(Square(1).add_updater(updater))
        self.play(t.animate.set_value(2 * TAU), run_time=10)


with tempconfig({"preview": False, "disable_caching": True}):
    MyScene().render()
