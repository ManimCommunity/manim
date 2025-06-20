from __future__ import annotations

from manim import BLUE, GREEN, RED, WHITE, DashedLine, Line, Scene
from manim.constants import DOWN, LEFT, RIGHT, UP


class GradientDemo(Scene):
    """
    Quick visual sanity-check: one solid and one dashed line showing that
    the first colour in the list is always mapped to the start anchor.
    """

    def construct(self):
        # ── Solid 2-stop gradient ───────────────────────────────────────────────
        solid = Line(LEFT * 4 + UP * 2, RIGHT * 4 + UP * 2, stroke_width=24).set_stroke(
            [BLUE, RED]
        )

        # ── Dashed 2-stop gradient (let Manim pick dash count) ─────────────────
        dashed = DashedLine(
            LEFT * 4 + DOWN * 2,
            RIGHT * 4 + DOWN * 2,
            stroke_width=24,
        ).set_stroke([GREEN, WHITE])

        self.add(solid, dashed)
