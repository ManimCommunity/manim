"""Metadata-only example: manim --fps 4 --timeline-output timeline.json timeline_scene.py TimelineExample."""

from pathlib import Path

from manim import RIGHT, Scene, Square

CLICK = Path(__file__).parent / "assets" / "click.wav"


class TimelineExample(Scene):
    def construct(self):
        self.next_section("opening")
        self.play(Square().animate.shift(RIGHT), run_time=0.3, subcaption="move")
        for _ in range(2):
            self.wait(0.3, frozen_frame=False)
        self.wait(0.3, frozen_frame=True)
        start = self.time
        self.wait(1, stop_condition=lambda: self.time >= start + 0.5)
        # Recorded with its duration; the file is not decoded.
        self.add_sound(CLICK)
