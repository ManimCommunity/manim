"""Scene whose timeline is stored in timeline-v1.json, the format reference.

It makes every kind of call that a timeline records. Line numbers matter: after
editing this file, regenerate timeline-v1.json.
"""

from manim import RIGHT, Scene, Square


class TimelineFormat(Scene):
    def construct(self):
        self.next_section("opening")
        self.play(Square().animate.shift(RIGHT), run_time=0.3, subcaption="move")
        for _ in range(2):
            self.wait(0.3, frozen_frame=False)
        self.wait(0.3, frozen_frame=True)
        start = self.time
        self.wait(1, stop_condition=lambda: self.time >= start + 0.5)
        self.add_sound("click.wav")
