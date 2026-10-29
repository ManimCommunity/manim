"""Scene for trying out timeline export.

manim --fps 4 --timeline-output timeline.json example_scenes/timeline_scene.py TimelineExample
"""

from manim import RIGHT, Scene, Square


class TimelineExample(Scene):
    def construct(self):
        self.next_section("opening")
        self.play(Square().animate.shift(RIGHT), run_time=0.3, subcaption="move")
        for _ in range(2):
            self.wait(0.3, frozen_frame=False)
        self.wait(0.3, frozen_frame=True)
        start = self.time
        self.wait(1, stop_condition=lambda: self.time >= start + 0.5)
        # Only recorded in the timeline; example.wav is a placeholder.
        self.add_sound("example.wav")
