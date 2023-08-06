from __future__ import annotations

from manim import ManimBanner
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "logo"


@frames_comparison(last_frame=False)
def test_banner(scene):
    banner = ManimBanner()
    scene.play(banner.create(), run_time=0.5)
    scene.play(banner.expand(), run_time=0.5)
