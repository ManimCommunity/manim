from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "speed"


@frames_comparison(last_frame=False)
def test_SpeedModifier(scene):
    a = Dot().shift(LEFT * 2 + 0.5 * UP)
    b = Dot().shift(LEFT * 2 + 0.5 * DOWN)
    c = Dot().shift(2 * RIGHT)
    ChangeSpeed.add_updater(c, lambda x, dt: x.rotate_about_origin(PI / 3.7 * dt))
    scene.add(a, b, c)
    scene.play(ChangeSpeed(Wait(0.5), speedinfo={0.3: 1, 0.4: 0.1, 0.6: 0.1, 1: 1}))
    scene.play(
        ChangeSpeed(
            AnimationGroup(
                a.animate(run_time=0.5, rate_func=linear).shift(RIGHT * 4),
                b.animate(run_time=0.5, rate_func=rush_from).shift(RIGHT * 4),
            ),
            speedinfo={0.3: 1, 0.4: 0.1, 0.6: 0.1, 1: 1},
            affects_speed_updaters=False,
        ),
    )
    scene.play(
        ChangeSpeed(
            AnimationGroup(
                a.animate(run_time=0.5, rate_func=linear).shift(LEFT * 4),
                b.animate(run_time=0.5, rate_func=rush_into).shift(LEFT * 4),
            ),
            speedinfo={0.3: 1, 0.4: 0.1, 0.6: 0.1, 1: 1},
            rate_func=there_and_back,
        ),
    )
