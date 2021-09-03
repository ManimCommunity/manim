"""Animation of a mobject boundary and tracing of points."""

__all__ = ["AnimatedBoundary", "TracedPath"]

from typing import Callable, Optional

from colour import Color

from .._config import config
from ..constants import *
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import BLUE_B, BLUE_D, BLUE_E, GREY_BROWN, WHITE
from ..utils.deprecation import deprecated_params
from ..utils.rate_functions import smooth
from .opengl_compatibility import ConvertToOpenGL


class AnimatedBoundary(VGroup):
    """Boundary of a :class:`.VMobject` with animated color change.

    Examples
    --------
    .. manim:: AnimatedBoundaryExample

        class AnimatedBoundaryExample(Scene):
            def construct(self):
                text = Text("So shiny!")
                boundary = AnimatedBoundary(text, colors=[RED, GREEN, BLUE],
                                            cycle_rate=3)
                self.add(text, boundary)
                self.wait(2)

    """

    def __init__(
        self,
        vmobject,
        colors=[BLUE_D, BLUE_B, BLUE_E, GREY_BROWN],
        max_stroke_width=3,
        cycle_rate=0.5,
        back_and_forth=True,
        draw_rate_func=smooth,
        fade_rate_func=smooth,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.colors = colors
        self.max_stroke_width = max_stroke_width
        self.cycle_rate = cycle_rate
        self.back_and_forth = back_and_forth
        self.draw_rate_func = draw_rate_func
        self.fade_rate_func = fade_rate_func
        self.vmobject = vmobject
        self.boundary_copies = [
            vmobject.copy().set_style(stroke_width=0, fill_opacity=0) for x in range(2)
        ]
        self.add(*self.boundary_copies)
        self.total_time = 0
        self.add_updater(lambda m, dt: self.update_boundary_copies(dt))

    def update_boundary_copies(self, dt):
        # Not actual time, but something which passes at
        # an altered rate to make the implementation below
        # cleaner
        time = self.total_time * self.cycle_rate
        growing, fading = self.boundary_copies
        colors = self.colors
        msw = self.max_stroke_width
        vmobject = self.vmobject

        index = int(time % len(colors))
        alpha = time % 1
        draw_alpha = self.draw_rate_func(alpha)
        fade_alpha = self.fade_rate_func(alpha)

        if self.back_and_forth and int(time) % 2 == 1:
            bounds = (1 - draw_alpha, 1)
        else:
            bounds = (0, draw_alpha)
        self.full_family_become_partial(growing, vmobject, *bounds)
        growing.set_stroke(colors[index], width=msw)

        if time >= 1:
            self.full_family_become_partial(fading, vmobject, 0, 1)
            fading.set_stroke(color=colors[index - 1], width=(1 - fade_alpha) * msw)

        self.total_time += dt

    def full_family_become_partial(self, mob1, mob2, a, b):
        family1 = mob1.family_members_with_points()
        family2 = mob2.family_members_with_points()
        for sm1, sm2 in zip(family1, family2):
            sm1.pointwise_become_partial(sm2, a, b)
        return self


class TracedPath(VMobject, metaclass=ConvertToOpenGL):
    """Traces the path of a point returned by a function call.

    Parameters
    ----------
    traced_point_func
        The function to be traced.
    stroke_width
        The width of the trace.
    stroke_color
        The color of the trace.
    dissipating_time
        The time taken for the path to dissipate. Default set to ``None``
        which disables dissipation.

    Examples
    --------
    .. manim:: TracedPathExample

        class TracedPathExample(Scene):
            def construct(self):
                circ = Circle(color=RED).shift(4*LEFT)
                dot = Dot(color=RED).move_to(circ.get_start())
                rolling_circle = VGroup(circ, dot)
                trace = TracedPath(circ.get_start)
                rolling_circle.add_updater(lambda m: m.rotate(-0.3))
                self.add(trace, rolling_circle)
                self.play(rolling_circle.animate.shift(8*RIGHT), run_time=4, rate_func=linear)

    .. manim:: DissipatingPathExample

        class DissipatingPathExample(Scene):
            def construct(self):
                a = Dot(RIGHT * 2)
                b = TracedPath(a.get_center, dissipating_time=0.5, stroke_opacity=[0, 1])
                self.add(a, b)
                self.play(a.animate(path_arc=PI / 4).shift(LEFT * 2))
                self.play(a.animate(path_arc=-PI / 4).shift(LEFT * 2))
                self.wait()

    """

    @deprecated_params(
        params="min_distance_to_new_point",
        since="v0.10.0",
        until="v0.12.0",
    )
    def __init__(
        self,
        traced_point_func: Callable,
        stroke_width: float = 2,
        stroke_color: Color = WHITE,
        dissipating_time: Optional[float] = None,
        **kwargs
    ):
        kwargs.pop("min_distance_to_new_point", None)  #
        super().__init__(stroke_color=stroke_color, stroke_width=stroke_width, **kwargs)
        self.traced_point_func = traced_point_func
        self.dissipating_time = dissipating_time
        self.time = 1 if self.dissipating_time else None
        self.add_updater(self.update_path)

    def update_path(self, mob, dt):
        new_point = self.traced_point_func()
        if not self.has_points():
            self.start_new_path(new_point)
        self.add_line_to(new_point)
        if self.dissipating_time:
            self.time += dt
            if self.time - 1 > self.dissipating_time:
                if config["renderer"] == "opengl":
                    nppcc = self.n_points_per_curve
                else:
                    nppcc = self.n_points_per_cubic_curve
                self.set_points(self.points[nppcc:])
