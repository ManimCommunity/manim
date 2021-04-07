"""Utilities for Manim's logo and banner."""

__all__ = ["ManimBanner"]

from manim.utils.color import RED
from manim.animation.animation import Animation
import numpy as np

from ..animation.update import UpdateFromAlphaFunc
from ..animation.composition import AnimationGroup, Succession
from ..animation.fading import FadeIn
from ..animation.transform import ApplyMethod
from ..constants import DOWN, LEFT, ORIGIN, RIGHT, TAU, UP
from ..mobject.geometry import Circle, Square, Triangle
from ..mobject.svg.tex_mobject import MathTex, Tex
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.rate_functions import ease_out_sine
from ..utils.tex_templates import TexFontTemplates


class ManimBanner(VGroup):
    r"""Convenience class representing Manim's banner.

    Can be animated using custom methods.

    Parameters
    ----------
    dark_theme
        If ``True`` (the default), the dark theme version of the logo
        (with light text font) will be rendered. Otherwise, if ``False``,
        the light theme version (with dark text font) is used.

    Examples
    --------

    .. manim:: BannerDarkBackground

        class BannerDarkBackground(Scene):
            def construct(self):
                banner = ManimBanner().scale(0.5).to_corner(DR)
                self.play(FadeIn(banner))
                self.play(banner.expand())
                self.play(FadeOut(banner))

    .. manim:: BannerLightBackground

        class BannerLightBackground(Scene):
            def construct(self):
                self.camera.background_color = "#ece6e2"
                banner_large = ManimBanner(dark_theme=False).scale(0.7)
                banner_small = ManimBanner(dark_theme=False).scale(0.35)
                banner_small.next_to(banner_large, DOWN)
                self.play(banner_large.create(), banner_small.create())
                self.play(banner_large.expand(), banner_small.expand())
                self.play(FadeOut(banner_large), FadeOut(banner_small))

    """

    def __init__(self, dark_theme: bool = True):
        VGroup.__init__(self)

        logo_green = "#81b29a"
        logo_blue = "#454866"
        logo_red = "#e07a5f"
        m_height_over_anim_height = 0.53385

        self.font_color = "#ece6e2" if dark_theme else "#343434"
        self.scale_factor = 1

        self.M = MathTex(r"\mathbb{M}").scale(7).set_color(self.font_color)
        self.M.shift(2.25 * LEFT + 1.5 * UP)

        self.circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
        self.square = Square(color=logo_blue, fill_opacity=1).shift(UP)
        self.triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
        self.shape = VGroup(self.triangle, self.square, self.circle)
        self.add(self.shape, self.M)
        self.move_to(ORIGIN)

        anim = VGroup()
        for i, ch in enumerate(["a","n","\\i","m"]):
            tex = Tex(
                "\\textbf{" + ch + "}",
                # ch,
                tex_template=TexFontTemplates.gnu_freeserif_freesans,
            )
            if i != 0:
                tex.next_to(anim, buff=0.01)
            tex.align_to(self.M, DOWN)
            anim.add(tex)
        anim.set_color(self.font_color)
        print(anim.height)
        anim.height = m_height_over_anim_height * self.M.height

        # Note: "anim" is only shown in the expanded state
        # and thus not yet added to the submobjects of self.
        self.anim = anim

    def scale(self, scale_factor: float, **kwargs) -> "ManimBanner":
        """Scale the banner by the specified scale factor.

        Parameters
        ----------
        scale_factor
            The factor used for scaling the banner.

        Returns
        -------
        :class:`~.ManimBanner`
            The scaled banner.
        """
        self.scale_factor *= scale_factor
        # Note: self.anim is only added to self after expand()
        if self.anim not in self.submobjects:
            self.anim.scale(scale_factor, **kwargs)
        return super().scale(scale_factor, **kwargs)

    def create(self, run_time: float = 2.1):
        """The creation animation for Manim's logo.

        Parameters
        ----------
        run_time
            The run time of the animation.

        Returns
        -------
        :class:`~.AnimationGroup`
            An animation to be used in a :meth:`.Scene.play` call.
        """
        shape_center = self.shape.get_center()
        expansion_factor = 8 * self.scale_factor

        for shape in self.shape:
            shape.final_position = shape.get_center()
            shape.initial_position = (
                shape.final_position
                + (shape.final_position - shape_center) * expansion_factor
            )
            shape.move_to(shape.initial_position)
            shape.save_state()

        def spiral_updater(shapes, alpha):
            for shape in shapes:
                shape.restore()
                shape.shift((shape.final_position - shape.initial_position) * alpha)
                shape.rotate(TAU * alpha, about_point=shape_center)
                shape.rotate(-TAU * alpha, about_point=shape.get_center_of_mass())
                shape.set_opacity(min(1, alpha*3))

        return AnimationGroup(
            UpdateFromAlphaFunc(self.shape, spiral_updater, run_time=run_time, rate_func=ease_out_sine),
            FadeIn(self.M, run_time=run_time / 2),
            lag_ratio=0.1
        )

    def expand(self, direction=RIGHT) -> Succession:
        """An animation that expands Manim's logo into its banner.

        The returned animation transforms the banner from its initial
        state (representing Manim's logo with just the icons) to its
        expanded state (showing the full name together with the icons).

        See the class documentation for how to use this.

        .. note::

            Before calling this method, the text "anim" is not a
            submobject of the banner object. After the expansion,
            it is added as a submobject so subsequent animations
            to the banner object apply to the text "anim" as well.

        Returns
        -------
        :class:`~.Succession`
            An animation to be used in a :meth:`.Scene.play` call.

        """
        m_shape_offset = 6.25 * self.scale_factor
        m_anim_buff = 0.06
        self.add(self.anim)
        self.anim.next_to(self.M, buff=m_anim_buff)\
                 .align_to(self.M, DOWN)\
                #  .shift(m_shape_offset * LEFT)\
        self.anim.set_opacity(0)
        print(self.anim[2].get_subpaths())



        def slide_and_uncover(shape, alpha):
            shape.restore()
            shape.shift(alpha * m_shape_offset * RIGHT)
            for letter in self.anim:
                if self.square.get_center()[0] > letter.get_center()[0]:
                    letter.set_opacity(1)

        self.shape.save_state()
        return AnimationGroup(
            Animation(self.anim, run_time=0),
            UpdateFromAlphaFunc(self.shape, slide_and_uncover, run_time=2),
            self.M.animate.scale(1),
        )
