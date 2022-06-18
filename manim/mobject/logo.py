"""Utilities for Manim's logo and banner."""

from __future__ import annotations

__all__ = ["ManimBanner"]

from manim.animation.updaters.update import UpdateFromAlphaFunc
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square, Triangle
from manim.mobject.text.tex_mobject import MathTex, Tex

from ..animation.animation import override_animation
from ..animation.composition import AnimationGroup, Succession
from ..animation.creation import Create, SpiralIn
from ..animation.fading import FadeIn
from ..constants import DOWN, LEFT, ORIGIN, RIGHT, TAU, UP
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.rate_functions import ease_in_out_cubic, ease_out_sine, smooth
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
    .. manim:: DarkThemeBanner

        class DarkThemeBanner(Scene):
            def construct(self):
                banner = ManimBanner()
                self.play(banner.create())
                self.play(banner.expand())
                self.wait()
                self.play(Unwrite(banner))

    .. manim:: LightThemeBanner

        class LightThemeBanner(Scene):
            def construct(self):
                self.camera.background_color = "#ece6e2"
                banner = ManimBanner(dark_theme=False)
                self.play(banner.create())
                self.play(banner.expand())
                self.wait()
                self.play(Unwrite(banner))

    """

    def __init__(self, dark_theme: bool = True):
        super().__init__()

        logo_green = "#81b29a"
        logo_blue = "#454866"
        logo_red = "#e07a5f"
        m_height_over_anim_height = 0.75748

        self.font_color = "#ece6e2" if dark_theme else "#343434"
        self.scale_factor = 1

        self.M = MathTex(r"\mathbb{M}").scale(7).set_color(self.font_color)
        self.M.shift(2.25 * LEFT + 1.5 * UP)

        self.circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
        self.square = Square(color=logo_blue, fill_opacity=1).shift(UP)
        self.triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
        self.shapes = VGroup(self.triangle, self.square, self.circle)
        self.add(self.shapes, self.M)
        self.move_to(ORIGIN)

        anim = VGroup()
        for i, ch in enumerate("anim"):
            tex = Tex(
                "\\textbf{" + ch + "}",
                tex_template=TexFontTemplates.gnu_freeserif_freesans,
            )
            if i != 0:
                tex.next_to(anim, buff=0.01)
            tex.align_to(self.M, DOWN)
            anim.add(tex)
        anim.set_color(self.font_color)
        anim.height = m_height_over_anim_height * self.M.height

        # Note: "anim" is only shown in the expanded state
        # and thus not yet added to the submobjects of self.
        self.anim = anim

    def scale(self, scale_factor: float, **kwargs) -> ManimBanner:
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

    @override_animation(Create)
    def create(self, run_time: float = 2) -> AnimationGroup:
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
        return AnimationGroup(
            SpiralIn(self.shapes, run_time=run_time),
            FadeIn(self.M, run_time=run_time / 2),
            lag_ratio=0.1,
        )

    def expand(self, run_time: float = 1.5, direction="center") -> Succession:
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

        Parameters
        ----------
        run_time
            The run time of the animation.
        direction
            The direction in which the logo is expanded.

        Returns
        -------
        :class:`~.Succession`
            An animation to be used in a :meth:`.Scene.play` call.

        Examples
        --------
        .. manim:: ExpandDirections

            class ExpandDirections(Scene):
                def construct(self):
                    banners = [ManimBanner().scale(0.5).shift(UP*x) for x in [-2, 0, 2]]
                    self.play(
                        banners[0].expand(direction="right"),
                        banners[1].expand(direction="center"),
                        banners[2].expand(direction="left"),
                    )

        """
        if direction not in ["left", "right", "center"]:
            raise ValueError("direction must be 'left', 'right' or 'center'.")

        m_shape_offset = 6.25 * self.scale_factor
        shape_sliding_overshoot = self.scale_factor * 0.8
        m_anim_buff = 0.06
        self.anim.next_to(self.M, buff=m_anim_buff).align_to(self.M, DOWN)
        self.anim.set_opacity(0)
        self.shapes.save_state()
        m_clone = self.anim[-1].copy()
        self.add(m_clone)
        m_clone.move_to(self.shapes)

        self.M.save_state()
        left_group = VGroup(self.M, self.anim, m_clone)

        def shift(vector):
            self.shapes.restore()
            left_group.align_to(self.M.saved_state, LEFT)
            if direction == "right":
                self.shapes.shift(vector)
            elif direction == "center":
                self.shapes.shift(vector / 2)
                left_group.shift(-vector / 2)
            elif direction == "left":
                left_group.shift(-vector)

        def slide_and_uncover(mob, alpha):
            shift(alpha * (m_shape_offset + shape_sliding_overshoot) * RIGHT)

            # Add letters when they are covered
            for letter in mob.anim:
                if mob.square.get_center()[0] > letter.get_center()[0]:
                    letter.set_opacity(1)
                    self.add(letter)

            # Finish animation
            if alpha == 1:
                self.remove(*[self.anim])
                self.add_to_back(self.anim)
                mob.shapes.set_z_index(0)
                mob.shapes.save_state()
                mob.M.save_state()

        def slide_back(mob, alpha):
            if alpha == 0:
                m_clone.set_opacity(1)
                m_clone.move_to(mob.anim[-1])
                mob.anim.set_opacity(1)

            shift(alpha * shape_sliding_overshoot * LEFT)

            if alpha == 1:
                mob.remove(m_clone)
                mob.add_to_back(mob.shapes)

        return Succession(
            UpdateFromAlphaFunc(
                self,
                slide_and_uncover,
                run_time=run_time * 2 / 3,
                rate_func=ease_in_out_cubic,
            ),
            UpdateFromAlphaFunc(
                self,
                slide_back,
                run_time=run_time * 1 / 3,
                rate_func=smooth,
            ),
        )
