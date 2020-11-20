"""Manim Banner utilities."""

__all__ = ["ManimBanner"]

from ..constants import LEFT, UP, RIGHT, DOWN, ORIGIN
from ..animation.composition import Succession, AnimationGroup
from ..animation.transform import ApplyMethod
from ..mobject.geometry import Circle, Square, Triangle
from ..mobject.svg.tex_mobject import Tex, MathTex
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.tex_templates import TexFontTemplates


class ManimBanner(VGroup):
    r"""Convenience class representing Manim's banner.

    Can be animated using the costum methods.

    Examples
    --------

    .. manim:: BannerExample

        from manim.utils.manim_banner import ManimBanner

        class BannerExample(Scene):
            def construct(self):
                banner = ManimBanner().scale(0.25).to_corner(DR)
                self.add(banner)
                self.wait()
                self.play(banner.expand())
                self.wait()

    """

    def __init__(self, dark_theme=True):
        VGroup.__init__(self)

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
        self.add(self.triangle, self.square, self.circle, self.M)
        self.move_to(ORIGIN)

        anim = VGroup()
        for i, ch in enumerate("anim"):
            tex = Tex(
                "\\textbf{" + ch + "}",
                tex_template=TexFontTemplates.gnu_freeserif_freesans,
            )
            if i != 0:
                tex.next_to(anim, buff=0.01)
            tex.align_to(ds_m, DOWN)
            anim.add(tex)
        anim.set_color(self.font_color).set_height(
            m_height_over_anim_height * ds_m.get_height()
        )

        self.anim = anim
        self.anim.set_opacity(0)

    def updater(self):
        self.shift(LEFT * 0.1)

    def scale(self, scale_factor, **kwargs):
        self.scale_factor *= scale_factor
        self.anim.scale(scale_factor, **kwargs)
        return super().scale(scale_factor, **kwargs)

    def expand(self):
        m_shape_offset = 5.7 * self.scale_factor
        m_anim_buff = 0.06
        self.anim.next_to(self.M, buff=m_anim_buff).shift(
            m_shape_offset * LEFT
        ).align_to(self.M, DOWN)
        move_left = AnimationGroup(
            ApplyMethod(self.triangle.shift, m_shape_offset * LEFT),
            ApplyMethod(self.square.shift, m_shape_offset * LEFT),
            ApplyMethod(self.circle.shift, m_shape_offset * LEFT),
            ApplyMethod(self.M.shift, m_shape_offset * LEFT),
            ApplyMethod(self.anim.set_opacity, 0),
        )
        move_right = AnimationGroup(
            ApplyMethod(self.triangle.shift, m_shape_offset * RIGHT),
            ApplyMethod(self.square.shift, m_shape_offset * RIGHT),
            ApplyMethod(self.circle.shift, m_shape_offset * RIGHT),
            ApplyMethod(self.M.shift, 0 * LEFT),
            AnimationGroup(
                *[ApplyMethod(obj.set_opacity, 1) for obj in self.anim], lag_ratio=0.15
            )
            # for whatever reason, FadeIn(self.anim, lag_ratio=1) does the weirdest stuff
        )
        return Succession(move_left, move_right)
