"""Utilities for Manim's logo and banner."""

from __future__ import annotations

__all__ = ["ManimBanner"]

import svgelements as se

from manim.animation.updaters.update import UpdateFromAlphaFunc
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square, Triangle

from .. import constants as cst
from ..animation.animation import override_animation
from ..animation.composition import AnimationGroup, Succession
from ..animation.creation import Create, SpiralIn
from ..animation.fading import FadeIn
from ..mobject.svg.svg_mobject import VMobjectFromSVGPath
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.rate_functions import ease_in_out_cubic, smooth

MANIM_SVG_PATHS: list[se.Path] = [
    se.Path(  # double stroke letter M
        "M4.64259-2.092154L2.739726-6.625156C2.660025-6.824408 2.650062-6.824408 "
        "2.381071-6.824408H.52802C.348692-6.824408 .199253-6.824408 .199253-6.645"
        "081C.199253-6.475716 .37858-6.475716 .428394-6.475716C.547945-6.475716 ."
        "816936-6.455791 1.036115-6.37609V-1.05604C1.036115-.846824 1.036115-.408"
        "468 .358655-.348692C.169365-.328767 .169365-.18929 .169365-.179328C.1693"
        "65 0 .328767 0 .508095 0H2.052304C2.231631 0 2.381071 0 2.381071-.179328"
        "C2.381071-.268991 2.30137-.33873 2.221669-.348692C1.454545-.408468 1.454"
        "545-.826899 1.454545-1.05604V-6.017435L1.464508-6.027397L3.895392-.20921"
        "5C3.975093-.029888 4.044832 0 4.104608 0C4.224159 0 4.254047-.079701 4.3"
        "03861-.199253L6.744707-6.027397L6.75467-6.017435V-1.05604C6.75467-.84682"
        "4 6.75467-.408468 6.07721-.348692C5.88792-.328767 5.88792-.18929 5.88792"
        "-.179328C5.88792 0 6.047323 0 6.22665 0H8.886675C9.066002 0 9.215442 0 9"
        ".215442-.179328C9.215442-.268991 9.135741-.33873 9.05604-.348692C8.28891"
        "7-.408468 8.288917-.826899 8.288917-1.05604V-5.768369C8.288917-5.977584 "
        "8.288917-6.41594 8.966376-6.475716C9.066002-6.485679 9.155666-6.535492 9"
        ".155666-6.645081C9.155666-6.824408 9.006227-6.824408 8.826899-6.824408H6"
        ".90411C6.645081-6.824408 6.625156-6.824408 6.535492-6.615193L4.64259-2.0"
        "92154ZM4.343711-1.912827C4.423412-1.743462 4.433375-1.733499 4.552927-1."
        "693649L4.11457-.637609H4.094645L1.823163-6.057285C1.77335-6.1868 1.69364"
        "9-6.356164 1.554172-6.475716H2.420922L4.343711-1.912827ZM1.334994-.34869"
        "2H1.165629C1.185554-.37858 1.205479-.408468 1.225405-.428394C1.235367-.4"
        "38356 1.235367-.448319 1.24533-.458281L1.334994-.348692ZM7.103362-6.4757"
        "16H8.159402C7.940224-6.22665 7.940224-5.967621 7.940224-5.788294V-1.0361"
        "15C7.940224-.856787 7.940224-.597758 8.169365-.348692H6.884184C7.103362-"
        ".597758 7.103362-.856787 7.103362-1.036115V-6.475716Z"
    ),
    se.Path(  # letter a
        "M1.464508-4.024907C1.464508-4.234122 1.743462-4.393524 2.092154-4.393524"
        "C2.669988-4.393524 2.929016-4.124533 2.929016-3.516812V-2.789539C1.77335"
        "-2.440847 .249066-2.042341 .249066-.916563C.249066-.308842 .71731 .13947"
        "7 1.354919 .139477C1.92279 .139477 2.381071-.059776 2.929016-.557908C3.0"
        "38605-.049813 3.257783 .139477 3.745953 .139477C4.174346 .139477 4.48318"
        "8-.019925 4.861768-.428394L4.712329-.637609L4.612702-.537983C4.582814-.5"
        "08095 4.552927-.498132 4.503113-.498132C4.363636-.498132 4.293898-.58779"
        "6 4.293898-.747198V-3.347447C4.293898-4.184309 3.536737-4.712329 2.32129"
        "5-4.712329C1.195517-4.712329 .438356-4.204234 .438356-3.457036C.438356-3"
        ".048568 .67746-2.799502 1.085928-2.799502C1.484433-2.799502 1.763387-3.0"
        "38605 1.763387-3.377335C1.763387-3.676214 1.464508-3.88543 1.464508-4.02"
        "4907ZM2.919054-.996264C2.650062-.687422 2.450809-.56787 2.211706-.56787C"
        "1.912827-.56787 1.703611-.836862 1.703611-1.235367C1.703611-1.8132 2.122"
        "042-2.231631 2.919054-2.440847V-.996264Z"
    ),
    se.Path(  # letter n
        "M2.948941-4.044832C3.297634-4.044832 3.466999-3.775841 3.466999-3.217933"
        "V-.806974C3.466999-.438356 3.337484-.278954 2.998755-.239103V0H5.339975V"
        "-.239103C4.951432-.268991 4.851806-.388543 4.851806-.806974V-3.307597C4."
        "851806-4.164384 4.323786-4.712329 3.506849-4.712329C2.909091-4.712329 2."
        "450809-4.433375 2.082192-3.845579V-4.592777H.179328V-4.353674C.617684-4."
        "283935 .707347-4.184309 .707347-3.765878V-.836862C.707347-.418431 .62764"
        "6-.328767 .179328-.239103V0H2.580324V-.239103C2.211706-.288917 2.092154-"
        ".438356 2.092154-.806974V-3.466999C2.092154-3.576588 2.530511-4.044832 2"
        ".948941-4.044832Z"
    ),
    se.Path(  # letter i
        "M2.15193-4.592777H.239103V-4.353674C.67746-4.26401 .767123-4.174346 .767"
        "123-3.765878V-.836862C.767123-.428394 .697385-.348692 .239103-.239103V0H"
        "2.6401V-.239103C2.291407-.288917 2.15193-.428394 2.15193-.806974V-4.5927"
        "77ZM1.454545-6.884184C1.026152-6.884184 .67746-6.535492 .67746-6.117061C"
        ".67746-5.668742 1.006227-5.339975 1.444583-5.339975S2.221669-5.668742 2."
        "221669-6.107098C2.221669-6.535492 1.882939-6.884184 1.454545-6.884184Z"
    ),
    se.Path(  # letter m
        "M2.929016-4.044832C3.317559-4.044832 3.466999-3.815691 3.466999-3.217933"
        "V-.806974C3.466999-.398506 3.35741-.268991 2.988792-.239103V0H5.32005V-."
        "239103C4.971357-.278954 4.851806-.428394 4.851806-.806974V-3.466999C4.85"
        "1806-3.576588 5.310087-4.044832 5.69863-4.044832C6.07721-4.044832 6.2266"
        "5-3.805729 6.22665-3.217933V-.806974C6.22665-.388543 6.117061-.268991 5."
        "738481-.239103V0H8.109589V-.239103C7.721046-.259029 7.611457-.37858 7.61"
        "1457-.806974V-3.307597C7.611457-4.164384 7.083437-4.712329 6.266501-4.71"
        "2329C5.69863-4.712329 5.32005-4.483188 4.801993-3.845579C4.503113-4.4732"
        "25 4.154421-4.712329 3.526775-4.712329S2.440847-4.443337 2.062267-3.8455"
        "79V-4.592777H.179328V-4.353674C.617684-4.293898 .707347-4.174346 .707347"
        "-3.765878V-.836862C.707347-.428394 .617684-.318804 .179328-.239103V0H2.5"
        "50436V-.239103C2.201743-.288917 2.092154-.428394 2.092154-.806974V-3.466"
        "999C2.092154-3.58655 2.530511-4.044832 2.929016-4.044832Z"
    ),
]


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

        self.M = VMobjectFromSVGPath(MANIM_SVG_PATHS[0]).flip(cst.RIGHT).center()
        self.M.set(stroke_width=0).scale(
            7 * cst.DEFAULT_FONT_SIZE * cst.SCALE_FACTOR_PER_FONT_POINT
        )
        self.M.set_fill(color=self.font_color, opacity=1).shift(
            2.25 * cst.LEFT + 1.5 * cst.UP
        )

        self.circle = Circle(color=logo_green, fill_opacity=1).shift(cst.LEFT)
        self.square = Square(color=logo_blue, fill_opacity=1).shift(cst.UP)
        self.triangle = Triangle(color=logo_red, fill_opacity=1).shift(cst.RIGHT)
        self.shapes = VGroup(self.triangle, self.square, self.circle)
        self.add(self.shapes, self.M)
        self.move_to(cst.ORIGIN)

        anim = VGroup()
        for ind, path in enumerate(MANIM_SVG_PATHS[1:]):
            tex = VMobjectFromSVGPath(path).flip(cst.RIGHT).center()
            tex.set(stroke_width=0).scale(
                cst.DEFAULT_FONT_SIZE * cst.SCALE_FACTOR_PER_FONT_POINT
            )
            if ind > 0:
                tex.next_to(anim, buff=0.01)
            tex.align_to(self.M, cst.DOWN)
            anim.add(tex)
        anim.set_fill(color=self.font_color, opacity=1)
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
        self.anim.next_to(self.M, buff=m_anim_buff).align_to(self.M, cst.DOWN)
        self.anim.set_opacity(0)
        self.shapes.save_state()
        m_clone = self.anim[-1].copy()
        self.add(m_clone)
        m_clone.move_to(self.shapes)

        self.M.save_state()
        left_group = VGroup(self.M, self.anim, m_clone)

        def shift(vector):
            self.shapes.restore()
            left_group.align_to(self.M.saved_state, cst.LEFT)
            if direction == "right":
                self.shapes.shift(vector)
            elif direction == "center":
                self.shapes.shift(vector / 2)
                left_group.shift(-vector / 2)
            elif direction == "left":
                left_group.shift(-vector)

        def slide_and_uncover(mob, alpha):
            shift(alpha * (m_shape_offset + shape_sliding_overshoot) * cst.RIGHT)

            # Add letters when they are covered
            for letter in mob.anim:
                if mob.square.get_center()[0] > letter.get_center()[0]:
                    letter.set_opacity(1)
                    self.add_to_back(letter)

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

            shift(alpha * shape_sliding_overshoot * cst.LEFT)

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
