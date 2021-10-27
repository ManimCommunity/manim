r"""Animate the display or removal of a mobject from a scene.

.. manim:: CreationModule
    :hide_source:

    from manim import ManimBanner
    class CreationModule(Scene):
        def construct(self):
            s1 = Square()
            s2 = Square()
            s3 = Square()
            s4 = Square()
            VGroup(s1, s2, s3, s4).set_x(0).arrange(buff=1.9).shift(UP)
            s5 = Square()
            s6 = Square()
            s7 = Square()
            VGroup(s5, s6, s7).set_x(0).arrange(buff=2.6).shift(2 * DOWN)
            t1 = Text("Write", font_size=24).next_to(s1, UP)
            t2 = Text("AddTextLetterByLetter", font_size=24).next_to(s2, UP)
            t3 = Text("Create", font_size=24).next_to(s3, UP)
            t4 = Text("Uncreate", font_size=24).next_to(s4, UP)
            t5 = Text("DrawBorderThenFill", font_size=24).next_to(s5, UP)
            t6 = Text("ShowIncreasingSubsets", font_size=22).next_to(s6, UP)
            t7 = Text("ShowSubmobjectsOneByOne", font_size=22).next_to(s7, UP)

            self.add(s1, s2, s3, s4, s5, s6, s7, t1, t2, t3, t4, t5, t6, t7)

            texts = [Text("manim", font_size=29), Text("manim", font_size=29)]
            texts[0].move_to(s1.get_center())
            texts[1].move_to(s2.get_center())
            self.add(*texts)

            objs = [ManimBanner().scale(0.25) for _ in range(5)]
            objs[0].move_to(s3.get_center())
            objs[1].move_to(s4.get_center())
            objs[2].move_to(s5.get_center())
            objs[3].move_to(s6.get_center())
            objs[4].move_to(s7.get_center())
            self.add(*objs)

            self.play(
                # text creation
                Write(texts[0]),
                AddTextLetterByLetter(texts[1]),
                # mobject creation
                Create(objs[0]),
                Uncreate(objs[1]),
                DrawBorderThenFill(objs[2]),
                ShowIncreasingSubsets(objs[3]),
                ShowSubmobjectsOneByOne(objs[4]),
                run_time=3,
            )

            self.wait()

"""


__all__ = [
    "Create",
    "Uncreate",
    "DrawBorderThenFill",
    "Write",
    "Unwrite",
    "ShowPartial",
    "ShowIncreasingSubsets",
    "AddTextLetterByLetter",
    "ShowSubmobjectsOneByOne",
    "AddTextWordByWord",
]


import itertools as it
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from colour import Color

if TYPE_CHECKING:
    from manim.mobject.svg.text_mobject import Text

from ..animation.animation import Animation
from ..animation.composition import Succession
from ..mobject.mobject import Group, Mobject
from ..mobject.types.opengl_surface import OpenGLSurface
from ..mobject.types.opengl_vectorized_mobject import OpenGLVMobject
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.bezier import integer_interpolate
from ..utils.rate_functions import double_smooth, linear, smooth


class ShowPartial(Animation):
    """Abstract class for Animations that show the VMobject partially.

    Raises
    ------
    :class:`TypeError`
        If ``mobject`` is not an instance of :class:`~.VMobject`.

    See Also
    --------
    :class:`Create`, :class:`~.ShowPassingFlash`

    """

    def __init__(
        self, mobject: Union[VMobject, OpenGLVMobject, OpenGLSurface, None], **kwargs
    ):
        pointwise = getattr(mobject, "pointwise_become_partial", None)
        if not callable(pointwise):
            raise NotImplementedError("This animation is not defined for this Mobject.")
        super().__init__(mobject, **kwargs)

    def interpolate_submobject(
        self,
        submobject: Mobject,
        starting_submobject: Mobject,
        alpha: float,
    ) -> None:
        submobject.pointwise_become_partial(
            starting_submobject, *self._get_bounds(alpha)
        )

    def _get_bounds(self, alpha: float) -> None:
        raise NotImplementedError("Please use Create or ShowPassingFlash")


class Create(ShowPartial):
    """Incrementally show a VMobject.

    Parameters
    ----------
    mobject : :class:`~.VMobject`
        The VMobject to animate.

    Raises
    ------
    :class:`TypeError`
        If ``mobject`` is not an instance of :class:`~.VMobject`.

    Examples
    --------
    .. manim:: CreateScene

        class CreateScene(Scene):
            def construct(self):
                self.play(Create(Square()))

    See Also
    --------
    :class:`~.ShowPassingFlash`

    """

    def __init__(
        self,
        mobject: Union[VMobject, OpenGLVMobject, OpenGLSurface],
        lag_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(mobject, lag_ratio=lag_ratio, **kwargs)

    def _get_bounds(self, alpha: float) -> Tuple[int, float]:
        return (0, alpha)


class Uncreate(Create):
    """Like :class:`Create` but in reverse.

    Examples
    --------
    .. manim:: ShowUncreate

        class ShowUncreate(Scene):
            def construct(self):
                self.play(Uncreate(Square()))

    See Also
    --------
    :class:`Create`

    """

    def __init__(
        self,
        mobject: Union[VMobject, OpenGLVMobject],
        rate_func: Callable[[float], float] = lambda t: smooth(1 - t),
        remover: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(mobject, rate_func=rate_func, remover=remover, **kwargs)


class DrawBorderThenFill(Animation):
    """Draw the border first and then show the fill.

    Examples
    --------
    .. manim:: ShowDrawBorderThenFill

        class ShowDrawBorderThenFill(Scene):
            def construct(self):
                self.play(DrawBorderThenFill(Square(fill_opacity=1, fill_color=ORANGE)))
    """

    def __init__(
        self,
        vmobject: Union[VMobject, OpenGLVMobject],
        run_time: float = 2,
        rate_func: Callable[[float], float] = double_smooth,
        stroke_width: float = 2,
        stroke_color: str = None,
        draw_border_animation_config: Dict = {},  # what does this dict accept?
        fill_animation_config: Dict = {},
        **kwargs,
    ) -> None:
        self._typecheck_input(vmobject)
        super().__init__(vmobject, run_time=run_time, rate_func=rate_func, **kwargs)
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.draw_border_animation_config = draw_border_animation_config
        self.fill_animation_config = fill_animation_config
        self.outline = self.get_outline()

    def _typecheck_input(self, vmobject: Union[VMobject, OpenGLVMobject]) -> None:
        if not isinstance(vmobject, (VMobject, OpenGLVMobject)):
            raise TypeError("DrawBorderThenFill only works for vectorized Mobjects")

    def begin(self) -> None:
        self.outline = self.get_outline()
        super().begin()

    def get_outline(self) -> Mobject:
        outline = self.mobject.copy()
        outline.set_fill(opacity=0)
        for sm in outline.family_members_with_points():
            sm.set_stroke(color=self.get_stroke_color(sm), width=self.stroke_width)
        return outline

    def get_stroke_color(self, vmobject: Union[VMobject, OpenGLVMobject]) -> Color:
        if self.stroke_color:
            return self.stroke_color
        elif vmobject.get_stroke_width() > 0:
            return vmobject.get_stroke_color()
        return vmobject.get_color()

    def get_all_mobjects(self) -> Sequence[Mobject]:
        return [*super().get_all_mobjects(), self.outline]

    def interpolate_submobject(
        self,
        submobject: Mobject,
        starting_submobject: Mobject,
        outline,
        alpha: float,
    ) -> None:  # Fixme: not matching the parent class? What is outline doing here?
        index: int
        subalpha: int
        index, subalpha = integer_interpolate(0, 2, alpha)
        if index == 0:
            submobject.pointwise_become_partial(outline, 0, subalpha)
            submobject.match_style(outline)
        else:
            submobject.interpolate(outline, starting_submobject, subalpha)


class Write(DrawBorderThenFill):
    """Simulate hand-writing a :class:`~.Text` or hand-drawing a :class:`~.VMobject`.

    Examples
    --------
    .. manim:: ShowWrite

        class ShowWrite(Scene):
            def construct(self):
                self.play(Write(Text("Hello", font_size=144)))

    .. manim:: ShowWriteReversed

        class ShowWriteReversed(Scene):
            def construct(self):
                self.play(Write(Text("Hello", font_size=144), reverse=True))
    """

    def __init__(
        self,
        vmobject: Union[VMobject, OpenGLVMobject],
        rate_func: Callable[[float], float] = linear,
        reverse: bool = False,
        **kwargs,
    ) -> None:
        run_time: Optional[float] = kwargs.pop("run_time", None)
        lag_ratio: Optional[float] = kwargs.pop("lag_ratio", None)
        run_time, lag_ratio = self._set_default_config_from_length(
            vmobject,
            run_time,
            lag_ratio,
        )
        self.reverse = reverse
        super().__init__(
            vmobject,
            rate_func=rate_func,
            run_time=run_time,
            lag_ratio=lag_ratio,
            **kwargs,
        )

    def _set_default_config_from_length(
        self,
        vmobject: Union[VMobject, OpenGLVMobject],
        run_time: Optional[float],
        lag_ratio: Optional[float],
    ) -> Tuple[float, float]:
        length = len(vmobject.family_members_with_points())
        if run_time is None:
            if length < 15:
                run_time = 1
            else:
                run_time = 2
        if lag_ratio is None:
            lag_ratio = min(4.0 / length, 0.2)
        return run_time, lag_ratio

    def reverse_submobjects(self) -> None:
        self.mobject.invert(recursive=True)

    def begin(self) -> None:
        if self.reverse:
            self.reverse_submobjects()
        super().begin()

    def finish(self) -> None:
        super().finish()
        if self.reverse:
            self.reverse_submobjects()


class Unwrite(Write):
    """Simulate erasing by hand a :class:`~.Text` or a :class:`~.VMobject`.

    Parameters
    ----------
    reverse : :class:`bool`
        Set True to have the animation start erasing from the last submobject first.

    Examples
    --------

    .. manim :: UnwriteReverseTrue

        class UnwriteReverseTrue(Scene):
            def construct(self):
                text = Tex("Alice and Bob").scale(3)
                self.add(text)
                self.play(Unwrite(text))

    .. manim:: UnwriteReverseFalse

        class UnwriteReverseFalse(Scene):
            def construct(self):
                text = Tex("Alice and Bob").scale(3)
                self.add(text)
                self.play(Unwrite(text, reverse=False))
    """

    def __init__(
        self,
        vmobject: VMobject,
        rate_func: Callable[[float], float] = linear,
        reverse: bool = True,
        **kwargs,
    ) -> None:

        run_time: Optional[float] = kwargs.pop("run_time", None)
        lag_ratio: Optional[float] = kwargs.pop("lag_ratio", None)
        run_time, lag_ratio = self._set_default_config_from_length(
            vmobject,
            run_time,
            lag_ratio,
        )
        super().__init__(
            vmobject,
            run_time=run_time,
            lag_ratio=lag_ratio,
            rate_func=lambda t: -rate_func(t) + 1,
            reverse=reverse,
            **kwargs,
        )


class ShowIncreasingSubsets(Animation):
    """Show one submobject at a time, leaving all previous ones displayed on screen.

    Examples
    --------

    .. manim:: ShowIncreasingSubsetsScene

        class ShowIncreasingSubsetsScene(Scene):
            def construct(self):
                p = VGroup(Dot(), Square(), Triangle())
                self.add(p)
                self.play(ShowIncreasingSubsets(p))
                self.wait()
    """

    def __init__(
        self,
        group: Mobject,
        suspend_mobject_updating: bool = False,
        int_func: Callable[[np.ndarray], np.ndarray] = np.floor,
        **kwargs,
    ) -> None:
        self.all_submobs = list(group.submobjects)
        self.int_func = int_func
        for mobj in self.all_submobs:
            mobj.set_opacity(0)
        super().__init__(
            group, suspend_mobject_updating=suspend_mobject_updating, **kwargs
        )

    def interpolate_mobject(self, alpha: float) -> None:
        n_submobs = len(self.all_submobs)
        index = int(self.int_func(self.rate_func(alpha) * n_submobs))
        self.update_submobject_list(index)

    def update_submobject_list(self, index: int) -> None:
        for mobj in self.all_submobs[:index]:
            mobj.set_opacity(1)


class AddTextLetterByLetter(ShowIncreasingSubsets):
    """Show a :class:`~.Text` letter by letter on the scene.

    Parameters
    ----------
    time_per_char : :class:`float`
        Frequency of appearance of the letters.

    .. tip::

        This is currently only possible for class:`~.Text` and not for class:`~.MathTex`

    """

    def __init__(
        self,
        text: "Text",
        suspend_mobject_updating: bool = False,
        int_func: Callable[[np.ndarray], np.ndarray] = np.ceil,
        rate_func: Callable[[float], float] = linear,
        time_per_char: float = 0.1,
        run_time: Optional[float] = None,
        **kwargs,
    ) -> None:
        # time_per_char must be above 0.06, or the animation won't finish
        self.time_per_char = time_per_char
        if run_time is None:
            run_time = np.max((0.06, self.time_per_char)) * len(text)

        super().__init__(
            text,
            suspend_mobject_updating=suspend_mobject_updating,
            int_func=int_func,
            rate_func=rate_func,
            run_time=run_time,
            **kwargs,
        )


class ShowSubmobjectsOneByOne(ShowIncreasingSubsets):
    """Show one submobject at a time, removing all previously displayed ones from screen."""

    def __init__(
        self,
        group: Iterable[Mobject],
        int_func: Callable[[np.ndarray], np.ndarray] = np.ceil,
        **kwargs,
    ) -> None:
        new_group = Group(*group)
        super().__init__(new_group, int_func=int_func, **kwargs)

    def update_submobject_list(self, index: int) -> None:
        current_submobjects = self.all_submobs[:index]
        for mobj in current_submobjects[:-1]:
            mobj.set_opacity(0)
        if len(current_submobjects) > 0:
            current_submobjects[-1].set_opacity(1)


# TODO, this is broken...
class AddTextWordByWord(Succession):
    """Show a :class:`~.Text` word by word on the scene. Note: currently broken."""

    def __init__(
        self,
        text_mobject: "Text",
        run_time: float = None,
        time_per_char: float = 0.06,
        **kwargs,
    ) -> None:
        self.time_per_char = time_per_char
        tpc = self.time_per_char
        anims = it.chain(
            *(
                [
                    ShowIncreasingSubsets(word, run_time=tpc * len(word)),
                    Animation(word, run_time=0.005 * len(word) ** 1.5),
                ]
                for word in text_mobject
            )
        )
        super().__init__(*anims, **kwargs)
