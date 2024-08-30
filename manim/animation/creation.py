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

from __future__ import annotations

__all__ = [
    "Create",
    "Uncreate",
    "DrawBorderThenFill",
    "Write",
    "Unwrite",
    "ShowPartial",
    "ShowIncreasingSubsets",
    "SpiralIn",
    "AddTextLetterByLetter",
    "RemoveTextLetterByLetter",
    "ShowSubmobjectsOneByOne",
    "AddTextWordByWord",
    "TypeWithCursor",
    "UntypeWithCursor",
]


import itertools as it
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from manim.mobject.text.text_mobject import Text
    from manim.scene.scene import Scene

from manim.constants import RIGHT, TAU
from manim.mobject.opengl.opengl_surface import OpenGLSurface
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.utils.color import ManimColor

from .. import config
from ..animation.animation import Animation
from ..animation.composition import Succession
from ..mobject.mobject import Group, Mobject
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.bezier import integer_interpolate
from ..utils.rate_functions import double_smooth, linear


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
        self,
        mobject: VMobject | OpenGLVMobject | OpenGLSurface | None,
        **kwargs,
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
    mobject
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
        mobject: VMobject | OpenGLVMobject | OpenGLSurface,
        lag_ratio: float = 1.0,
        introducer: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(mobject, lag_ratio=lag_ratio, introducer=introducer, **kwargs)

    def _get_bounds(self, alpha: float) -> tuple[int, float]:
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
        mobject: VMobject | OpenGLVMobject,
        reverse_rate_function: bool = True,
        remover: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            mobject,
            reverse_rate_function=reverse_rate_function,
            introducer=False,
            remover=remover,
            **kwargs,
        )


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
        vmobject: VMobject | OpenGLVMobject,
        run_time: float = 2,
        rate_func: Callable[[float], float] = double_smooth,
        stroke_width: float = 2,
        stroke_color: str = None,
        draw_border_animation_config: dict = {},  # what does this dict accept?
        fill_animation_config: dict = {},
        introducer: bool = True,
        **kwargs,
    ) -> None:
        self._typecheck_input(vmobject)
        super().__init__(
            vmobject,
            run_time=run_time,
            introducer=introducer,
            rate_func=rate_func,
            **kwargs,
        )
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.draw_border_animation_config = draw_border_animation_config
        self.fill_animation_config = fill_animation_config
        self.outline = self.get_outline()

    def _typecheck_input(self, vmobject: VMobject | OpenGLVMobject) -> None:
        if not isinstance(vmobject, (VMobject, OpenGLVMobject)):
            raise TypeError(
                f"{self.__class__.__name__} only works for vectorized Mobjects"
            )

    def begin(self) -> None:
        self.outline = self.get_outline()
        super().begin()

    def get_outline(self) -> Mobject:
        outline = self.mobject.copy()
        outline.set_fill(opacity=0)
        for sm in outline.family_members_with_points():
            sm.set_stroke(color=self.get_stroke_color(sm), width=self.stroke_width)
        return outline

    def get_stroke_color(self, vmobject: VMobject | OpenGLVMobject) -> ManimColor:
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
        subalpha: float
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
                self.play(Write(Text("Hello", font_size=144), reverse=True, remover=False))

    Tests
    -----

    Check that creating empty :class:`.Write` animations works::

        >>> from manim import Write, Text
        >>> Write(Text(''))
        Write(Text(''))
    """

    def __init__(
        self,
        vmobject: VMobject | OpenGLVMobject,
        rate_func: Callable[[float], float] = linear,
        reverse: bool = False,
        **kwargs,
    ) -> None:
        run_time: float | None = kwargs.pop("run_time", None)
        lag_ratio: float | None = kwargs.pop("lag_ratio", None)
        run_time, lag_ratio = self._set_default_config_from_length(
            vmobject,
            run_time,
            lag_ratio,
        )
        self.reverse = reverse
        if "remover" not in kwargs:
            kwargs["remover"] = reverse
        super().__init__(
            vmobject,
            rate_func=rate_func,
            run_time=run_time,
            lag_ratio=lag_ratio,
            introducer=not reverse,
            **kwargs,
        )

    def _set_default_config_from_length(
        self,
        vmobject: VMobject | OpenGLVMobject,
        run_time: float | None,
        lag_ratio: float | None,
    ) -> tuple[float, float]:
        length = len(vmobject.family_members_with_points())
        if run_time is None:
            run_time = 1 if length < 15 else 2
        if lag_ratio is None:
            lag_ratio = min(4.0 / max(1.0, length), 0.2)
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
    reverse
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
        run_time: float | None = kwargs.pop("run_time", None)
        lag_ratio: float | None = kwargs.pop("lag_ratio", None)
        run_time, lag_ratio = self._set_default_config_from_length(
            vmobject,
            run_time,
            lag_ratio,
        )
        super().__init__(
            vmobject,
            run_time=run_time,
            lag_ratio=lag_ratio,
            reverse_rate_function=True,
            reverse=reverse,
            **kwargs,
        )


class SpiralIn(Animation):
    r"""Create the Mobject with sub-Mobjects flying in on spiral trajectories.

    Parameters
    ----------
    shapes
        The Mobject on which to be operated.

    scale_factor
        The factor used for scaling the effect.

    fade_in_fraction
        Fractional duration of initial fade-in of sub-Mobjects as they fly inward.

    Examples
    --------
    .. manim :: SpiralInExample

        class SpiralInExample(Scene):
            def construct(self):
                pi = MathTex(r"\pi").scale(7)
                pi.shift(2.25 * LEFT + 1.5 * UP)
                circle = Circle(color=GREEN_C, fill_opacity=1).shift(LEFT)
                square = Square(color=BLUE_D, fill_opacity=1).shift(UP)
                shapes = VGroup(pi, circle, square)
                self.play(SpiralIn(shapes))
    """

    def __init__(
        self,
        shapes: Mobject,
        scale_factor: float = 8,
        fade_in_fraction=0.3,
        **kwargs,
    ) -> None:
        self.shapes = shapes.copy()
        self.scale_factor = scale_factor
        self.shape_center = shapes.get_center()
        self.fade_in_fraction = fade_in_fraction
        for shape in shapes:
            shape.final_position = shape.get_center()
            shape.initial_position = (
                shape.final_position
                + (shape.final_position - self.shape_center) * self.scale_factor
            )
            shape.move_to(shape.initial_position)
            shape.save_state()

        super().__init__(shapes, introducer=True, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        alpha = self.rate_func(alpha)
        for original_shape, shape in zip(self.shapes, self.mobject):
            shape.restore()
            fill_opacity = original_shape.get_fill_opacity()
            stroke_opacity = original_shape.get_stroke_opacity()
            new_fill_opacity = min(
                fill_opacity, alpha * fill_opacity / self.fade_in_fraction
            )
            new_stroke_opacity = min(
                stroke_opacity, alpha * stroke_opacity / self.fade_in_fraction
            )
            shape.shift((shape.final_position - shape.initial_position) * alpha)
            shape.rotate(TAU * alpha, about_point=self.shape_center)
            shape.rotate(-TAU * alpha, about_point=shape.get_center_of_mass())
            shape.set_fill(opacity=new_fill_opacity)
            shape.set_stroke(opacity=new_stroke_opacity)


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
        reverse_rate_function=False,
        **kwargs,
    ) -> None:
        self.all_submobs = list(group.submobjects)
        self.int_func = int_func
        for mobj in self.all_submobs:
            mobj.set_opacity(0)
        super().__init__(
            group,
            suspend_mobject_updating=suspend_mobject_updating,
            reverse_rate_function=reverse_rate_function,
            **kwargs,
        )

    def interpolate_mobject(self, alpha: float) -> None:
        n_submobs = len(self.all_submobs)
        value = (
            1 - self.rate_func(alpha)
            if self.reverse_rate_function
            else self.rate_func(alpha)
        )
        index = int(self.int_func(value * n_submobs))
        self.update_submobject_list(index)

    def update_submobject_list(self, index: int) -> None:
        for mobj in self.all_submobs[:index]:
            mobj.set_opacity(1)
        for mobj in self.all_submobs[index:]:
            mobj.set_opacity(0)


class AddTextLetterByLetter(ShowIncreasingSubsets):
    """Show a :class:`~.Text` letter by letter on the scene.

    Parameters
    ----------
    time_per_char
        Frequency of appearance of the letters.

    .. tip::

        This is currently only possible for class:`~.Text` and not for class:`~.MathTex`

    """

    def __init__(
        self,
        text: Text,
        suspend_mobject_updating: bool = False,
        int_func: Callable[[np.ndarray], np.ndarray] = np.ceil,
        rate_func: Callable[[float], float] = linear,
        time_per_char: float = 0.1,
        run_time: float | None = None,
        reverse_rate_function=False,
        introducer=True,
        **kwargs,
    ) -> None:
        self.time_per_char = time_per_char
        # Check for empty text using family_members_with_points()
        if not text.family_members_with_points():
            raise ValueError(
                f"The text mobject {text} does not seem to contain any characters."
            )
        if run_time is None:
            # minimum time per character is 1/frame_rate, otherwise
            # the animation does not finish.
            run_time = np.max((1 / config.frame_rate, self.time_per_char)) * len(text)
        super().__init__(
            text,
            suspend_mobject_updating=suspend_mobject_updating,
            int_func=int_func,
            rate_func=rate_func,
            run_time=run_time,
            reverse_rate_function=reverse_rate_function,
            introducer=introducer,
            **kwargs,
        )


class RemoveTextLetterByLetter(AddTextLetterByLetter):
    """Remove a :class:`~.Text` letter by letter from the scene.

    Parameters
    ----------
    time_per_char
        Frequency of appearance of the letters.

    .. tip::

        This is currently only possible for class:`~.Text` and not for class:`~.MathTex`

    """

    def __init__(
        self,
        text: Text,
        suspend_mobject_updating: bool = False,
        int_func: Callable[[np.ndarray], np.ndarray] = np.ceil,
        rate_func: Callable[[float], float] = linear,
        time_per_char: float = 0.1,
        run_time: float | None = None,
        reverse_rate_function=True,
        introducer=False,
        remover=True,
        **kwargs,
    ) -> None:
        super().__init__(
            text,
            suspend_mobject_updating=suspend_mobject_updating,
            int_func=int_func,
            rate_func=rate_func,
            time_per_char=time_per_char,
            run_time=run_time,
            reverse_rate_function=reverse_rate_function,
            introducer=introducer,
            remover=remover,
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
        text_mobject: Text,
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


class TypeWithCursor(AddTextLetterByLetter):
    """Similar to :class:`~.AddTextLetterByLetter` , but with an additional cursor mobject at the end.

    Parameters
    ----------
    time_per_char
        Frequency of appearance of the letters.
    cursor
        :class:`~.Mobject` shown after the last added letter.
    buff
        Controls how far away the cursor is to the right of the last added letter.
    keep_cursor_y
        If ``True``, the cursor's y-coordinate is set to the center of the ``Text`` and remains the same throughout the animation. Otherwise, it is set to the center of the last added letter.
    leave_cursor_on
        Whether to show the cursor after the animation.

    .. tip::
        This is currently only possible for class:`~.Text` and not for class:`~.MathTex`.


    Examples
    --------

    .. manim:: InsertingTextExample
        :ref_classes: Blink

        class InsertingTextExample(Scene):
            def construct(self):
                text = Text("Inserting", color=PURPLE).scale(1.5).to_edge(LEFT)
                cursor = Rectangle(
                    color = GREY_A,
                    fill_color = GREY_A,
                    fill_opacity = 1.0,
                    height = 1.1,
                    width = 0.5,
                ).move_to(text[0]) # Position the cursor

                self.play(TypeWithCursor(text, cursor))
                self.play(Blink(cursor, blinks=2))

    """

    def __init__(
        self,
        text: Text,
        cursor: Mobject,
        buff: float = 0.1,
        keep_cursor_y: bool = True,
        leave_cursor_on: bool = True,
        time_per_char: float = 0.1,
        reverse_rate_function=False,
        introducer=True,
        **kwargs,
    ) -> None:
        self.cursor = cursor
        self.buff = buff
        self.keep_cursor_y = keep_cursor_y
        self.leave_cursor_on = leave_cursor_on
        super().__init__(
            text,
            time_per_char=time_per_char,
            reverse_rate_function=reverse_rate_function,
            introducer=introducer,
            **kwargs,
        )

    def begin(self) -> None:
        self.y_cursor = self.cursor.get_y()
        self.cursor.initial_position = self.mobject.get_center()
        if self.keep_cursor_y:
            self.cursor.set_y(self.y_cursor)

        self.cursor.set_opacity(0)
        self.mobject.add(self.cursor)
        super().begin()

    def finish(self) -> None:
        if self.leave_cursor_on:
            self.cursor.set_opacity(1)
        else:
            self.cursor.set_opacity(0)
            self.mobject.remove(self.cursor)
        super().finish()

    def clean_up_from_scene(self, scene: Scene) -> None:
        if not self.leave_cursor_on:
            scene.remove(self.cursor)
        super().clean_up_from_scene(scene)

    def update_submobject_list(self, index: int) -> None:
        for mobj in self.all_submobs[:index]:
            mobj.set_opacity(1)

        for mobj in self.all_submobs[index:]:
            mobj.set_opacity(0)

        if index != 0:
            self.cursor.next_to(
                self.all_submobs[index - 1], RIGHT, buff=self.buff
            ).set_y(self.cursor.initial_position[1])
        else:
            self.cursor.move_to(self.all_submobs[0]).set_y(
                self.cursor.initial_position[1]
            )

        if self.keep_cursor_y:
            self.cursor.set_y(self.y_cursor)
        self.cursor.set_opacity(1)


class UntypeWithCursor(TypeWithCursor):
    """Similar to :class:`~.RemoveTextLetterByLetter` , but with an additional cursor mobject at the end.

    Parameters
    ----------
    time_per_char
        Frequency of appearance of the letters.
    cursor
        :class:`~.Mobject` shown after the last added letter.
    buff
        Controls how far away the cursor is to the right of the last added letter.
    keep_cursor_y
        If ``True``, the cursor's y-coordinate is set to the center of the ``Text`` and remains the same throughout the animation. Otherwise, it is set to the center of the last added letter.
    leave_cursor_on
        Whether to show the cursor after the animation.

    .. tip::
        This is currently only possible for class:`~.Text` and not for class:`~.MathTex`.


    Examples
    --------

    .. manim:: DeletingTextExample
        :ref_classes: Blink

        class DeletingTextExample(Scene):
            def construct(self):
                text = Text("Deleting", color=PURPLE).scale(1.5).to_edge(LEFT)
                cursor = Rectangle(
                    color = GREY_A,
                    fill_color = GREY_A,
                    fill_opacity = 1.0,
                    height = 1.1,
                    width = 0.5,
                ).move_to(text[0]) # Position the cursor

                self.play(UntypeWithCursor(text, cursor))
                self.play(Blink(cursor, blinks=2))

    """

    def __init__(
        self,
        text: Text,
        cursor: VMobject | None = None,
        time_per_char: float = 0.1,
        reverse_rate_function=True,
        introducer=False,
        remover=True,
        **kwargs,
    ) -> None:
        super().__init__(
            text,
            cursor=cursor,
            time_per_char=time_per_char,
            reverse_rate_function=reverse_rate_function,
            introducer=introducer,
            remover=remover,
            **kwargs,
        )
