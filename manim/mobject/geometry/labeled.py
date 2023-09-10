r"""Mobjects that inherit from lines and contain a label along the length."""

from __future__ import annotations

__all__ = ["LabeledLine", "LabeledArrow"]

from manim.constants import *
from manim.mobject.geometry.line import Arrow, Line
from manim.mobject.geometry.shape_matchers import (
    BackgroundRectangle,
    SurroundingRectangle,
)
from manim.mobject.text.tex_mobject import MathTex, Tex
from manim.mobject.text.text_mobject import Text
from manim.utils.color import WHITE, ManimColor, ParsableManimColor


class LabeledLine(Line):
    """Constructs a line containing a label box somewhere along its length.

    Parameters
    ----------
    label : str | Tex | MathTex | Text
        Label that will be displayed on the line.
    label_position : float | optional
        A ratio in the range [0-1] to indicate the position of the label with respect to the length of the line. Default value is 0.5.
    font_size : float | optional
        Control font size for the label. This parameter is only used when `label` is of type `str`.
    label_color: ParsableManimColor | optional
        The color of the label's text. This parameter is only used when `label` is of type `str`.
    label_frame : Bool | optional
        Add a `SurroundingRectangle` frame to the label box.
    frame_fill_color : ParsableManimColor | optional
        Background color to fill the label box. If no value is provided, the background color of the canvas will be used.
    frame_fill_opacity : float | optional
        Determine the opacity of the label box by passing a value in the range [0-1], where 0 indicates complete transparency and 1 means full opacity.

    .. seealso::
        :class:`LabeledArrow`

    Examples
    --------
    .. manim:: LabeledLineExample
        :save_last_frame:

        class LabeledLineExample(Scene):
            def construct(self):
                line = LabeledLine(
                    label          = '0.5',
                    label_position = 0.8,
                    font_size      = 20,
                    label_color    = WHITE,
                    label_frame    = True,

                    start=LEFT+DOWN,
                    end=RIGHT+UP)


                line.set_length(line.get_length() * 2)
                self.add(line)
    """

    def __init__(
        self,
        label: str | Tex | MathTex | Text,
        label_position: float = 0.5,
        font_size: float = DEFAULT_FONT_SIZE,
        label_color: ParsableManimColor = WHITE,
        label_frame: bool = True,
        frame_fill_color: ParsableManimColor = None,
        frame_fill_opacity: float = 1,
        *args,
        **kwargs,
    ) -> None:
        label_color = ManimColor(label_color)
        frame_fill_color = ManimColor(frame_fill_color)
        if isinstance(label, str):
            from manim import MathTex

            rendered_label = MathTex(label, color=label_color, font_size=font_size)
        else:
            rendered_label = label

        super().__init__(*args, **kwargs)

        # calculating the vector for the label position
        line_start, line_end = self.get_start_and_end()
        new_vec = (line_end - line_start) * label_position
        label_coords = line_start + new_vec

        # rendered_label.move_to(self.get_vector() * label_position)
        rendered_label.move_to(label_coords)

        box = BackgroundRectangle(
            rendered_label,
            buff=0.05,
            color=frame_fill_color,
            fill_opacity=frame_fill_opacity,
            stroke_width=0.5,
        )
        self.add(box)

        if label_frame:
            box_frame = SurroundingRectangle(
                rendered_label, buff=0.05, color=label_color, stroke_width=0.5
            )

            self.add(box_frame)

        self.add(rendered_label)


class LabeledArrow(LabeledLine, Arrow):
    """Constructs an arrow containing a label box somewhere along its length.
    This class inherits its label properties from `LabeledLine`, so the main parameters controlling it are the same.

    Parameters
    ----------
    label : str | Tex | MathTex | Text
        Label that will be displayed on the line.
    label_position : float | optional
        A ratio in the range [0-1] to indicate the position of the label with respect to the length of the line. Default value is 0.5.
    font_size : float | optional
        Control font size for the label. This parameter is only used when `label` is of type `str`.
    label_color: ParsableManimColor | optional
        The color of the label's text. This parameter is only used when `label` is of type `str`.
    label_frame : Bool | optional
        Add a `SurroundingRectangle` frame to the label box.
    frame_fill_color : ParsableManimColor | optional
        Background color to fill the label box. If no value is provided, the background color of the canvas will be used.
    frame_fill_opacity : float | optional
        Determine the opacity of the label box by passing a value in the range [0-1], where 0 indicates complete transparency and 1 means full opacity.


    .. seealso::
        :class:`LabeledLine`

    Examples
    --------
    .. manim:: LabeledArrowExample
        :save_last_frame:

        class LabeledArrowExample(Scene):
            def construct(self):
                l_arrow = LabeledArrow("0.5", start=LEFT*3, end=RIGHT*3 + UP*2, label_position=0.5)

                self.add(l_arrow)
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
