"""Mobjects representing objects from probability theory and statistics."""

__all__ = ["SampleSpace", "BarChart"]


import typing
from typing import Iterable, Optional, Sequence, Union


from .. import config
from ..constants import *
from ..mobject.coordinate_systems import Axes
from ..mobject.geometry import Rectangle
from ..mobject.mobject import Mobject
from ..mobject.opengl_mobject import OpenGLMobject
from ..mobject.svg.brace import Brace
from ..mobject.svg.tex_mobject import MathTex, Tex
# from ..mobject.svg.text_mobject import *
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.color import (
    BLUE,
    BLUE_E,
    DARK_GREY,
    GREEN,
    GREEN_E,
    LIGHT_GREY,
    MAROON_B,
    WHITE,
    YELLOW,
    color_gradient,
)
from ..utils.iterables import tuplify

EPSILON = 0.0001


class SampleSpace(Rectangle):
    """

    Examples
    --------

    .. manim:: ExampleSampleSpace
        :save_last_frame:

        class ExampleSampleSpace(Scene):
            def construct(self):
                poly1 = SampleSpace(stroke_width=15, fill_opacity=1)
                poly2 = SampleSpace(width=5, height=3, stroke_width=5, fill_opacity=0.5)
                poly3 = SampleSpace(width=2, height=2, stroke_width=5, fill_opacity=0.1)
                poly3.divide_vertically(p_list=np.array([0.37, 0.13, 0.5]), colors=[BLACK, WHITE, GRAY], vect=RIGHT)
                poly_group = VGroup(poly1, poly2, poly3).arrange()
                self.add(poly_group)
    """

    def __init__(
        self,
        height=3,
        width=3,
        fill_color=DARK_GREY,
        fill_opacity=1,
        stroke_width=0.5,
        stroke_color=LIGHT_GREY,
        default_label_scale_val=1,
    ):
        super().__init__(
            height=height,
            width=width,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
        )
        self.default_label_scale_val = default_label_scale_val

    def add_title(self, title="Sample space", buff=MED_SMALL_BUFF):
        # TODO, should this really exist in SampleSpaceScene
        title_mob = Tex(title)
        if title_mob.width > self.width:
            title_mob.width = self.width
        title_mob.next_to(self, UP, buff=buff)
        self.title = title_mob
        self.add(title_mob)

    def add_label(self, label):
        self.label = label

    def complete_p_list(self, p_list):
        new_p_list = list(tuplify(p_list))
        remainder = 1.0 - sum(new_p_list)
        if abs(remainder) > EPSILON:
            new_p_list.append(remainder)
        return new_p_list

    def get_division_along_dimension(self, p_list, dim, colors, vect):
        p_list = self.complete_p_list(p_list)
        colors = color_gradient(colors, len(p_list))

        last_point = self.get_edge_center(-vect)
        parts = VGroup()
        for factor, color in zip(p_list, colors):
            part = SampleSpace()
            part.set_fill(color, 1)
            part.replace(self, stretch=True)
            part.stretch(factor, dim)
            part.move_to(last_point, -vect)
            last_point = part.get_edge_center(vect)
            parts.add(part)
        return parts

    def get_horizontal_division(self, p_list, colors=[GREEN_E, BLUE_E], vect=DOWN):
        return self.get_division_along_dimension(p_list, 1, colors, vect)

    def get_vertical_division(self, p_list, colors=[MAROON_B, YELLOW], vect=RIGHT):
        return self.get_division_along_dimension(p_list, 0, colors, vect)

    def divide_horizontally(self, *args, **kwargs):
        self.horizontal_parts = self.get_horizontal_division(*args, **kwargs)
        self.add(self.horizontal_parts)

    def divide_vertically(self, *args, **kwargs):
        self.vertical_parts = self.get_vertical_division(*args, **kwargs)
        self.add(self.vertical_parts)

    def get_subdivision_braces_and_labels(
        self,
        parts,
        labels,
        direction,
        buff=SMALL_BUFF,
        min_num_quads=1,
    ):
        label_mobs = VGroup()
        braces = VGroup()
        for label, part in zip(labels, parts):
            brace = Brace(part, direction, min_num_quads=min_num_quads, buff=buff)
            if isinstance(label, (Mobject, OpenGLMobject)):
                label_mob = label
            else:
                label_mob = MathTex(label)
                label_mob.scale(self.default_label_scale_val)
            label_mob.next_to(brace, direction, buff)

            braces.add(brace)
            label_mobs.add(label_mob)
        parts.braces = braces
        parts.labels = label_mobs
        parts.label_kwargs = {
            "labels": label_mobs.copy(),
            "direction": direction,
            "buff": buff,
        }
        return VGroup(parts.braces, parts.labels)

    def get_side_braces_and_labels(self, labels, direction=LEFT, **kwargs):
        assert hasattr(self, "horizontal_parts")
        parts = self.horizontal_parts
        return self.get_subdivision_braces_and_labels(
            parts, labels, direction, **kwargs
        )

    def get_top_braces_and_labels(self, labels, **kwargs):
        assert hasattr(self, "vertical_parts")
        parts = self.vertical_parts
        return self.get_subdivision_braces_and_labels(parts, labels, UP, **kwargs)

    def get_bottom_braces_and_labels(self, labels, **kwargs):
        assert hasattr(self, "vertical_parts")
        parts = self.vertical_parts
        return self.get_subdivision_braces_and_labels(parts, labels, DOWN, **kwargs)

    def add_braces_and_labels(self):
        for attr in "horizontal_parts", "vertical_parts":
            if not hasattr(self, attr):
                continue
            parts = getattr(self, attr)
            for subattr in "braces", "labels":
                if hasattr(parts, subattr):
                    self.add(getattr(parts, subattr))

    def __getitem__(self, index):
        if hasattr(self, "horizontal_parts"):
            return self.horizontal_parts[index]
        elif hasattr(self, "vertical_parts"):
            return self.vertical_parts[index]
        return self.split()[index]


class BarChart(Axes):
    """This is a class for easily creating a Bar Chart. The only required parameter is a list of `values`. Everything else is automatically calculated to have a good looking on screen.

    Parameters
    ----------
    values
        A list of values for each `bar`. It also accepts negative values.
    bar_names
        A list of names for each `bar`. It is optional to match the `values` list length.
    x_length
        The x_axis length. If `None` it is automatically adjusted depending on the number of values and the screen frame width.
    x_label_scale_value
        The scale value for `x_labels`. If `None` it is automatically calculated depending on the bar's width.
    y_length
        The y_axis length. If `None` it is automatically adjusted depending on the number of values and the screen frame height.
    y_range
        The y_axis range of values. If `None` it is automatically calculated caring about negative values.
    y_step
        The step value between `y_labels`. If `None` it is calculated to have one label each `y_length` unit, using two decimal places.
    y_include_numbers
        Whether or not to include 'y_labels'. Setting this parameter to `False` won't disappear the `y_axis` ticks, only the numbers.
    y_number_scale_value
        The scale value for 'y_axis' numbers. This parameter is automatically passed to 'y_axis_config' parameter of 'y_axis'.
    bar_colors
        The color for the bars. It is possible to give a single color as well as a list of colors. If the 'bar_colors' list length doesn't match the 'values' list length intermediate colors will be automatically calculated.
    bar_buff
        The space between a bar an the next one. This value is set in terms of 'x_axis' scale.
    bar_fill_opacity
        The fill opacity for the bars.
    bar_stroke_width
        The stroke width for the bars.

    Examples
    --------
    .. manim:: BarChartExample
        :save_last_frame:

        class BarChartExample(Scene):
            def construct(self):
                pull_req = [54, 23, 47, 48, 40, 64, 112, 87]
                versions = [
                    "v0.1.0",
                    "v0.1.1",
                    "v0.2.0",
                    "v0.3.0",
                    "v0.4.0",
                    "v0.5.0",
                    "v0.6.0",
                    "v0.7.0",
                ]
                colors = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]
                bar = BarChart(
                    values = pull_req,
                    bar_names = versions,
                    bar_colors = colors
                )
                self.add(bar)
    """

    def __init__(
        self,
        values: Iterable[float],
        bar_names: Optional[Iterable[str]] = None,
        y_range: Optional[Sequence[float]] = None,
        x_length: Optional[float] = None,
        x_label_buff=None,
        y_length: Optional[float] = config.frame_height - 4,
        bar_colors: Optional[Union[str, Iterable[str]]] = [
            "#003f5c",
            "#58508d",
            "#bc5090",
            "#ff6361",
            "#ffa600",
        ],
        bar_buff: Optional[float] = MED_LARGE_BUFF,
        bar_fill_opacity: Optional[float] = 0.7,
        bar_stroke_width: Optional[float] = 3,
        **kwargs,
    ) -> "VGroup":

        self.values = values
        self.bar_names = bar_names
        self.x_label_buff = x_label_buff
        self.bar_colors = bar_colors
        self.bar_buff = bar_buff
        self.bar_fill_opacity = bar_fill_opacity
        self.bar_stroke_width = bar_stroke_width

        if len(y_range) == 2:
            y_range = [*y_range,y_range[1]/len(self.values)]  

        x_range = [0, len(self.values), 1]

        if y_range is None:
            y_range = [
                min(0, min(self.values)),
                max(0, max(self.values)),
                self.y_step,
            ]
        elif len(y_range) == 2:
            self.y_step = round(max(self.values) / y_length, 2)

        if x_length is None:
            x_length = min(len(self.values), config.frame_width - 2)


        self.bars = None
        self.x_labels = None
        self.y_labels = None
        self.bar_labels = None

        super().__init__(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
            axis_config=self.axis_config,
            tips=kwargs.pop("tips", False),
            **kwargs,
        )

        self.add_bars()
        self.add_x_labels()
        self.center()

    def get_bars(self):
        return self.bars

    def get_bar_labels(
        self, color=None, scale=None, buff=MED_SMALL_BUFF, label_constructor=Tex
    ):
        if self.bar_labels is not None:
            return self.bar_labels
        self.bar_labels = VGroup()

        for bar, value in zip(self.bars, self.values):
            bar_lbl = label_constructor(str(value))

            if color is None:
                bar_lbl.set_color(bar.get_fill_color())
            else:
                bar_lbl.set_color(color)

            if scale is None:
                bar_lbl.scale_to_fit_width(min(bar.width * 0.6, 0.5))
                bar_lbl.height = min(bar_lbl.height, 0.25)
            else:
                bar_lbl.scale(scale)
            pos = UP if (value >= 0) else DOWN
            bar_lbl.next_to(bar, pos, buff=buff)
            self.bar_labels.add(bar_lbl)
        return self.bar_labels

    def get_values(self):
        return self.values

    def get_bar_names(self):
        return self.bar_names

    def add_bars(self):
        if self.bars is not None:
            return
        self.bars = VGroup()

        for i, value in enumerate(self.values):
            bar_h = abs(self.c2p(0, value)[1] - self.c2p(0, 0)[1])
            bar_w = self.c2p(1 - self.bar_buff, 0)[0] - self.c2p(0, 0)[0]
            bar = Rectangle(
                height=bar_h,
                width=bar_w,
                stroke_width=self.bar_stroke_width,
                fill_opacity=self.bar_fill_opacity,
            )

            pos = UP if (value >= 0) else DOWN
            bar.next_to(self.c2p(i + 0.5, 0), pos, buff=0)
            self.bars.add(bar)
        if isinstance(self.bar_colors, str):
            self.bars.set_color_by_gradient(self.bar_colors)
        else:
            self.bars.set_color_by_gradient(*self.bar_colors)

        self.add_to_back(self.bars)

    def add_x_labels(self):
        if self.x_labels is not None or self.bar_names is None:
            return
        self.x_labels = VGroup()

        max_lbl_width = 0
        max_lbl_height = 0
        for i, name in enumerate(self.bar_names):
            if i == len(self.values):
                self.bar_names = self.bar_names[:i]
                break
            label = self.x_label_constructor(name)
            if max_lbl_width < label.width:
                max_lbl_width = label.width
            if max_lbl_height < label.height:
                max_lbl_height = label.height
            self.x_labels.add(label)

        if self.x_label_scale_value is None:
            unit_width = self.c2p(1, 0)[0] - self.c2p(0, 0)[0]
            self.x_label_scale_value = min(unit_width * 0.75 / max_lbl_width, 0.75)

        if self.x_label_buff is None:
            self.x_label_buff = max_lbl_height * self.x_label_scale_value * 0.75

        for i, label in enumerate(self.x_labels):
            label.scale(self.x_label_scale_value)
            pos = DOWN if (self.values[i] >= 0) else UP
            label.move_to(
                self.c2p(i + 0.5, 0)
                + pos * max_lbl_height * self.x_label_scale_value / 2
                + pos * self.x_label_buff
            )

        self.add(self.x_labels)
