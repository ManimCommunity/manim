import pytest

from manim.constants import LEFT
from manim.mobject.probability import BarChart
from manim.mobject.svg.tex_mobject import MathTex
from manim.utils.color import BLUE, GREEN, RED, WHITE, YELLOW
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "probability"


@frames_comparison
def test_default_chart(scene):
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

    chart = BarChart(pull_req, versions)
    scene.add(chart)


@frames_comparison
def test_get_bar_labels(scene):
    chart = BarChart(values=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], y_range=[0, 10, 1])

    c_bar_lbls = chart.get_bar_labels(
        color=WHITE, label_constructor=MathTex, font_size=36
    )

    scene.add(chart, c_bar_lbls)


@frames_comparison
def test_label_constructor(scene):
    chart = BarChart(
        values=[25, 46, 50, 10],
        bar_names=[r"\alpha \beta \gamma", r"a+c", r"\sqrt{a \over b}", r"a^2+b^2"],
        y_length=5,
        x_length=5,
        bar_width=0.8,
        x_axis_config={"font_size": 36, "label_constructor": MathTex},
    )

    scene.add(chart)


@frames_comparison
def test_negative_values(scene):
    chart = BarChart(
        values=[-5, 40, -10, 20, -3],
        bar_names=["one", "two", "three", "four", "five"],
        y_range=[-20, 50, 10],
    )

    c_bar_lbls = chart.get_bar_labels()

    scene.add(chart, c_bar_lbls)


@frames_comparison
def test_advanced_customization(scene):
    """Tests to make sure advanced customization can be done through :class:`~.BarChart`"""
    chart = BarChart(values=[10, 40, 10, 20], bar_names=["one", "two", "three", "four"])

    c_x_lbls = chart.x_axis.labels
    c_x_lbls.set_color_by_gradient(GREEN, RED, YELLOW)

    c_y_nums = chart.y_axis.numbers
    c_y_nums.set_color_by_gradient(BLUE, WHITE).shift(LEFT)

    c_y_axis = chart.y_axis
    c_y_axis.ticks.set_color(YELLOW)

    c_bar_lbls = chart.get_bar_labels()

    scene.add(chart, c_bar_lbls)


@frames_comparison
def test_change_bar_values_some_vals(scene):
    chart = BarChart(
        values=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
        y_range=[-10, 10, 2],
        y_axis_config={"font_size": 24},
    )
    scene.add(chart)

    chart.change_bar_values([-6, -4, -2])

    scene.add(chart.get_bar_labels(font_size=24))


@frames_comparison
def test_change_bar_values_negative(scene):
    chart = BarChart(
        values=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
        y_range=[-10, 10, 2],
        y_axis_config={"font_size": 24},
    )
    scene.add(chart)

    chart.change_bar_values(list(reversed([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])))

    scene.add(chart.get_bar_labels(font_size=24))
