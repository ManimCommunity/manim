from manim.mobject.graphing.probability import BarChart


def test_values_close_to_zero():
    """Checks that BarChart supports values/heights close to zero without crashing if ."""
    values = [1 / 10000 for _ in range(8)]
    names = [i for i in range(len(values))]
    chart = BarChart(
        values=values,
        bar_names=["one", "two", "three", "four", "five"],
        y_range=[0, 2 / 10000],
        y_length=6,
        x_length=10,
        x_axis_config={"font_size": 36},
    )
