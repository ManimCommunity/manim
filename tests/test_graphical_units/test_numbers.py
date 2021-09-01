from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "numbers"


@frames_comparison(last_frame=False)
def test_set_value_with_updaters(scene):
    """Test that the position of the decimal updates properly"""
    decimal = DecimalNumber(
        0,
        show_ellipsis=True,
        num_decimal_places=3,
        include_sign=True,
    )
    square = Square().to_edge(UP)

    decimal.add_updater(lambda d: d.next_to(square, RIGHT))
    decimal.add_updater(lambda d: d.set_value(square.get_center()[1]))
    scene.add(square, decimal)
    scene.play(
        square.animate.to_edge(DOWN),
        rate_func=there_and_back,
    )
