from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "tabular"


@frames_comparison
def test_tabular(scene):
    t = Tabular(
        [["1", "2"], ["3", "4"]],
        row_labels=[Tex("R1"), Tex("R2")],
        col_labels=[Tex("C1"), Tex("C2")],
        top_left_entry=MathTex("TOP"),
        element_to_mobject=Tex,
        include_outer_lines=True,
    )
    scene.add(t)


@frames_comparison
def test_MathTabular(scene):
    t = MathTabular([[1, 2], [3, 4]])
    scene.add(t)


@frames_comparison
def test_MobjectTabular(scene):
    a = Circle().scale(0.5)
    t = MobjectTabular([[a.copy(), a.copy()], [a.copy(), a.copy()]])
    scene.add(t)


@frames_comparison
def test_IntegerTabular(scene):
    t = IntegerTabular(
        np.arange(1, 21).reshape(5, 4),
    )
    scene.add(t)


@frames_comparison
def test_DecimalTabular(scene):
    t = DecimalTabular(
        np.linspace(0, 0.9, 20).reshape(5, 4),
    )
    scene.add(t)
