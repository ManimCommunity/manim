from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "tables"


@frames_comparison
def test_Table(scene):
    t = Table(
        [["1", "2"], ["3", "4"]],
        row_labels=[Tex("R1"), Tex("R2")],
        col_labels=[Tex("C1"), Tex("C2")],
        top_left_entry=MathTex("TOP"),
        element_to_mobject=Tex,
        include_outer_lines=True,
    )
    scene.add(t)


@frames_comparison
def test_MathTable(scene):
    t = MathTable([[1, 2], [3, 4]])
    scene.add(t)


@frames_comparison
def test_MobjectTable(scene):
    a = Circle().scale(0.5)
    t = MobjectTable([[a.copy(), a.copy()], [a.copy(), a.copy()]])
    scene.add(t)


@frames_comparison
def test_IntegerTable(scene):
    t = IntegerTable(
        np.arange(1, 21).reshape(5, 4),
    )
    scene.add(t)


@frames_comparison
def test_DecimalTable(scene):
    t = DecimalTable(
        np.linspace(0, 0.9, 20).reshape(5, 4),
    )
    scene.add(t)
