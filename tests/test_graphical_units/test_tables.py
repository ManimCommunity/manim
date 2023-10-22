from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

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


##############################################################################
# tests for add_highlighted_cell(function STARTS HERE
##############################################################################

########################
# Without Row/Col Labels
########################

######
# 1x1
######


@frames_comparison
def test_Table1x1_Empty(scene):
    t = Table([[""]])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty(scene):
    t = Table([["Cell 1"]])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


######
# 2x2
######


@frames_comparison
def test_Table2x2_Empty_1(scene):
    t = Table([["", "Cell 2"], ["Cell 3", "Cell 4"]])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_2(scene):
    t = Table([["", "Cell 2"], ["Cell 3", "Cell 4"]])
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_3(scene):
    t = Table([["", "Cell 2"], ["Cell 3", "Cell 4"]])
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_4(scene):
    t = Table([["", "Cell 2"], ["Cell 3", "Cell 4"]])
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_1(scene):
    t = Table([["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_2(scene):
    t = Table([["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]])
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_3(scene):
    t = Table([["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]])
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_4(scene):
    t = Table([["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]])
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


########################
# With only Row Labels
########################


######
# 1x1
######


@frames_comparison
def test_Table1x1_Empty_RL_1(scene):
    t = Table([[""]], row_labels=[Text("R1")])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_Empty_RL_2(scene):
    t = Table([[""]], row_labels=[Text("R1")])
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty_RL_1(scene):
    t = Table([["Cell 1"]], row_labels=[Text("R1")])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty_RL_2(scene):
    t = Table([["Cell 1"]], row_labels=[Text("R1")])
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


######
# 2x2
######


@frames_comparison
def test_Table2x2_Empty_RL_1(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], row_labels=[Text("R1"), Text("R2")]
    )
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RL_2(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], row_labels=[Text("R1"), Text("R2")]
    )
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RL_3(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], row_labels=[Text("R1"), Text("R2")]
    )
    t.add_highlighted_cell((1, 3), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RL_4(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], row_labels=[Text("R1"), Text("R2")]
    )
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RL_5(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], row_labels=[Text("R1"), Text("R2")]
    )
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RL_6(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], row_labels=[Text("R1"), Text("R2")]
    )
    t.add_highlighted_cell((2, 3), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RL_1(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
    )
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RL_2(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
    )
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RL_3(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
    )
    t.add_highlighted_cell((1, 3), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RL_4(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
    )
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RL_5(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
    )
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RL_6(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
    )
    t.add_highlighted_cell((2, 3), color=BLUE)
    scene.add(t)


########################
# With only Col Labels
########################


######
# 1x1
######


@frames_comparison
def test_Table1x1_Empty_CL_1(scene):
    t = Table([[""]], col_labels=[Text("C1")])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_Empty_CL_2(scene):
    t = Table([[""]], col_labels=[Text("C1")])
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty_CL_1(scene):
    t = Table([["Cell 1"]], col_labels=[Text("C1")])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty_CL_2(scene):
    t = Table([["Cell 1"]], col_labels=[Text("C1")])
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


######
# 2x2
######


@frames_comparison
def test_Table2x2_Empty_CL_1(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], col_labels=[Text("C1"), Text("C2")]
    )
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_CL_2(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], col_labels=[Text("C1"), Text("C2")]
    )
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_CL_3(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], col_labels=[Text("C1"), Text("C2")]
    )
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_CL_4(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], col_labels=[Text("C1"), Text("C2")]
    )
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_CL_5(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], col_labels=[Text("C1"), Text("C2")]
    )
    t.add_highlighted_cell((3, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_CL_6(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]], col_labels=[Text("C1"), Text("C2")]
    )
    t.add_highlighted_cell((3, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_CL_1(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_CL_2(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_CL_3(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_CL_4(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_CL_5(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((3, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_CL_6(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((3, 2), color=BLUE)
    scene.add(t)


##########################
# With Row and Col Labels
##########################


######
# 1x1
######


@frames_comparison
def test_Table1x1_Empty_RCL_1(scene):
    t = Table([[""]], row_labels=[Text("R1")], col_labels=[Text("C1")])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_Empty_RCL_2(scene):
    t = Table([[""]], row_labels=[Text("R1")], col_labels=[Text("C1")])
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_Empty_RCL_3(scene):
    t = Table([[""]], row_labels=[Text("R1")], col_labels=[Text("C1")])
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_Empty_RCL_4(scene):
    t = Table([[""]], row_labels=[Text("R1")], col_labels=[Text("C1")])
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty_RCL_1(scene):
    t = Table([["Cell 1"]], row_labels=[Text("R1")], col_labels=[Text("C1")])
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty_RCL_2(scene):
    t = Table([["Cell 1"]], row_labels=[Text("R1")], col_labels=[Text("C1")])
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty_RCL_3(scene):
    t = Table([["Cell 1"]], row_labels=[Text("R1")], col_labels=[Text("C1")])
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table1x1_NonEmpty_RCL_4(scene):
    t = Table([["Cell 1"]], row_labels=[Text("R1")], col_labels=[Text("C1")])
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


######
# 2x2
######


@frames_comparison
def test_Table2x2_Empty_RCL_1(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RCL_2(scene):
    t = Table(
        [["", "Cell 2"], ["", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RCL_3(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((1, 3), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RCL_4(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RCL_5(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RCL_6(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((2, 3), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RCL_7(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((3, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RCL_8(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((3, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_Empty_RCL_9(scene):
    t = Table(
        [["", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((3, 3), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_1(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((1, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_2(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((1, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_3(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((1, 3), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_4(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((2, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_5(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((2, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_6(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((2, 3), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_7(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((3, 1), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_8(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((3, 2), color=BLUE)
    scene.add(t)


@frames_comparison
def test_Table2x2_NonEmpty_RCL_9(scene):
    t = Table(
        [["Cell 1", "Cell 2"], ["Cell 3", "Cell 4"]],
        row_labels=[Text("R1"), Text("R2")],
        col_labels=[Text("C1"), Text("C2")],
    )
    t.add_highlighted_cell((3, 3), color=BLUE)
    scene.add(t)


##############################################################################
# tests for add_highlighted_cell function ENDS HERE
##############################################################################
