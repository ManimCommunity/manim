r"""Mobjects representing tables.

Examples
--------

.. manim:: MatrixExamples
    :save_last_frame:

    class TabularExamples(Scene):
            def construct(self):
                t0 = Tabular(
                    [["First", "Second"],
                    ["Third","Fourth"]],
                    row_labels=[Text("R1"), Text("R2")],
                    col_labels=[Text("C1"), Text("C2")],
                    top_left_entry=Text("TOP"))
                t1 = MathTabular(
                    [["+", 0, 5, 10],
                    [0, 0, 5, 10],
                    [2, 2, 7, 12],
                    [4, 4, 9, 14]])
                t1.get_horizontal_lines()[0].set_color(BLUE)
                t1.get_vertical_lines()[0].set_color(BLUE)
                t1.get_horizontal_lines()[0].set_z_index(1)
                t2 = DecimalTabular(
                    [[-2,-1,0,1,2],
                    [3.333, 7.42, 5.0, 2.222, 3.141]],
                    row_labels=[MathTex("x"), MathTex("f(x)")]
                )
                cross = VGroup(
                    Line(UP + LEFT, DOWN + RIGHT),
                    Line(UP + RIGHT, DOWN + LEFT))
                a = Circle().set_color(RED).scale(0.5)
                b = cross.set_color(BLUE).scale(0.5)
                t3 = MobjectTabular(
                    [[a.copy(),b.copy(),a.copy()],
                    [b.copy(),a.copy(),a.copy()],
                    [a.copy(),b.copy(),b.copy()]])
                t3.add(Line(
                    t3.get_corner(DL), t3.get_corner(UR)
                ).set_color(RED))
                g = Group(t0, t1, t2, t3).scale(0.5).arrange_in_grid(buff=1.5)
                self.add(g)
"""

__all__ = [
    "Tabular",
    "MathTabular",
    "MobjectTabular",
    "IntegerTabular",
    "DecimalTabular",
]


import itertools as it

from ..constants import *
from ..mobject.geometry import Line
from ..mobject.numbers import DecimalNumber, Integer
from ..mobject.svg.tex_mobject import MathTex
from ..mobject.svg.text_mobject import Paragraph, Text
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import WHITE


class Tabular(VGroup):
    def __init__(
        self,
        table,
        row_labels=None,
        col_labels=None,
        top_left_entry=None,
        v_buff=0.8,
        h_buff=1.3,
        add_background_rectangles_to_entries=False,
        include_background_rectangle=False,
        element_to_mobject=Paragraph,
        element_to_mobject_config={},
        arrange_in_grid_config={},
        line_config={},
        **kwargs,
    ):
        """

        Parameters
        ----------
        table : :class:`typing.Iterable`
            A 2d array or list of lists
        row_labels : List[:class:`~.Mobject`], optional
            List of Mobjects representing labels of every row
        col_labels : List[:class:`~.Mobject`], optional
            List of Mobjects representing labels of every column
        top_left_entry : :class:`~.Mobject`, optional
            Top-left entry of the table, only possible if row and
            column labels are given
        v_buff : :class:`float`, optional
            vertical buffer, by default 0.8
        h_buff : :class:`float`, optional
            horizontal buffer, by default 1.3
        add_background_rectangles_to_entries : :class:`bool`, optional
            `True` if should add backgraound rectangles to entries, by default False
        include_background_rectangle : :class:`bool`, optional
            `True` if should include background rectangle, by default False
        element_to_mobject : :class:`~.Mobject`, optional
            element to mobject, by default Paragraph
        element_to_mobject_config : Dict[:class:`str`, :class:`~.Mobject`], optional
            element to mobject config, by default {}
        arrange_in_grid_config : Dict[:class:`str`, :class:`~.Mobject`], optional
            dict passed to `arrange_in_grid`, customizes the arrangement of the table
        line_config : Dict[:class:`str`, :class:`~.Mobject`], optional
            dict passed to :class:`~.Line`, customizes the lines of the table

        """

        self.row_labels = row_labels
        self.col_labels = col_labels
        self.top_left_entry = top_left_entry
        self.v_buff = v_buff
        self.h_buff = h_buff
        self.add_background_rectangles_to_entries = add_background_rectangles_to_entries
        self.include_background_rectangle = include_background_rectangle
        self.element_to_mobject = element_to_mobject
        self.element_to_mobject_config = element_to_mobject_config
        self.arrange_in_grid_config = arrange_in_grid_config
        self.line_config = line_config
        VGroup.__init__(self, **kwargs)
        mob_table = self.table_to_mob_table(table)
        self.elements_without_labels = VGroup(*it.chain(*mob_table))
        mob_table = self.add_labels(mob_table)
        self.organize_mob_table(mob_table)
        self.elements = VGroup(*it.chain(*mob_table))
        self.add(self.elements)
        self.center()
        self.mob_table = mob_table
        self.add_horizontal_lines()
        self.add_vertical_lines()
        if self.add_background_rectangles_to_entries:
            for mob in self.elements:
                mob.add_background_rectangle()
        if self.include_background_rectangle:
            self.add_background_rectangle()

    def table_to_mob_table(self, table):
        """Used internally."""
        return [
            [
                self.element_to_mobject(item, **self.element_to_mobject_config)
                for item in row
            ]
            for row in table
        ]

    def organize_mob_table(self, table):
        """Used internally."""
        help_table = VGroup()
        for i, row in enumerate(table):
            for j, _ in enumerate(row):
                help_table.add(table[i][j])
        help_table.arrange_in_grid(
            rows=len(table),
            cols=len(table[0]),
            buff=(self.h_buff, self.v_buff),
            **self.arrange_in_grid_config,
        )
        return help_table

    def add_labels(self, mob_table):
        """Used internally."""
        if self.row_labels is not None:
            for k in range(len(self.row_labels)):
                mob_table[k] = [self.row_labels[k]] + mob_table[k]
        if self.col_labels is not None:
            if self.row_labels is not None:
                if self.top_left_entry is not None:
                    col_labels = [self.top_left_entry] + self.col_labels
                    mob_table.insert(0, col_labels)
                else:
                    col_labels = [VMobject()] + self.col_labels
                    mob_table.insert(0, col_labels)
            else:
                mob_table.insert(0, self.col_labels)
        return mob_table

    def add_horizontal_lines(self):
        """Used internally."""
        anchor_left = self.get_left()[0] - 0.5 * self.h_buff
        anchor_right = self.get_right()[0] + 0.5 * self.h_buff
        line_group = VGroup()
        for k in range(len(self.mob_table) - 1):
            anchor = self.get_rows()[k + 1].get_top()[1] + 0.5 * (
                self.get_rows()[k].get_bottom()[1] - self.get_rows()[k + 1].get_top()[1]
            )
            line = Line(
                [anchor_left, anchor, 0], [anchor_right, anchor, 0], **self.line_config
            )
            line_group.add(line)
            self.add(line)
        self.horizontal_lines = line_group
        return self

    def add_vertical_lines(self):
        """Used internally."""
        anchor_top = self.get_top()[1] + 0.5 * self.v_buff
        anchor_bottom = self.get_bottom()[1] - 0.5 * self.v_buff
        line_group = VGroup()
        for k in range(len(self.mob_table[0]) - 1):
            anchor = self.get_columns()[k + 1].get_left()[0] + 0.5 * (
                self.get_columns()[k].get_right()[0]
                - self.get_columns()[k + 1].get_left()[0]
            )
            line = Line(
                [anchor, anchor_bottom, 0], [anchor, anchor_top, 0], **self.line_config
            )
            line_group.add(line)
            self.add(line)
        self.vertical_lines = line_group
        return self

    def get_horizontal_lines(self):
        """Return the horizontal lines.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing all horizontal lines

        Examples
        --------

        .. manim:: GetHorizontalLinesExample
            :save_last_frame:

            class GetHorizontalLinesExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.get_horizontal_lines().set_color(RED)
                    self.add(table)
        """
        return self.horizontal_lines

    def get_vertical_lines(self):
        """Return the vertical lines.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing all vertical lines

        Examples
        --------

        .. manim:: GetVerticalLinesExample
            :save_last_frame:

            class GetVerticalLinesExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.get_vertical_lines()[0].set_color(RED)
                    self.add(table)
        """
        return self.vertical_lines

    def get_columns(self):
        """Return columns of the table as VGroups.

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a column of the table.

        Examples
        --------

        .. manim:: GetColumnsExample
            :save_last_frame:

            class GetColumnsExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.add(SurroundingRectangle(table.get_columns()[1]))
                    self.add(table)
        """
        return VGroup(
            *[
                VGroup(*[row[i] for row in self.mob_table])
                for i in range(len(self.mob_table[0]))
            ]
        )

    def get_rows(self):
        """Return rows of the table as VGroups.

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a row of the table.

        Examples
        --------

        .. manim:: GetRowsExample
            :save_last_frame:

            class GetRowsExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.add(SurroundingRectangle(table.get_rows()[1]))
                    self.add(table)
        """
        return VGroup(*[VGroup(*row) for row in self.mob_table])

    def set_column_colors(self, *colors):
        columns = self.get_columns()
        for color, column in zip(colors, columns):
            column.set_color(color)
        return self

    def set_row_colors(self, *colors):
        rows = self.get_rows()
        for color, row in zip(colors, rows):
            row.set_color(color)
        return self

    def get_entries(self):
        return self.elements

    def get_entries_without_labels(self):
        return self.elements_without_labels

    def add_background_to_entries(self):
        for mob in self.get_entries():
            mob.add_background_rectangle()
        return self


class MathTabular(Tabular):
    def __init__(
        self,
        table,
        element_to_mobject=MathTex,
        **kwargs,
    ):
        Tabular.__init__(
            self,
            table,
            element_to_mobject=element_to_mobject,
            **kwargs,
        )


class MobjectTabular(Tabular):
    def __init__(self, table, element_to_mobject=lambda m: m, **kwargs):
        Tabular.__init__(self, table, element_to_mobject=element_to_mobject, **kwargs)


class IntegerTabular(Tabular):
    def __init__(self, table, element_to_mobject=Integer, **kwargs):
        Tabular.__init__(self, table, element_to_mobject=element_to_mobject, **kwargs)


class DecimalTabular(Tabular):
    def __init__(
        self,
        table,
        element_to_mobject=DecimalNumber,
        element_to_mobject_config={"num_decimal_places": 1},
        **kwargs,
    ):
        Tabular.__init__(
            self,
            table,
            element_to_mobject=element_to_mobject,
            element_to_mobject_config=element_to_mobject_config,
            **kwargs,
        )
