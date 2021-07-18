r"""Mobjects representing tables.

Examples
--------

.. manim:: TabularExamples
    :save_last_frame:

    class TabularExamples(Scene):
        def construct(self):
            t0 = Tabular(
                [["First", "Second"],
                ["Third","Fourth"]],
                row_labels=[Text("R1"), Text("R2")],
                col_labels=[Text("C1"), Text("C2")],
                top_left_entry=Text("TOP"))
            t0.add(t0.get_highlighted_cell((2,2), color=GREEN))
            x_vals = np.linspace(-2,2,5)
            y_vals = np.exp(x_vals)
            t1 = DecimalTabular(
                [x_vals, y_vals],
                row_labels=[MathTex("x"), MathTex("f(x)")],
                include_outer_lines=True)
            t1.add(t1.get_cell((2,2), color=RED))
            t2 = MathTabular(
                [["+", 0, 5, 10],
                [0, 0, 5, 10],
                [2, 2, 7, 12],
                [4, 4, 9, 14]],
                include_outer_lines=True)
            t2.get_horizontal_lines()[:3].set_color(BLUE)
            t2.get_vertical_lines()[:3].set_color(BLUE)
            t2.get_horizontal_lines()[:3].set_z_index(1)
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
            vals = np.arange(1,21).reshape(5,4)
            t4 = IntegerTabular(
                vals,
                include_outer_lines=True
            )
            g1 = Group(t0, t1).scale(0.5).arrange(buff=1).to_edge(UP, buff=1)
            g2 = Group(t2, t3, t4).scale(0.5).arrange(buff=1).to_edge(DOWN, buff=1)
            self.add(g1, g2)
"""

__all__ = [
    "Tabular",
    "MathTabular",
    "MobjectTabular",
    "IntegerTabular",
    "DecimalTabular",
]


import itertools as it
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type, Union

from colour import Color

from ..animation.composition import AnimationGroup
from ..animation.creation import *
from ..constants import *
from ..mobject.geometry import Line, Polygon
from ..mobject.numbers import DecimalNumber, Integer
from ..mobject.shape_matchers import BackgroundRectangle
from ..mobject.svg.tex_mobject import MathTex
from ..mobject.svg.text_mobject import Paragraph, Text
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import WHITE, YELLOW, Colors


class Tabular(VGroup):
    """A mobject that displays a table on the screen.

    Examples
    --------

    .. manim:: TabularExamples
        :save_last_frame:

        class TabularExamples(Scene):
            def construct(self):
                t0 = Tabular(
                    [["This", "is a"],
                    ["simple", "Table in \\n Manim."]])
                t1 = Tabular(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    row_labels=[Text("R1"), Text("R2")],
                    col_labels=[Text("C1"), Text("C2")])
                t1.add(t1.get_highlighted_cell((2,2), color=GREEN))
                t2 = Tabular(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    row_labels=[Text("R1"), Text("R2")],
                    col_labels=[Text("C1"), Text("C2")],
                    top_left_entry=Star().scale(0.3),
                    include_outer_lines=True,
                    arrange_in_grid_config={"cell_alignment": RIGHT})
                t2.add(t2.get_cell((2,2), color=RED))
                t3 = Tabular(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    row_labels=[Text("R1"), Text("R2")],
                    col_labels=[Text("C1"), Text("C2")],
                    top_left_entry=Star().scale(0.3),
                    include_outer_lines=True,
                    line_config={"stroke_width": 1, "color": YELLOW})
                t3.remove(*[line for line in t3.get_vertical_lines()])
                g = Group(
                    t0,t1,t2,t3
                ).scale(0.7).arrange_in_grid(buff=1)
                self.add(g)

    .. manim:: BackgroundRectanglesExample
        :save_last_frame:

        class BackgroundRectanglesExample(Scene):
            def construct(self):
                background = Rectangle().scale(3.2)
                background.set_fill(opacity=.5)
                background.set_color([TEAL, RED, YELLOW])
                self.add(background)
                t0 = Tabular(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    add_background_rectangles_to_entries=True)
                t1 = Tabular(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    include_background_rectangle=True)
                g = Group(t0, t1).scale(0.7).arrange(buff=0.5)
                self.add(g)
    """

    def __init__(
        self,
        table: Iterable[Iterable[Union[float, str, "VMobject"]]],
        row_labels: Optional[Iterable["VMobject"]] = None,
        col_labels: Optional[Iterable["VMobject"]] = None,
        top_left_entry: Optional["VMobject"] = None,
        v_buff: float = 0.8,
        h_buff: float = 1.3,
        include_outer_lines: Optional[bool] = False,
        add_background_rectangles_to_entries: Optional[bool] = False,
        include_background_rectangle: Optional[bool] = False,
        element_to_mobject: Callable[
            [Union[float, str, "VMobject"]], "VMobject"
        ] = Paragraph,
        element_to_mobject_config: Optional[dict] = {},
        arrange_in_grid_config: Optional[dict] = {},
        line_config: Optional[dict] = {},
        **kwargs,
    ):
        """
        Parameters
        ----------
        table
            A 2d array or list of lists. Content of the table has to be a valid input
            for the callable set in `element_to_mobject`.
        row_labels
            List of :class:`~.VMobject` representing labels of every row.
        col_labels
            List of :class:`~.VMobject` representing labels of every column.
        top_left_entry
            Top-left entry of the table, only possible if row and
            column labels are given.
        v_buff
            Vertical buffer passed to :meth:`~.Mobject.arrange_in_grid`, by default 0.8.
        h_buff
            Horizontal buffer passed to :meth:`~.Mobject.arrange_in_grid`, by default 1.3.
        include_outer_lines
            `True` if should include outer lines, by default False.
        add_background_rectangles_to_entries
            `True` if should add backgraound rectangles to entries, by default False.
        include_background_rectangle
            `True` if should include background rectangle, by default False.
        element_to_mobject
            Element to mobject, by default :class:`~.Paragraph`. For common choices, see :mod:`~.text_mobject`
            and :mod:`~.tex_mobject`.
        element_to_mobject_config
            Element to mobject config, by default {}.
        arrange_in_grid_config
            Dict passed to :meth:`~.Mobject.arrange_in_grid`, customizes the arrangement of the table.
        line_config
            Dict passed to :class:`~.Line`, customizes the lines of the table.
        kwargs : Any
            Additional arguments to be passed to :class:`~.VGroup`.
        """

        self.row_labels = row_labels
        self.col_labels = col_labels
        self.top_left_entry = top_left_entry
        self.row_dim = len(table)
        self.col_dim = len(table[0])
        self.v_buff = v_buff
        self.h_buff = h_buff
        self.include_outer_lines = include_outer_lines
        self.add_background_rectangles_to_entries = add_background_rectangles_to_entries
        self.include_background_rectangle = include_background_rectangle
        self.element_to_mobject = element_to_mobject
        self.element_to_mobject_config = element_to_mobject_config
        self.arrange_in_grid_config = arrange_in_grid_config
        self.line_config = line_config
        for row in table:
            if len(row) == len(table[0]):
                pass
            else:
                raise ValueError("Not all rows in table have the same length.")
        VGroup.__init__(self, **kwargs)
        mob_table = self.table_to_mob_table(table)
        self.elements_without_labels = VGroup(*it.chain(*mob_table))
        mob_table = self.add_labels(mob_table)
        self.organize_mob_table(mob_table)
        self.elements = VGroup(*it.chain(*mob_table))
        if len(self.elements[0].get_all_points()) == 0:
            self.elements.remove(self.elements[0])
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
                    dummy_mobject = VMobject()
                    col_labels = [dummy_mobject] + self.col_labels
                    mob_table.insert(0, col_labels)
            else:
                mob_table.insert(0, self.col_labels)
        return mob_table

    def add_horizontal_lines(self):
        """Used internally."""
        anchor_left = self.get_left()[0] - 0.5 * self.h_buff
        anchor_right = self.get_right()[0] + 0.5 * self.h_buff
        line_group = VGroup()
        if self.include_outer_lines:
            anchor = self.get_rows()[0].get_top()[1] + 0.5 * self.v_buff
            line = Line(
                [anchor_left, anchor, 0], [anchor_right, anchor, 0], **self.line_config
            )
            line_group.add(line)
            self.add(line)
            anchor = self.get_rows()[-1].get_bottom()[1] - 0.5 * self.v_buff
            line = Line(
                [anchor_left, anchor, 0], [anchor_right, anchor, 0], **self.line_config
            )
            line_group.add(line)
            self.add(line)
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
        anchor_top = self.get_rows().get_top()[1] + 0.5 * self.v_buff
        anchor_bottom = self.get_rows().get_bottom()[1] - 0.5 * self.v_buff
        line_group = VGroup()
        if self.include_outer_lines:
            anchor = self.get_columns()[0].get_left()[0] - 0.5 * self.h_buff
            line = Line(
                [anchor, anchor_top, 0], [anchor, anchor_bottom, 0], **self.line_config
            )
            line_group.add(line)
            self.add(line)
            anchor = self.get_columns()[-1].get_right()[0] + 0.5 * self.h_buff
            line = Line(
                [anchor, anchor_top, 0], [anchor, anchor_bottom, 0], **self.line_config
            )
            line_group.add(line)
            self.add(line)
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

    def get_horizontal_lines(self) -> "VGroup":
        """Return the horizontal lines.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing all horizontal lines.

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

    def get_vertical_lines(self) -> "VGroup":
        """Return the vertical lines.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing all vertical lines.

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

    def get_columns(self) -> List["VGroup"]:
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

    def get_rows(self) -> List["VGroup"]:
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

    def set_column_colors(self, *colors: List[Colors]) -> "Tabular":
        """Set individual colors for each column of the table.

        Parameters
        ----------
        colors
            The list of colors; each color specified corresponds to a column.

        Returns
        -------
        :class:`Tabular`
            The current table object (self).

        Examples
        --------

        .. manim:: SetColumnColorsExample
            :save_last_frame:

            class SetColumnColorsExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")]
                    ).set_column_colors([RED,BLUE], GREEN)
                    self.add(table)
        """
        columns = self.get_columns()
        for color, column in zip(colors, columns):
            column.set_color(color)
        return self

    def set_row_colors(self, *colors: List[Color]) -> "Tabular":
        """Set individual colors for each row of the table.

        Parameters
        ----------
        colors
            The list of colors; each color specified corresponds to a row.

        Returns
        -------
        :class:`Tabular`
            The current table object (self).

        Examples
        --------

        .. manim:: SetRowColorsExample
            :save_last_frame:

            class SetRowColorsExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")]
                    ).set_row_colors([RED,BLUE], GREEN)
                    self.add(table)
        """
        rows = self.get_rows()
        for color, row in zip(colors, rows):
            row.set_color(color)
        return self

    def get_entries(self, pos: Optional[Tuple[int, int]] = None) -> "VMobject":
        """Return the individual entries of the table (including labels) or one specific single entry
        if the position parameter is set.

        Parameters
        ----------
        pos
            The desired position as an iterable tuple, (1,1) being the top left entry
            of the table.

        Returns
        --------
        :class:`~.VMobject`
            VGroup containing all entries of the table (including labels)
            or the VMobject at the given position if `pos` is set.

        Examples
        --------

        .. manim:: GetEntriesExample
            :save_last_frame:

            class GetEntriesExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    ent = table.get_entries()
                    for item in ent:
                        item.set_color(random_bright_color())
                    table.get_entries((2,2)).rotate(PI)
                    self.add(table)
        """
        if pos is not None:
            if (
                self.row_labels is not None
                and self.col_labels is not None
                and self.top_left_entry is None
            ):
                index = len(self.mob_table) * (pos[0] - 1) + pos[1] - 2
                return self.elements[index]
            else:
                index = len(self.mob_table) * (pos[0] - 1) + pos[1] - 1
                return self.elements[index]
        else:
            return self.elements

    def get_entries_without_labels(
        self, pos: Optional[Tuple[int, int]] = None
    ) -> "VMobject":
        """Return the individual entries of the table (without labels) or one specific single entry
        if the position parameter is set.

        Parameters
        ----------
        pos
            The desired position as an iterable tuple, (1,1) being the top left entry
            of the table (without labels).

        Returns
        --------
        :class:`~.VMobject`
            VGroup containing all entries of the table (without labels)
            or the VMobject at the given position if `pos` is set.

        Examples
        --------

        .. manim:: GetEntriesWithoutLabelsExample
            :save_last_frame:

            class GetEntriesWithoutLabelsExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    ent = table.get_entries_without_labels()
                    colors = [BLUE, GREEN, YELLOW, RED]
                    for k in range(len(colors)):
                        ent[k].set_color(colors[k])
                    table.get_entries_without_labels((2,2)).rotate(PI)
                    self.add(table)
        """
        if pos is not None:
            index = self.row_dim * (pos[0] - 1) + pos[1] - 1
            return self.elements_without_labels[index]
        else:
            return self.elements_without_labels

    def get_row_labels(self) -> "VGroup":
        """Return the row labels of the table.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing the row labels of the table.

        Examples
        --------

        .. manim:: GetRowLabelsExample
            :save_last_frame:

            class GetRowLabelsExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    lab = table.get_row_labels()
                    for item in lab:
                        item.set_color(random_bright_color())
                    self.add(table)
        """

        return VGroup(*self.row_labels)

    def get_col_labels(self) -> "VGroup":
        """Return the column labels of the table.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing the column labels of the table.

        Examples
        --------

        .. manim:: GetColLabelsExample
            :save_last_frame:

            class GetColLabelsExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    lab = table.get_col_labels()
                    for item in lab:
                        item.set_color(random_bright_color())
                    self.add(table)
        """

        return VGroup(*self.col_labels)

    def get_labels(self) -> "VGroup":
        """Returns the labels of the table.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing all labels of the table.

        Examples
        --------

        .. manim:: GetLabelsExample
            :save_last_frame:

            class GetLabelsExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    lab = table.get_labels()
                    colors = [BLUE, GREEN, YELLOW, RED]
                    for k in range(len(colors)):
                        lab[k].set_color(colors[k])
                    self.add(table)
        """
        label_group = VGroup()
        if self.top_left_entry is not None:
            label_group.add(self.top_left_entry)
        for label in [self.col_labels, self.row_labels]:
            if label is not None:
                label_group.add(*label)
        return label_group

    def add_background_to_entries(self) -> "Tabular":
        """Add a black background rectangle to the table, see above for an
        example and comparison with `include_background_rectangle`.

        Returns
        -------
        :class:`Tabular`
            The current tabular object (self).
        """
        for mob in self.get_entries():
            mob.add_background_rectangle()
        return self

    def get_cell(self, pos: Tuple[int, int] = (1, 1), **kwargs) -> "Polygon":
        """Return one specific single cell as a rectangular :class:`~.Polygon` without the entry.

        Parameters
        ----------
        pos
            The desired position of the cell as an iterable tuple, (1,1) being the top left entry
            of the table.
        kwargs : Any
            Additional arguments to be passed to :class:`~.Polygon`.

        Returns
        --------
        :class:`~.Polygon`
            Polygon mimicking one specific cell of the tabular.

        Examples
        --------

        .. manim:: GetCellExample
            :save_last_frame:

            class GetCellExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    cell = table.get_cell((2,2), color=RED)
                    self.add(table, cell)
        """
        row = self.get_rows()[pos[0] - 1]
        col = self.get_columns()[pos[1] - 1]
        edge_UL = [
            col.get_left()[0] - self.h_buff / 2,
            row.get_top()[1] + self.v_buff / 2,
            0,
        ]
        edge_UR = [
            col.get_right()[0] + self.h_buff / 2,
            row.get_top()[1] + self.v_buff / 2,
            0,
        ]
        edge_DL = [
            col.get_left()[0] - self.h_buff / 2,
            row.get_bottom()[1] - self.v_buff / 2,
            0,
        ]
        edge_DR = [
            col.get_right()[0] + self.h_buff / 2,
            row.get_bottom()[1] - self.v_buff / 2,
            0,
        ]
        rec = Polygon(edge_UL, edge_UR, edge_DR, edge_DL, **kwargs)
        return rec

    def get_highlighted_cell(
        self, pos: Tuple[int, int] = (1, 1), color: Colors = YELLOW, **kwargs
    ) -> "BackgroundRectangle":
        """Return one highlighter of one cell.

        Parameters
        ----------
        pos
            The desired position of the cell as an iterable tuple, (1,1) being the top left entry
            of the table.
        color
            The color used to highlight the cell.
        kwargs : Any
            Additional arguments to be passed to :meth:`~.add_background_rectangle`.

        Returns
        --------
        :class:`~.BackgroundRectangle`
            Background rectangle of the given cell.

        Examples
        --------

        .. manim:: GetHighlightedCellExample
            :save_last_frame:

            class GetHighlightedCellExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    h_cell = table.get_highlighted_cell((2,2), color=GREEN)
                    self.add(table, h_cell)
        """
        cell = self.get_cell(pos)
        cell.add_background_rectangle(color=color, **kwargs)
        return cell[1].set_z_index(-1)

    def create(
        self,
        run_time: float = 1,
        lag_ratio: float = 1,
        line_animation: Callable[["VGroup"], None] = Create,
        label_animation: Callable[["VGroup"], None] = Write,
        element_animation: Callable[["VGroup"], None] = Write,
        **kwargs,
    ) -> "AnimationGroup":
        """Customized create-type function for tables.

        Parameters
        ----------
        run_time
            The run time of the line creation and the writing of the elements.
        lag_ratio
            The lag ratio of the animation.
        line_animation
            The animation style of the table lines, see :mod:`~.creation` for examples.
        label_animation
            The animation style of the table labels, see :mod:`~.creation` for examples.
        element_animation
            The animation style of the table elements, see :mod:`~.creation` for examples.
        kwargs : Any
            Further arguments passed to the creation animations.

        Returns
        --------
        :class:`~.AnimationGroup`
            AnimationGroup containing creation of the lines and of the elements.

        Examples
        --------

        .. manim:: CreateTableExample

            class CreateTableExample(Scene):
                def construct(self):
                    table = Tabular(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")],
                        include_outer_lines=True)
                    self.play(table.create())
        """
        if len(self.get_labels()) > 0:
            animations = [
                line_animation(
                    VGroup(self.vertical_lines, self.horizontal_lines),
                    run_time=run_time,
                    **kwargs,
                ),
                label_animation(self.get_labels(), run_time=run_time, **kwargs),
                element_animation(
                    self.elements_without_labels, run_time=run_time, **kwargs
                ),
            ]
        else:
            animations = [
                line_animation(
                    VGroup(self.vertical_lines, self.horizontal_lines),
                    run_time=run_time,
                    **kwargs,
                ),
                element_animation(self.elements, run_time=run_time, **kwargs),
            ]
        # if len(self.get_labels()) > 0:
        #     animations.insert(1, label_animation(self.get_labels(), run_time=run_time, **kwargs))
        return AnimationGroup(*animations, lag_ratio=lag_ratio)


class MathTabular(Tabular):
    """A mobject that displays a table with Latex entries on the screen.

    Examples
    --------

    .. manim:: MathTabularExample
        :save_last_frame:

        class MathTabularExample(Scene):
            def construct(self):
                t0 = MathTabular(
                    [["+", 0, 5, 10],
                    [0, 0, 5, 10],
                    [2, 2, 7, 12],
                    [4, 4, 9, 14]],
                    include_outer_lines=True)
                self.add(t0)
    """

    def __init__(
        self,
        table: Iterable[Iterable[Union[float, str]]],
        element_to_mobject: Callable[[Union[float, str]], "VMobject"] = MathTex,
        **kwargs,
    ):
        """
        Special case of :class:`~.Tabular` with `element_to_mobject` set to :class:`~.MathTex`.
        Every entry in `table` is set in a Latex `align` environment.

        Parameters
        ----------
        table
            A 2d array or list of lists. Content of the table have to be valid input
            for :class:`~.MathTex`.
        element_to_mobject
            Element to mobject, here set as :class:`~.MathTex`.
        kwargs : Any
            Additional arguments to be passed to :class:`~.Tabular`.
        """
        Tabular.__init__(
            self,
            table,
            element_to_mobject=element_to_mobject,
            **kwargs,
        )


class MobjectTabular(Tabular):
    """A mobject that displays a table with mobject entries on the screen.

    Examples
    --------

    .. manim:: MobjectTabularExample
        :save_last_frame:

        class MobjectTabularExample(Scene):
            def construct(self):
                cross = VGroup(
                    Line(UP + LEFT, DOWN + RIGHT),
                    Line(UP + RIGHT, DOWN + LEFT),
                )
                a = Circle().set_color(RED).scale(0.5)
                b = cross.set_color(BLUE).scale(0.5)
                t0 = MobjectTabular(
                    [[a.copy(),b.copy(),a.copy()],
                    [b.copy(),a.copy(),a.copy()],
                    [a.copy(),b.copy(),b.copy()]]
                )
                line = Line(
                    t0.get_corner(DL), t0.get_corner(UR)
                ).set_color(RED)
                self.add(t0, line)
    """

    def __init__(
        self,
        table: Iterable[Iterable["VMobject"]],
        element_to_mobject: Callable[["VMobject"], "VMobject"] = lambda m: m,
        **kwargs,
    ):
        """
        Special case of :class:`~.Tabular` with `element_to_mobject` set to an identity function.
        Here, every item in `table` has to be of type :class:`~.Mobject` already.

        Parameters
        ----------
        table
            A 2d array or list of lists. Content of the table have to be of type :class:`~.Mobject`.
        element_to_mobject
            Element to mobject, here set as the identity `lambda m: m`.
        kwargs : Any
            Additional arguments to be passed to :class:`~.Tabular`.
        """
        Tabular.__init__(self, table, element_to_mobject=element_to_mobject, **kwargs)


class IntegerTabular(Tabular):
    """A mobject that displays a table with integer entries on the screen.

    Examples
    --------

    .. manim:: IntegerTabularExample
        :save_last_frame:

        class IntegerTabularExample(Scene):
            def construct(self):
                t0 = IntegerTabular(
                    [[0,30,45,60,90],
                    [90,60,45,30,0]],
                    col_labels=[
                        MathTex("\\\\frac{\\sqrt{0}}{2}"),
                        MathTex("\\\\frac{\\sqrt{1}}{2}"),
                        MathTex("\\\\frac{\\sqrt{2}}{2}"),
                        MathTex("\\\\frac{\\sqrt{3}}{2}"),
                        MathTex("\\\\frac{\\sqrt{4}}{2}")],
                    row_labels=[MathTex("\\sin"), MathTex("\\cos")],
                    h_buff=1,
                    element_to_mobject_config={"unit": "^{\\circ}"})
                self.add(t0)
    """

    def __init__(
        self,
        table: Iterable[Iterable[Union[float, str]]],
        element_to_mobject: Callable[[Union[float, str]], "VMobject"] = Integer,
        **kwargs,
    ):
        """
        Special case of :class:`~.Tabular` with `element_to_mobject` set to :class:`~.Integer`.
        Will round if there are decimal entries in the table.

        Parameters
        ----------
        table
            A 2d array or list of lists. Content of the table has to be valid input
            for :class:`~.Integer`.
        element_to_mobject
            Element to mobject, here set as :class:`~.Integer`.
        kwargs : Any
            Additional arguments to be passed to :class:`~.Tabular`.
        """
        Tabular.__init__(self, table, element_to_mobject=element_to_mobject, **kwargs)


class DecimalTabular(Tabular):
    """A mobject that displays a table with decimal entries on the screen.

    Examples
    --------

    .. manim:: DecimalTabularExample
        :save_last_frame:

        class DecimalTabularExample(Scene):
            def construct(self):
                x_vals = [-2,-1,0,1,2]
                y_vals = np.exp(x_vals)
                t0 = DecimalTabular(
                    [x_vals, y_vals],
                    row_labels=[MathTex("x"), MathTex("f(x)=e^{x}")],
                    h_buff=1,
                    element_to_mobject_config={"num_decimal_places": 2})
                self.add(t0)
    """

    def __init__(
        self,
        table: Iterable[Iterable[Union[float, str]]],
        element_to_mobject: Callable[[Union[float, str]], "VMobject"] = DecimalNumber,
        element_to_mobject_config: dict = {"num_decimal_places": 1},
        **kwargs,
    ):
        """
        Special case of :class:`~.Tabular` with `element_to_mobject` set to :class:`~.DecimalNumber`.
        By default, `num_decimal_places` is set to 1.
        Will round/truncate the decimal places as per the provided config.

        Parameters
        ----------
        table
            A 2d array or list of lists. Content of the table has to be valid input
            for :class:`~.DecimalNumber`.
        element_to_mobject
            Element to mobject, here set as :class:`~.DecimalNumber`.
        element_to_mobject_config
            Element to mobject config, here set as {"num_decimal_places": 1}.
        kwargs : Any
            Additional arguments to be passed to :class:`~.Tabular`.
        """
        Tabular.__init__(
            self,
            table,
            element_to_mobject=element_to_mobject,
            element_to_mobject_config=element_to_mobject_config,
            **kwargs,
        )
