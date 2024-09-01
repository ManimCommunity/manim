r"""Mobjects representing tables.

Examples
--------

.. manim:: TableExamples
    :save_last_frame:

    class TableExamples(Scene):
        def construct(self):
            t0 = Table(
                [["First", "Second"],
                ["Third","Fourth"]],
                row_labels=[Text("R1"), Text("R2")],
                col_labels=[Text("C1"), Text("C2")],
                top_left_entry=Text("TOP"))
            t0.add_highlighted_cell((2,2), color=GREEN)
            x_vals = np.linspace(-2,2,5)
            y_vals = np.exp(x_vals)
            t1 = DecimalTable(
                [x_vals, y_vals],
                row_labels=[MathTex("x"), MathTex("f(x)")],
                include_outer_lines=True)
            t1.add(t1.get_cell((2,2), color=RED))
            t2 = MathTable(
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
            t3 = MobjectTable(
                [[a.copy(),b.copy(),a.copy()],
                [b.copy(),a.copy(),a.copy()],
                [a.copy(),b.copy(),b.copy()]])
            t3.add(Line(
                t3.get_corner(DL), t3.get_corner(UR)
            ).set_color(RED))
            vals = np.arange(1,21).reshape(5,4)
            t4 = IntegerTable(
                vals,
                include_outer_lines=True
            )
            g1 = Group(t0, t1).scale(0.5).arrange(buff=1).to_edge(UP, buff=1)
            g2 = Group(t2, t3, t4).scale(0.5).arrange(buff=1).to_edge(DOWN, buff=1)
            self.add(g1, g2)
"""

from __future__ import annotations

__all__ = [
    "Table",
    "MathTable",
    "MobjectTable",
    "IntegerTable",
    "DecimalTable",
]


import itertools as it
from collections.abc import Iterable, Sequence
from typing import Callable

from manim.mobject.geometry.line import Line
from manim.mobject.geometry.polygram import Polygon
from manim.mobject.geometry.shape_matchers import BackgroundRectangle
from manim.mobject.text.numbers import DecimalNumber, Integer
from manim.mobject.text.tex_mobject import MathTex
from manim.mobject.text.text_mobject import Paragraph

from ..animation.animation import Animation
from ..animation.composition import AnimationGroup
from ..animation.creation import Create, Write
from ..animation.fading import FadeIn
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import BLACK, YELLOW, ManimColor, ParsableManimColor
from .utils import get_vectorized_mobject_class


class Table(VGroup):
    r"""A mobject that displays a table on the screen.

    Parameters
    ----------
    table
        A 2D array or list of lists. Content of the table has to be a valid input
        for the callable set in ``element_to_mobject``.
    row_labels
        List of :class:`~.VMobject` representing the labels of each row.
    col_labels
        List of :class:`~.VMobject` representing the labels of each column.
    top_left_entry
        The top-left entry of the table, can only be specified if row and
        column labels are given.
    v_buff
        Vertical buffer passed to :meth:`~.Mobject.arrange_in_grid`, by default 0.8.
    h_buff
        Horizontal buffer passed to :meth:`~.Mobject.arrange_in_grid`, by default 1.3.
    include_outer_lines
        ``True`` if the table should include outer lines, by default False.
    add_background_rectangles_to_entries
        ``True`` if background rectangles should be added to entries, by default ``False``.
    entries_background_color
        Background color of entries if ``add_background_rectangles_to_entries`` is ``True``.
    include_background_rectangle
        ``True`` if the table should have a background rectangle, by default ``False``.
    background_rectangle_color
        Background color of table if ``include_background_rectangle`` is ``True``.
    element_to_mobject
        The :class:`~.Mobject` class applied to the table entries. by default :class:`~.Paragraph`. For common choices, see :mod:`~.text_mobject`/:mod:`~.tex_mobject`.
    element_to_mobject_config
        Custom configuration passed to :attr:`element_to_mobject`, by default {}.
    arrange_in_grid_config
        Dict passed to :meth:`~.Mobject.arrange_in_grid`, customizes the arrangement of the table.
    line_config
        Dict passed to :class:`~.Line`, customizes the lines of the table.
    kwargs
        Additional arguments to be passed to :class:`~.VGroup`.

    Examples
    --------

    .. manim:: TableExamples
        :save_last_frame:

        class TableExamples(Scene):
            def construct(self):
                t0 = Table(
                    [["This", "is a"],
                    ["simple", "Table in \\n Manim."]])
                t1 = Table(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    row_labels=[Text("R1"), Text("R2")],
                    col_labels=[Text("C1"), Text("C2")])
                t1.add_highlighted_cell((2,2), color=YELLOW)
                t2 = Table(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    row_labels=[Text("R1"), Text("R2")],
                    col_labels=[Text("C1"), Text("C2")],
                    top_left_entry=Star().scale(0.3),
                    include_outer_lines=True,
                    arrange_in_grid_config={"cell_alignment": RIGHT})
                t2.add(t2.get_cell((2,2), color=RED))
                t3 = Table(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    row_labels=[Text("R1"), Text("R2")],
                    col_labels=[Text("C1"), Text("C2")],
                    top_left_entry=Star().scale(0.3),
                    include_outer_lines=True,
                    line_config={"stroke_width": 1, "color": YELLOW})
                t3.remove(*t3.get_vertical_lines())
                g = Group(
                    t0,t1,t2,t3
                ).scale(0.7).arrange_in_grid(buff=1)
                self.add(g)

    .. manim:: BackgroundRectanglesExample
        :save_last_frame:

        class BackgroundRectanglesExample(Scene):
            def construct(self):
                background = Rectangle(height=6.5, width=13)
                background.set_fill(opacity=.5)
                background.set_color([TEAL, RED, YELLOW])
                self.add(background)
                t0 = Table(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    add_background_rectangles_to_entries=True)
                t1 = Table(
                    [["This", "is a"],
                    ["simple", "Table."]],
                    include_background_rectangle=True)
                g = Group(t0, t1).scale(0.7).arrange(buff=0.5)
                self.add(g)
    """

    def __init__(
        self,
        table: Iterable[Iterable[float | str | VMobject]],
        row_labels: Iterable[VMobject] | None = None,
        col_labels: Iterable[VMobject] | None = None,
        top_left_entry: VMobject | None = None,
        v_buff: float = 0.8,
        h_buff: float = 1.3,
        include_outer_lines: bool = False,
        add_background_rectangles_to_entries: bool = False,
        entries_background_color: ParsableManimColor = BLACK,
        include_background_rectangle: bool = False,
        background_rectangle_color: ParsableManimColor = BLACK,
        element_to_mobject: Callable[
            [float | str | VMobject],
            VMobject,
        ] = Paragraph,
        element_to_mobject_config: dict = {},
        arrange_in_grid_config: dict = {},
        line_config: dict = {},
        **kwargs,
    ):
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.top_left_entry = top_left_entry
        self.row_dim = len(table)
        self.col_dim = len(table[0])
        self.v_buff = v_buff
        self.h_buff = h_buff
        self.include_outer_lines = include_outer_lines
        self.add_background_rectangles_to_entries = add_background_rectangles_to_entries
        self.entries_background_color = ManimColor(entries_background_color)
        self.include_background_rectangle = include_background_rectangle
        self.background_rectangle_color = ManimColor(background_rectangle_color)
        self.element_to_mobject = element_to_mobject
        self.element_to_mobject_config = element_to_mobject_config
        self.arrange_in_grid_config = arrange_in_grid_config
        self.line_config = line_config

        for row in table:
            if len(row) == len(table[0]):
                pass
            else:
                raise ValueError("Not all rows in table have the same length.")

        super().__init__(**kwargs)
        mob_table = self._table_to_mob_table(table)
        self.elements_without_labels = VGroup(*it.chain(*mob_table))
        mob_table = self._add_labels(mob_table)
        self._organize_mob_table(mob_table)
        self.elements = VGroup(*it.chain(*mob_table))

        if len(self.elements[0].get_all_points()) == 0:
            self.elements.remove(self.elements[0])

        self.add(self.elements)
        self.center()
        self.mob_table = mob_table
        self._add_horizontal_lines()
        self._add_vertical_lines()
        if self.add_background_rectangles_to_entries:
            self.add_background_to_entries(color=self.entries_background_color)
        if self.include_background_rectangle:
            self.add_background_rectangle(color=self.background_rectangle_color)

    def _table_to_mob_table(
        self,
        table: Iterable[Iterable[float | str | VMobject]],
    ) -> list:
        """Initilaizes the entries of ``table`` as :class:`~.VMobject`.

        Parameters
        ----------
        table
            A 2D array or list of lists. Content of the table has to be a valid input
            for the callable set in ``element_to_mobject``.

        Returns
        --------
        List
            List of :class:`~.VMobject` from the entries of ``table``.
        """
        return [
            [
                self.element_to_mobject(item, **self.element_to_mobject_config)
                for item in row
            ]
            for row in table
        ]

    def _organize_mob_table(self, table: Iterable[Iterable[VMobject]]) -> VGroup:
        """Arranges the :class:`~.VMobject` of ``table`` in a grid.

        Parameters
        ----------
        table
            A 2D iterable object with :class:`~.VMobject` entries.

        Returns
        --------
        :class:`~.VGroup`
            The :class:`~.VMobject` of the ``table`` in a :class:`~.VGroup` already
            arranged in a table-like grid.
        """
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

    def _add_labels(self, mob_table: VGroup) -> VGroup:
        """Adds labels to an in a grid arranged :class:`~.VGroup`.

        Parameters
        ----------
        mob_table
            An in a grid organized class:`~.VGroup`.

        Returns
        --------
        :class:`~.VGroup`
            Returns the ``mob_table`` with added labels.
        """
        if self.row_labels is not None:
            for k in range(len(self.row_labels)):
                mob_table[k] = [self.row_labels[k]] + mob_table[k]
        if self.col_labels is not None:
            if self.row_labels is not None:
                if self.top_left_entry is not None:
                    col_labels = [self.top_left_entry] + self.col_labels
                    mob_table.insert(0, col_labels)
                else:
                    # Placeholder to use arrange_in_grid if top_left_entry is not set.
                    # Import OpenGLVMobject to work with --renderer=opengl
                    dummy_mobject = get_vectorized_mobject_class()()
                    col_labels = [dummy_mobject] + self.col_labels
                    mob_table.insert(0, col_labels)
            else:
                mob_table.insert(0, self.col_labels)
        return mob_table

    def _add_horizontal_lines(self) -> Table:
        """Adds the horizontal lines to the table."""
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

    def _add_vertical_lines(self) -> Table:
        """Adds the vertical lines to the table"""
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

    def get_horizontal_lines(self) -> VGroup:
        """Return the horizontal lines of the table.

        Returns
        --------
        :class:`~.VGroup`
            :class:`~.VGroup` containing all the horizontal lines of the table.

        Examples
        --------

        .. manim:: GetHorizontalLinesExample
            :save_last_frame:

            class GetHorizontalLinesExample(Scene):
                def construct(self):
                    table = Table(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.get_horizontal_lines().set_color(RED)
                    self.add(table)
        """
        return self.horizontal_lines

    def get_vertical_lines(self) -> VGroup:
        """Return the vertical lines of the table.

        Returns
        --------
        :class:`~.VGroup`
            :class:`~.VGroup` containing all the vertical lines of the table.

        Examples
        --------

        .. manim:: GetVerticalLinesExample
            :save_last_frame:

            class GetVerticalLinesExample(Scene):
                def construct(self):
                    table = Table(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.get_vertical_lines()[0].set_color(RED)
                    self.add(table)
        """
        return self.vertical_lines

    def get_columns(self) -> VGroup:
        """Return columns of the table as a :class:`~.VGroup` of :class:`~.VGroup`.

        Returns
        --------
        :class:`~.VGroup`
            :class:`~.VGroup` containing each column in a :class:`~.VGroup`.

        Examples
        --------

        .. manim:: GetColumnsExample
            :save_last_frame:

            class GetColumnsExample(Scene):
                def construct(self):
                    table = Table(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.add(SurroundingRectangle(table.get_columns()[1]))
                    self.add(table)
        """
        return VGroup(
            *(
                VGroup(*(row[i] for row in self.mob_table))
                for i in range(len(self.mob_table[0]))
            )
        )

    def get_rows(self) -> VGroup:
        """Return the rows of the table as a :class:`~.VGroup` of :class:`~.VGroup`.

        Returns
        --------
        :class:`~.VGroup`
            :class:`~.VGroup` containing each row in a :class:`~.VGroup`.

        Examples
        --------

        .. manim:: GetRowsExample
            :save_last_frame:

            class GetRowsExample(Scene):
                def construct(self):
                    table = Table(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.add(SurroundingRectangle(table.get_rows()[1]))
                    self.add(table)
        """
        return VGroup(*(VGroup(*row) for row in self.mob_table))

    def set_column_colors(self, *colors: Iterable[ParsableManimColor]) -> Table:
        """Set individual colors for each column of the table.

        Parameters
        ----------
        colors
            An iterable of colors; each color corresponds to a column.

        Examples
        --------

        .. manim:: SetColumnColorsExample
            :save_last_frame:

            class SetColumnColorsExample(Scene):
                def construct(self):
                    table = Table(
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

    def set_row_colors(self, *colors: Iterable[ParsableManimColor]) -> Table:
        """Set individual colors for each row of the table.

        Parameters
        ----------
        colors
            An iterable of colors; each color corresponds to a row.

        Examples
        --------

        .. manim:: SetRowColorsExample
            :save_last_frame:

            class SetRowColorsExample(Scene):
                def construct(self):
                    table = Table(
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

    def get_entries(
        self,
        pos: Sequence[int] | None = None,
    ) -> VMobject | VGroup:
        """Return the individual entries of the table (including labels) or one specific entry
        if the parameter, ``pos``,  is set.

        Parameters
        ----------
        pos
            The position of a specific entry on the table. ``(1,1)`` being the top left entry
            of the table.

        Returns
        -------
        Union[:class:`~.VMobject`, :class:`~.VGroup`]
            :class:`~.VGroup` containing all entries of the table (including labels)
            or the :class:`~.VMobject` at the given position if ``pos`` is set.

        Examples
        --------

        .. manim:: GetEntriesExample
            :save_last_frame:

            class GetEntriesExample(Scene):
                def construct(self):
                    table = Table(
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
                index = len(self.mob_table[0]) * (pos[0] - 1) + pos[1] - 2
                return self.elements[index]
            else:
                index = len(self.mob_table[0]) * (pos[0] - 1) + pos[1] - 1
                return self.elements[index]
        else:
            return self.elements

    def get_entries_without_labels(
        self,
        pos: Sequence[int] | None = None,
    ) -> VMobject | VGroup:
        """Return the individual entries of the table (without labels) or one specific entry
        if the parameter, ``pos``, is set.

        Parameters
        ----------
        pos
            The position of a specific entry on the table. ``(1,1)`` being the top left entry
            of the table (without labels).

        Returns
        -------
        Union[:class:`~.VMobject`, :class:`~.VGroup`]
            :class:`~.VGroup` containing all entries of the table (without labels)
            or the :class:`~.VMobject` at the given position if ``pos`` is set.

        Examples
        --------

        .. manim:: GetEntriesWithoutLabelsExample
            :save_last_frame:

            class GetEntriesWithoutLabelsExample(Scene):
                def construct(self):
                    table = Table(
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
            index = self.col_dim * (pos[0] - 1) + pos[1] - 1
            return self.elements_without_labels[index]
        else:
            return self.elements_without_labels

    def get_row_labels(self) -> VGroup:
        """Return the row labels of the table.

        Returns
        -------
        :class:`~.VGroup`
            :class:`~.VGroup` containing the row labels of the table.

        Examples
        --------

        .. manim:: GetRowLabelsExample
            :save_last_frame:

            class GetRowLabelsExample(Scene):
                def construct(self):
                    table = Table(
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

    def get_col_labels(self) -> VGroup:
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
                    table = Table(
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

    def get_labels(self) -> VGroup:
        """Returns the labels of the table.

        Returns
        --------
        :class:`~.VGroup`
            :class:`~.VGroup` containing all the labels of the table.

        Examples
        --------

        .. manim:: GetLabelsExample
            :save_last_frame:

            class GetLabelsExample(Scene):
                def construct(self):
                    table = Table(
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
        for label in (self.col_labels, self.row_labels):
            if label is not None:
                label_group.add(*label)
        return label_group

    def add_background_to_entries(self, color: ParsableManimColor = BLACK) -> Table:
        """Adds a black :class:`~.BackgroundRectangle` to each entry of the table."""
        for mob in self.get_entries():
            mob.add_background_rectangle(color=ManimColor(color))
        return self

    def get_cell(self, pos: Sequence[int] = (1, 1), **kwargs) -> Polygon:
        """Returns one specific cell as a rectangular :class:`~.Polygon` without the entry.

        Parameters
        ----------
        pos
            The position of a specific entry on the table. ``(1,1)`` being the top left entry
            of the table.
        kwargs
            Additional arguments to be passed to :class:`~.Polygon`.

        Returns
        -------
        :class:`~.Polygon`
            Polygon mimicking one specific cell of the Table.

        Examples
        --------

        .. manim:: GetCellExample
            :save_last_frame:

            class GetCellExample(Scene):
                def construct(self):
                    table = Table(
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
        self, pos: Sequence[int] = (1, 1), color: ParsableManimColor = YELLOW, **kwargs
    ) -> BackgroundRectangle:
        """Returns a :class:`~.BackgroundRectangle` of the cell at the given position.

        Parameters
        ----------
        pos
            The position of a specific entry on the table. ``(1,1)`` being the top left entry
            of the table.
        color
            The color used to highlight the cell.
        kwargs
            Additional arguments to be passed to :class:`~.BackgroundRectangle`.

        Examples
        --------

        .. manim:: GetHighlightedCellExample
            :save_last_frame:

            class GetHighlightedCellExample(Scene):
                def construct(self):
                    table = Table(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    highlight = table.get_highlighted_cell((2,2), color=GREEN)
                    table.add_to_back(highlight)
                    self.add(table)
        """
        cell = self.get_cell(pos)
        bg_cell = BackgroundRectangle(cell, color=ManimColor(color), **kwargs)
        return bg_cell

    def add_highlighted_cell(
        self, pos: Sequence[int] = (1, 1), color: ParsableManimColor = YELLOW, **kwargs
    ) -> Table:
        """Highlights one cell at a specific position on the table by adding a :class:`~.BackgroundRectangle`.

        Parameters
        ----------
        pos
            The position of a specific entry on the table. ``(1,1)`` being the top left entry
            of the table.
        color
            The color used to highlight the cell.
        kwargs
            Additional arguments to be passed to :class:`~.BackgroundRectangle`.

        Examples
        --------

        .. manim:: AddHighlightedCellExample
            :save_last_frame:

            class AddHighlightedCellExample(Scene):
                def construct(self):
                    table = Table(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")])
                    table.add_highlighted_cell((2,2), color=GREEN)
                    self.add(table)
        """
        bg_cell = self.get_highlighted_cell(pos, color=ManimColor(color), **kwargs)
        self.add_to_back(bg_cell)
        entry = self.get_entries(pos)
        entry.background_rectangle = bg_cell
        return self

    def create(
        self,
        lag_ratio: float = 1,
        line_animation: Callable[[VMobject | VGroup], Animation] = Create,
        label_animation: Callable[[VMobject | VGroup], Animation] = Write,
        element_animation: Callable[[VMobject | VGroup], Animation] = Create,
        entry_animation: Callable[[VMobject | VGroup], Animation] = FadeIn,
        **kwargs,
    ) -> AnimationGroup:
        """Customized create-type function for tables.

        Parameters
        ----------
        lag_ratio
            The lag ratio of the animation.
        line_animation
            The animation style of the table lines, see :mod:`~.creation` for examples.
        label_animation
            The animation style of the table labels, see :mod:`~.creation` for examples.
        element_animation
            The animation style of the table elements, see :mod:`~.creation` for examples.
        entry_animation
            The entry animation of the table background, see :mod:`~.creation` for examples.
        kwargs
            Further arguments passed to the creation animations.

        Returns
        -------
        :class:`~.AnimationGroup`
            AnimationGroup containing creation of the lines and of the elements.

        Examples
        --------

        .. manim:: CreateTableExample

            class CreateTableExample(Scene):
                def construct(self):
                    table = Table(
                        [["First", "Second"],
                        ["Third","Fourth"]],
                        row_labels=[Text("R1"), Text("R2")],
                        col_labels=[Text("C1"), Text("C2")],
                        include_outer_lines=True)
                    self.play(table.create())
                    self.wait()
        """
        animations: Sequence[Animation] = [
            line_animation(
                VGroup(self.vertical_lines, self.horizontal_lines),
                **kwargs,
            ),
            element_animation(self.elements_without_labels.set_z_index(2), **kwargs),
        ]

        if self.get_labels():
            animations += [
                label_animation(self.get_labels(), **kwargs),
            ]

        if self.get_entries():
            for entry in self.elements_without_labels:
                try:
                    animations += [
                        entry_animation(
                            entry.background_rectangle,
                            **kwargs,
                        )
                    ]
                except AttributeError:
                    continue

        return AnimationGroup(*animations, lag_ratio=lag_ratio)

    def scale(self, scale_factor: float, **kwargs):
        # h_buff and v_buff must be adjusted so that Table.get_cell
        # can construct an accurate polygon for a cell.
        self.h_buff *= scale_factor
        self.v_buff *= scale_factor
        super().scale(scale_factor, **kwargs)
        return self


class MathTable(Table):
    """A specialized :class:`~.Table` mobject for use with LaTeX.

    Examples
    --------

    .. manim:: MathTableExample
        :save_last_frame:

        class MathTableExample(Scene):
            def construct(self):
                t0 = MathTable(
                    [["+", 0, 5, 10],
                    [0, 0, 5, 10],
                    [2, 2, 7, 12],
                    [4, 4, 9, 14]],
                    include_outer_lines=True)
                self.add(t0)
    """

    def __init__(
        self,
        table: Iterable[Iterable[float | str]],
        element_to_mobject: Callable[[float | str], VMobject] = MathTex,
        **kwargs,
    ):
        """
        Special case of :class:`~.Table` with `element_to_mobject` set to :class:`~.MathTex`.
        Every entry in `table` is set in a Latex `align` environment.

        Parameters
        ----------
        table
            A 2d array or list of lists. Content of the table have to be valid input
            for :class:`~.MathTex`.
        element_to_mobject
            The :class:`~.Mobject` class applied to the table entries. Set as :class:`~.MathTex`.
        kwargs
            Additional arguments to be passed to :class:`~.Table`.
        """
        super().__init__(
            table,
            element_to_mobject=element_to_mobject,
            **kwargs,
        )


class MobjectTable(Table):
    """A specialized :class:`~.Table` mobject for use with :class:`~.Mobject`.

    Examples
    --------

    .. manim:: MobjectTableExample
        :save_last_frame:

        class MobjectTableExample(Scene):
            def construct(self):
                cross = VGroup(
                    Line(UP + LEFT, DOWN + RIGHT),
                    Line(UP + RIGHT, DOWN + LEFT),
                )
                a = Circle().set_color(RED).scale(0.5)
                b = cross.set_color(BLUE).scale(0.5)
                t0 = MobjectTable(
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
        table: Iterable[Iterable[VMobject]],
        element_to_mobject: Callable[[VMobject], VMobject] = lambda m: m,
        **kwargs,
    ):
        """
        Special case of :class:`~.Table` with ``element_to_mobject`` set to an identity function.
        Here, every item in ``table`` must already be of type :class:`~.Mobject`.

        Parameters
        ----------
        table
            A 2D array or list of lists. Content of the table must be of type :class:`~.Mobject`.
        element_to_mobject
            The :class:`~.Mobject` class applied to the table entries. Set as ``lambda m : m`` to return itself.
        kwargs
            Additional arguments to be passed to :class:`~.Table`.
        """
        super().__init__(table, element_to_mobject=element_to_mobject, **kwargs)


class IntegerTable(Table):
    r"""A specialized :class:`~.Table` mobject for use with :class:`~.Integer`.

    Examples
    --------

    .. manim:: IntegerTableExample
        :save_last_frame:

        class IntegerTableExample(Scene):
            def construct(self):
                t0 = IntegerTable(
                    [[0,30,45,60,90],
                    [90,60,45,30,0]],
                    col_labels=[
                        MathTex(r"\frac{\sqrt{0}}{2}"),
                        MathTex(r"\frac{\sqrt{1}}{2}"),
                        MathTex(r"\frac{\sqrt{2}}{2}"),
                        MathTex(r"\frac{\sqrt{3}}{2}"),
                        MathTex(r"\frac{\sqrt{4}}{2}")],
                    row_labels=[MathTex(r"\sin"), MathTex(r"\cos")],
                    h_buff=1,
                    element_to_mobject_config={"unit": r"^{\circ}"})
                self.add(t0)
    """

    def __init__(
        self,
        table: Iterable[Iterable[float | str]],
        element_to_mobject: Callable[[float | str], VMobject] = Integer,
        **kwargs,
    ):
        """
        Special case of :class:`~.Table` with `element_to_mobject` set to :class:`~.Integer`.
        Will round if there are decimal entries in the table.

        Parameters
        ----------
        table
            A 2d array or list of lists. Content of the table has to be valid input
            for :class:`~.Integer`.
        element_to_mobject
            The :class:`~.Mobject` class applied to the table entries. Set as :class:`~.Integer`.
        kwargs
            Additional arguments to be passed to :class:`~.Table`.
        """
        super().__init__(table, element_to_mobject=element_to_mobject, **kwargs)


class DecimalTable(Table):
    """A specialized :class:`~.Table` mobject for use with :class:`~.DecimalNumber` to display decimal entries.

    Examples
    --------

    .. manim:: DecimalTableExample
        :save_last_frame:

        class DecimalTableExample(Scene):
            def construct(self):
                x_vals = [-2,-1,0,1,2]
                y_vals = np.exp(x_vals)
                t0 = DecimalTable(
                    [x_vals, y_vals],
                    row_labels=[MathTex("x"), MathTex("f(x)=e^{x}")],
                    h_buff=1,
                    element_to_mobject_config={"num_decimal_places": 2})
                self.add(t0)
    """

    def __init__(
        self,
        table: Iterable[Iterable[float | str]],
        element_to_mobject: Callable[[float | str], VMobject] = DecimalNumber,
        element_to_mobject_config: dict = {"num_decimal_places": 1},
        **kwargs,
    ):
        """
        Special case of :class:`~.Table` with ``element_to_mobject`` set to :class:`~.DecimalNumber`.
        By default, ``num_decimal_places`` is set to 1.
        Will round/truncate the decimal places based on the provided ``element_to_mobject_config``.

        Parameters
        ----------
        table
            A 2D array, or a list of lists. Content of the table must be valid input
            for :class:`~.DecimalNumber`.
        element_to_mobject
            The :class:`~.Mobject` class applied to the table entries. Set as :class:`~.DecimalNumber`.
        element_to_mobject_config
            Element to mobject config, here set as {"num_decimal_places": 1}.
        kwargs
            Additional arguments to be passed to :class:`~.Table`.
        """
        super().__init__(
            table,
            element_to_mobject=element_to_mobject,
            element_to_mobject_config=element_to_mobject_config,
            **kwargs,
        )
