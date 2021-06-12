r"""Mobjects representing table."""

__all__ = [
    "Tabular"
]


import itertools as it

from ..constants import *
from ..mobject.geometry import Line
from ..mobject.numbers import DecimalNumber, Integer
from ..mobject.svg.tex_mobject import MathTex, Tex
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import WHITE

from manim.mobject.types.vectorized_mobject import VGroup


class Tabular(VGroup):
    def __init__(
        self,
        table,
        row_labels=None,
        col_labels=None,
        top_left_entry=None,
        v_buff=0.8,
        h_buff=1.3,
        bracket_h_buff=MED_SMALL_BUFF,
        bracket_v_buff=MED_SMALL_BUFF,
        add_background_rectangles_to_entries=False,
        include_background_rectangle=False,
        element_to_mobject=MathTex,
        element_to_mobject_config={},
        element_alignment_corner=DR,
        **kwargs,
    ):

        self.row_labels = row_labels
        self.col_labels = col_labels
        self.top_left_entry = top_left_entry
        self.v_buff = v_buff
        self.h_buff = h_buff
        self.bracket_h_buff = bracket_h_buff
        self.bracket_v_buff = bracket_v_buff
        self.add_background_rectangles_to_entries = add_background_rectangles_to_entries
        self.include_background_rectangle = include_background_rectangle
        self.element_to_mobject = element_to_mobject
        self.element_to_mobject_config = element_to_mobject_config
        self.element_alignment_corner = element_alignment_corner
        VGroup.__init__(self, **kwargs)
        mob_table = self.table_to_mob_table(table)
        # add lables
        mob_table = self.add_labels(mob_table, row_labels, col_labels, top_left_entry)
        self.organize_mob_table(mob_table)
        self.elements = VGroup(*it.chain(*mob_table))
        self.add(self.elements)
        self.center()
        self.mob_table = mob_table
        self.add_horitontal_lines()
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
        help_table.arrange_in_grid(buff=(self.h_buff, self.v_buff))
        return help_table

    def add_labels(self, mob_table, row_labels, col_labels, top_left_entry):
        if row_labels is not None:
            for k in range(len(row_labels)):
                mob_table[k] = [row_labels[k]] + mob_table[k]
        if col_labels is not None:
            if row_labels is not None:
                if top_left_entry is not None:
                    col_labels = [top_left_entry] + col_labels
                    mob_table.insert(0,col_labels)
                else:
                    col_labels = [VMobject()] + col_labels
                    mob_table.insert(0,col_labels)
            else:
                mob_table.insert(0,col_labels)
        return mob_table

    def add_horitontal_lines(self):
        anchor_left = self.get_left()[0] - 0.5 * self.h_buff
        anchor_right = self.get_right()[0] + 0.5 * self.h_buff
        line_group = VGroup()
        for k in range(len(self.mob_table)-1):
            anchor = self.get_rows()[k+1].get_top()[1] + 0.5 * ( self.get_rows()[k].get_bottom()[1] - self.get_rows()[k+1].get_top()[1] )
            line = Line([anchor_left, anchor, 0], [anchor_right, anchor, 0])
            line_group.add(line)
            self.add(line)
        self.horizontal_lines = line_group
        return self

    def add_vertical_lines(self):
        anchor_top = self.get_top()[1] + 0.5 * self.v_buff
        anchor_bottom = self.get_bottom()[1] - 0.5 * self.v_buff
        line_group = VGroup()
        for k in range(len(self.mob_table[0])-1):
            anchor = self.get_columns()[k+1].get_left()[0] + 0.5 * ( self.get_columns()[k].get_right()[0] - self.get_columns()[k+1].get_left()[0] )
            line = Line([anchor, anchor_bottom, 0], [anchor, anchor_top, 0])
            line_group.add(line)
            self.add(line)
        self.vertical_lines = line_group
        return self

    def get_columns(self):
        return VGroup(
            *[
                VGroup(*[row[i] for row in self.mob_table])
                for i in range(len(self.mob_table[0]))
            ]
        )

    def get_rows(self):
        return VGroup(*[VGroup(*row) for row in self.mob_table])