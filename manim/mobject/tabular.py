r"""Mobjects representing tables."""

__all__ = [
    "Tabular",
    "MathTabular",
    "MobjectTabular",
    "IntegerTabular",
    "DecimalTabular"
]


import itertools as it

from ..constants import *
from ..mobject.geometry import Line
from ..mobject.numbers import DecimalNumber, Integer
from ..mobject.svg.tex_mobject import MathTex, Tex
from ..mobject.svg.text_mobject import Text, Paragraph, MarkupText
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
        element_to_mobject=Text,
        element_to_mobject_config={},
        arrange_in_grid_config={},
        line_config={},
        **kwargs,
    ):

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
        # add lables
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
        help_table.arrange_in_grid(rows=len(table), cols=len(table[0]), buff=(self.h_buff, self.v_buff), **self.arrange_in_grid_config)
        return help_table

    def add_labels(self, mob_table):
        if self.row_labels is not None:
            for k in range(len(self.row_labels)):
                mob_table[k] = [self.row_labels[k]] + mob_table[k]
        if self.col_labels is not None:
            if self.row_labels is not None:
                if self.top_left_entry is not None:
                    col_labels = [self.top_left_entry] + self.col_labels
                    mob_table.insert(0,col_labels)
                else:
                    col_labels = [VMobject()] + self.col_labels
                    mob_table.insert(0,col_labels)
            else:
                mob_table.insert(0,self.col_labels)
        return mob_table

    def add_horizontal_lines(self):
        anchor_left = self.get_left()[0] - 0.5 * self.h_buff
        anchor_right = self.get_right()[0] + 0.5 * self.h_buff
        line_group = VGroup()
        for k in range(len(self.mob_table)-1):
            anchor = self.get_rows()[k+1].get_top()[1] + 0.5 * ( self.get_rows()[k].get_bottom()[1] - self.get_rows()[k+1].get_top()[1] )
            line = Line([anchor_left, anchor, 0], [anchor_right, anchor, 0], **self.line_config)
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
            line = Line([anchor, anchor_bottom, 0], [anchor, anchor_top, 0], **self.line_config)
            line_group.add(line)
            self.add(line)
        self.vertical_lines = line_group
        return self

    def get_horizontal_lines(self):
        return self.horizontal_lines

    def get_vertical_lines(self):
        return self.vertical_lines

    def get_columns(self):
        return VGroup(
            *[
                VGroup(*[row[i] for row in self.mob_table])
                for i in range(len(self.mob_table[0]))
            ]
        )

    def get_rows(self):
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