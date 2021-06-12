r"""Mobjects representing table."""

__all__ = [
    "Tabular"
]


import itertools as it

import numpy as np

from ..constants import *
from ..mobject.numbers import DecimalNumber, Integer
from ..mobject.shape_matchers import BackgroundRxectangle
from ..mobject.svg.tex_mobject import MathTex, Tex
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import WHITE

from manim.mobject.types.vectorized_mobject import VGroup


class Tabular(VGroup):
    def __init__(
        self,
        table,
        lables,
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
        self.organize_mob_table(mob_table)
        self.elements = VGroup(*it.chain(*mob_table))
        self.add(self.elements)
        self.center()
        self.mob_table = mob_table
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
        for i, row in enumerate(table):
            for j, _ in enumerate(row):
                mob = table[i][j]
                mob.move_to(
                    i * self.v_buff * DOWN + j * self.h_buff * RIGHT,
                    self.element_alignment_corner,
                )
        return self

    def add_lines(self, left="[", right="]"):
        bracket_pair = MathTex(left, right)
        bracket_pair.scale(2)
        bracket_pair.stretch_to_fit_height(self.height + 2 * self.bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(self, LEFT, self.bracket_h_buff)
        r_bracket.next_to(self, RIGHT, self.bracket_h_buff)
        self.add(l_bracket, r_bracket)
        self.brackets = VGroup(l_bracket, r_bracket)
        return self

    def add_labels(self, left="[", right="]"):
        bracket_pair = MathTex(left, right)
        bracket_pair.scale(2)
        bracket_pair.stretch_to_fit_height(self.height + 2 * self.bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(self, LEFT, self.bracket_h_buff)
        r_bracket.next_to(self, RIGHT, self.bracket_h_buff)
        self.add(l_bracket, r_bracket)
        self.brackets = VGroup(l_bracket, r_bracket)
        return self