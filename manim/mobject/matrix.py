"""Mobjects representing matrices."""

__all__ = [
    "Matrix",
    "DecimalMatrix",
    "IntegerMatrix",
    "MobjectMatrix",
    "matrix_to_tex_string",
    "matrix_to_mobject",
    "vector_coordinate_label",
    "get_det_text",
]


import numpy as np

from ..constants import *
from ..mobject.numbers import DecimalNumber
from ..mobject.numbers import Integer
from ..mobject.shape_matchers import BackgroundRectangle
from ..mobject.svg.tex_mobject import MathTex
from ..mobject.svg.tex_mobject import Tex
from ..mobject.types.vectorized_mobject import VGroup
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.color import WHITE

VECTOR_LABEL_SCALE_FACTOR = 0.8


def matrix_to_tex_string(matrix):
    matrix = np.array(matrix).astype("str")
    if matrix.ndim == 1:
        matrix = matrix.reshape((matrix.size, 1))
    n_rows, n_cols = matrix.shape
    prefix = "\\left[ \\begin{array}{%s}" % ("c" * n_cols)
    suffix = "\\end{array} \\right]"
    rows = [" & ".join(row) for row in matrix]
    return prefix + " \\\\ ".join(rows) + suffix


def matrix_to_mobject(matrix):
    return MathTex(matrix_to_tex_string(matrix))


def vector_coordinate_label(vector_mob, integer_labels=True, n_dim=2, color=WHITE):
    vect = np.array(vector_mob.get_end())
    if integer_labels:
        vect = np.round(vect).astype(int)
    vect = vect[:n_dim]
    vect = vect.reshape((n_dim, 1))
    label = Matrix(vect, add_background_rectangles_to_entries=True)
    label.scale(VECTOR_LABEL_SCALE_FACTOR)

    shift_dir = np.array(vector_mob.get_end())
    if shift_dir[0] >= 0:  # Pointing right
        shift_dir -= label.get_left() + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * LEFT
    else:  # Pointing left
        shift_dir -= label.get_right() + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * RIGHT
    label.shift(shift_dir)
    label.set_color(color)
    label.rect = BackgroundRectangle(label)
    label.add_to_back(label.rect)
    return label


class Matrix(VMobject):
    def __init__(
        self,
        matrix,
        v_buff=0.8,
        h_buff=1.3,
        bracket_h_buff=MED_SMALL_BUFF,
        bracket_v_buff=MED_SMALL_BUFF,
        add_background_rectangles_to_entries=False,
        include_background_rectangle=False,
        element_to_mobject=MathTex,
        element_to_mobject_config={},
        element_alignment_corner=DR,
        left_bracket="\\big[",
        right_bracket="\\big]",
        **kwargs,
    ):
        """
        Matrix can either either include numbers, tex_strings,
        or mobjects
        """
        self.v_buff = v_buff
        self.h_buff = h_buff
        self.bracket_h_buff = bracket_h_buff
        self.bracket_v_buff = bracket_v_buff
        self.add_background_rectangles_to_entries = add_background_rectangles_to_entries
        self.include_background_rectangle = include_background_rectangle
        self.element_to_mobject = element_to_mobject
        self.element_to_mobject_config = element_to_mobject_config
        self.element_alignment_corner = element_alignment_corner
        self.left_bracket = left_bracket
        self.right_bracket = right_bracket
        VMobject.__init__(self, **kwargs)
        matrix = np.array(matrix)
        if len(matrix.shape) < 2:
            raise ValueError(
                f"{self.__str__()} class requires a two-dimensional array!"
            )
        mob_matrix = self.matrix_to_mob_matrix(matrix)
        self.organize_mob_matrix(mob_matrix)
        self.elements = VGroup(*mob_matrix.flatten())
        self.add(self.elements)
        self.add_brackets(self.left_bracket, self.right_bracket)
        self.center()
        self.mob_matrix = mob_matrix
        if self.add_background_rectangles_to_entries:
            for mob in self.elements:
                mob.add_background_rectangle()
        if self.include_background_rectangle:
            self.add_background_rectangle()

    def matrix_to_mob_matrix(self, matrix):
        return np.vectorize(self.element_to_mobject)(
            matrix, **self.element_to_mobject_config
        )

    def organize_mob_matrix(self, matrix):
        for i, row in enumerate(matrix):
            for j, elem in enumerate(row):
                mob = matrix[i][j]
                mob.move_to(
                    i * self.v_buff * DOWN + j * self.h_buff * RIGHT,
                    self.element_alignment_corner,
                )
        return self

    def add_brackets(self, left="\\big[", right="\\big]"):
        bracket_pair = MathTex(left, right)
        bracket_pair.scale(2)
        bracket_pair.stretch_to_fit_height(self.get_height() + 2 * self.bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(self, LEFT, self.bracket_h_buff)
        r_bracket.next_to(self, RIGHT, self.bracket_h_buff)
        self.add(l_bracket, r_bracket)
        self.brackets = VGroup(l_bracket, r_bracket)
        return self

    def get_columns(self):
        return VGroup(
            *[VGroup(*self.mob_matrix[:, i]) for i in range(self.mob_matrix.shape[1])]
        )

    def set_column_colors(self, *colors):
        columns = self.get_columns()
        for color, column in zip(colors, columns):
            column.set_color(color)
        return self

    def get_rows(self):
        """Return rows of the matrix as VGroups

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a row of the matrix.
        """
        return VGroup(
            *[VGroup(*self.mob_matrix[i, :]) for i in range(self.mob_matrix.shape[0])]
        )

    def set_row_colors(self, *colors):
        """Set individual colors for each row of the matrix

        Parameters
        ----------
        colors : :class:`str`
            The list of colors; each color specified corresponds to a row.

        Returns
        -------
        :class:`Matrix`
            The current matrix object (self).
        """
        rows = self.get_rows()
        for color, row in zip(colors, rows):
            row.set_color(color)
        return self

    def add_background_to_entries(self):
        for mob in self.get_entries():
            mob.add_background_rectangle()
        return self

    def get_mob_matrix(self):
        return self.mob_matrix

    def get_entries(self):
        return VGroup(*self.get_mob_matrix().flatten())

    def get_brackets(self):
        return self.brackets


class DecimalMatrix(Matrix):
    def __init__(
        self,
        matrix,
        element_to_mobject=DecimalNumber,
        element_to_mobject_config={"num_decimal_places": 1},
        **kwargs,
    ):
        Matrix.__init__(
            self,
            matrix,
            element_to_mobject=element_to_mobject,
            element_to_mobject_config=element_to_mobject_config,
            **kwargs,
        )


class IntegerMatrix(Matrix):
    def __init__(self, matrix, element_to_mobject=Integer, **kwargs):
        Matrix.__init__(self, matrix, element_to_mobject=element_to_mobject, **kwargs)


class MobjectMatrix(Matrix):
    def __init__(self, matrix, element_to_mobject=lambda m: m, **kwargs):
        Matrix.__init__(self, matrix, element_to_mobject=element_to_mobject, **kwargs)


def get_det_text(
    matrix, determinant=None, background_rect=False, initial_scale_factor=2
):
    parens = MathTex("(", ")")
    parens.scale(initial_scale_factor)
    parens.stretch_to_fit_height(matrix.get_height())
    l_paren, r_paren = parens.split()
    l_paren.next_to(matrix, LEFT, buff=0.1)
    r_paren.next_to(matrix, RIGHT, buff=0.1)
    det = Tex("det")
    det.scale(initial_scale_factor)
    det.next_to(l_paren, LEFT, buff=0.1)
    if background_rect:
        det.add_background_rectangle()
    det_text = VGroup(det, l_paren, r_paren)
    if determinant is not None:
        eq = MathTex("=")
        eq.next_to(r_paren, RIGHT, buff=0.1)
        result = MathTex(str(determinant))
        result.next_to(eq, RIGHT, buff=0.2)
        det_text.add(eq, result)
    return det_text
