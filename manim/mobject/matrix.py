r"""Mobjects representing matrices.

Examples
--------

.. manim:: MatrixExamples
    :save_last_frame:

    class MatrixExamples(Scene):
        def construct(self):
            m0 = Matrix([["\\pi", 0], [-1, 1]])
            m1 = IntegerMatrix([[1.5, 0.], [12, -1.3]],
                left_bracket="(",
                right_bracket=")")
            m2 = DecimalMatrix(
                [[3.456, 2.122], [33.2244, 12.33]],
                element_to_mobject_config={"num_decimal_places": 2},
                left_bracket="\\{",
                right_bracket="\\}")
            m3 = MobjectMatrix(
                [[Circle().scale(0.3), Square().scale(0.3)],
                [MathTex("\\pi").scale(2), Star().scale(0.3)]],
                left_bracket="\\langle",
                right_bracket="\\rangle")
            g = Group(m0, m1, m2, m3).arrange_in_grid(buff=2)
            self.add(g)
"""

__all__ = [
    "Matrix",
    "DecimalMatrix",
    "IntegerMatrix",
    "MobjectMatrix",
    "matrix_to_tex_string",
    "matrix_to_mobject",
    "get_det_text",
]


import itertools as it

import numpy as np

from ..constants import *
from ..mobject.numbers import DecimalNumber, Integer
from ..mobject.svg.tex_mobject import MathTex, Tex
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import WHITE
from .opengl_compatibility import ConvertToOpenGL

# TO DO : The following two functions are not used in this file.
#         Not sure if we should keep it or not.


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


class Matrix(VMobject, metaclass=ConvertToOpenGL):
    """A mobject that displays a matrix on the screen.

    Examples
    --------
    The first example shows a variety of uses of this module while the second example
    exlpains the use of the options `add_background_rectangles_to_entries` and
    `include_background_rectangle`.

    .. manim:: MatrixExamples
        :save_last_frame:

        class MatrixExamples(Scene):
            def construct(self):
                m0 = Matrix([[2, "\\pi"], [-1, 1]])
                m1 = Matrix([[2, 0, 4], [-1, 1, 5]],
                    v_buff=1.3,
                    h_buff=0.8,
                    bracket_h_buff=SMALL_BUFF,
                    bracket_v_buff=SMALL_BUFF,
                    left_bracket="\\{",
                    right_bracket="\\}")
                m1.add(SurroundingRectangle(m1.get_columns()[1]))
                m2 = Matrix([[2, 1], [-1, 3]],
                    element_alignment_corner=UL,
                    left_bracket="(",
                    right_bracket=")")
                m3 = Matrix([[2, 1], [-1, 3]],
                    left_bracket="\\\\langle",
                    right_bracket="\\\\rangle")
                m4 = Matrix([[2, 1], [-1, 3]],
                ).set_column_colors(RED, GREEN)
                m5 = Matrix([[2, 1], [-1, 3]],
                ).set_row_colors(RED, GREEN)
                g = Group(
                    m0,m1,m2,m3,m4,m5
                ).arrange_in_grid(buff=2)
                self.add(g)

    .. manim:: BackgroundRectanglesExample
        :save_last_frame:

        class BackgroundRectanglesExample(Scene):
            def construct(self):
                background= Rectangle().scale(3.2)
                background.set_fill(opacity=.5)
                background.set_color([TEAL, RED, YELLOW])
                self.add(background)
                m0 = Matrix([[12, -30], [-1, 15]],
                    add_background_rectangles_to_entries=True)
                m1 = Matrix([[2, 0], [-1, 1]],
                    include_background_rectangle=True)
                m2 = Matrix([[12, -30], [-1, 15]])
                g = Group(m0, m1, m2).arrange(buff=2)
                self.add(g)
    """

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
        left_bracket="[",
        right_bracket="]",
        **kwargs,
    ):
        """

        Parameters
        ----------
        matrix : :class:`typing.Iterable`
            A numpy 2d array or list of lists
        v_buff : :class:`float`, optional
            vertical buffer, by default 0.8
        h_buff : :class:`float`, optional
            horizontal buffer, by default 1.3
        bracket_h_buff : :class:`float`, optional
            bracket horizontal buffer, by default MED_SMALL_BUFF
        bracket_v_buff : :class:`float`, optional
            bracket vertical buffer, by default MED_SMALL_BUFF
        add_background_rectangles_to_entries : :class:`bool`, optional
            `True` if should add backgraound rectangles to entries, by default False
        include_background_rectangle : :class:`bool`, optional
            `True` if should include background rectangle, by default False
        element_to_mobject : :class:`~.Mobject`, optional
            element to mobject, by default MathTex
        element_to_mobject_config : Dict[:class:`str`, :class:`~.Mobject`], optional
            element to mobject config, by default {}
        element_alignment_corner : :class:`np.ndarray`, optional
            the element alignment corner, by default DR
        left_bracket : :class:`str`, optional
            the left bracket type, by default "["
        right_bracket : :class:`str`, optional
            the right bracket type, by default "]"

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
        super().__init__(**kwargs)
        mob_matrix = self.matrix_to_mob_matrix(matrix)
        self.organize_mob_matrix(mob_matrix)
        self.elements = VGroup(*it.chain(*mob_matrix))
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
        """Used internally."""
        return [
            [
                self.element_to_mobject(item, **self.element_to_mobject_config)
                for item in row
            ]
            for row in matrix
        ]

    def organize_mob_matrix(self, matrix):
        """Used internally."""
        for i, row in enumerate(matrix):
            for j, _ in enumerate(row):
                mob = matrix[i][j]
                mob.move_to(
                    i * self.v_buff * DOWN + j * self.h_buff * RIGHT,
                    self.element_alignment_corner,
                )
        return self

    def add_brackets(self, left="[", right="]"):
        """Used internally. Adds the brackets to the Matrix mobject.

        See Latex document for various bracket types.

        Parameters
        ----------
        left : :class:`str`, optional
            the left bracket, by default "["
        right : :class:`str`, optional
            the right bracket, by default "]"

        Returns
        -------
        :class:`Matrix`
            The current matrix object (self).
        """

        bracket_pair = MathTex(left, right)
        bracket_pair.scale(2)
        bracket_pair.stretch_to_fit_height(self.height + 2 * self.bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(self, LEFT, self.bracket_h_buff)
        r_bracket.next_to(self, RIGHT, self.bracket_h_buff)
        self.add(l_bracket, r_bracket)
        self.brackets = VGroup(l_bracket, r_bracket)
        return self

    def get_columns(self):
        """Return columns of the matrix as VGroups.

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a column of the matrix.

        Examples
        --------

        .. manim:: GetColumnsExample
            :save_last_frame:

            class GetColumnsExample(Scene):
                def construct(self):
                    m0 = Matrix([["\\pi", 3], [1, 5]])
                    m0.add(SurroundingRectangle(m0.get_columns()[1]))
                    self.add(m0)
        """

        return VGroup(
            *(
                VGroup(*(row[i] for row in self.mob_matrix))
                for i in range(len(self.mob_matrix[0]))
            )
        )

    def set_column_colors(self, *colors):
        """Set individual colors for each columns of the matrix.

        Parameters
        ----------
        colors : :class:`str`
            The list of colors; each color specified corresponds to a column.

        Returns
        -------
        :class:`Matrix`
            The current matrix object (self).

        Examples
        --------

        .. manim:: SetColumnColorsExample
            :save_last_frame:

            class SetColumnColorsExample(Scene):
                def construct(self):
                    m0 = Matrix([["\\pi", 1], [-1, 3]],
                    ).set_column_colors([RED,BLUE], GREEN)
                    self.add(m0)
        """
        columns = self.get_columns()
        for color, column in zip(colors, columns):
            column.set_color(color)
        return self

    def get_rows(self):
        """Return rows of the matrix as VGroups.

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a row of the matrix.

        Examples
        --------

        .. manim:: GetRowsExample
            :save_last_frame:

            class GetRowsExample(Scene):
                def construct(self):
                    m0 = Matrix([["\\pi", 3], [1, 5]])
                    m0.add(SurroundingRectangle(m0.get_rows()[1]))
                    self.add(m0)
        """
        return VGroup(*(VGroup(*row) for row in self.mob_matrix))

    def set_row_colors(self, *colors):
        """Set individual colors for each row of the matrix.

        Parameters
        ----------
        colors : :class:`str`
            The list of colors; each color specified corresponds to a row.

        Returns
        -------
        :class:`Matrix`
            The current matrix object (self).

        Examples
        --------

        .. manim:: SetRowColorsExample
            :save_last_frame:

            class SetRowColorsExample(Scene):
                def construct(self):
                    m0 = Matrix([["\\pi", 1], [-1, 3]],
                    ).set_row_colors([RED,BLUE], GREEN)
                    self.add(m0)
        """
        rows = self.get_rows()
        for color, row in zip(colors, rows):
            row.set_color(color)
        return self

    def add_background_to_entries(self):
        """Add a black background rectangle to the matrix,
        see above for an example.

        Returns
        -------
        :class:`Matrix`
            The current matrix object (self).
        """
        for mob in self.get_entries():
            mob.add_background_rectangle()
        return self

    def get_mob_matrix(self):
        """Return the underlying mob matrix mobjects.

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a row of the matrix.
        """
        return self.mob_matrix

    def get_entries(self):
        """Return the individual entries of the matrix.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing entries of the matrix.

        Examples
        --------

        .. manim:: GetEntriesExample
            :save_last_frame:

            class GetEntriesExample(Scene):
                def construct(self):
                    m0 = Matrix([[2, 3], [1, 5]])
                    ent = m0.get_entries()
                    colors = [BLUE, GREEN, YELLOW, RED]
                    for k in range(len(colors)):
                        ent[k].set_color(colors[k])
                    self.add(m0)
        """
        return self.elements

    def get_brackets(self):
        """Return the bracket mobjects.

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a bracket

        Examples
        --------

        .. manim:: GetBracketsExample
            :save_last_frame:

            class GetBracketsExample(Scene):
                def construct(self):
                    m0 = Matrix([["\\pi", 3], [1, 5]])
                    bra = m0.get_brackets()
                    colors = [BLUE, GREEN]
                    for k in range(len(colors)):
                        bra[k].set_color(colors[k])
                    self.add(m0)
        """
        return self.brackets


class DecimalMatrix(Matrix):
    """A mobject that displays a matrix with decimal entries on the screen.

    Examples
    --------

    .. manim:: DecimalMatrixExample
        :save_last_frame:

        class DecimalMatrixExample(Scene):
            def construct(self):
                m0 = DecimalMatrix(
                    [[3.456, 2.122], [33.2244, 12]],
                    element_to_mobject_config={"num_decimal_places": 2},
                    left_bracket="\\{",
                    right_bracket="\\}")
                self.add(m0)
    """

    def __init__(
        self,
        matrix,
        element_to_mobject=DecimalNumber,
        element_to_mobject_config={"num_decimal_places": 1},
        **kwargs,
    ):
        """
        Will round/truncate the decimal places as per the provided config.

        Parameters
        ----------
        matrix : :class:`typing.Iterable`
            A numpy 2d array or list of lists
        element_to_mobject : :class:`~.Mobject`, optional
            Mobject to use, by default DecimalNumber
        element_to_mobject_config : Dict[:class:`str`, :class:`~.Mobject`], optional
            Config for the desired mobject, by default {"num_decimal_places": 1}
        """
        super().__init__(
            matrix,
            element_to_mobject=element_to_mobject,
            element_to_mobject_config=element_to_mobject_config,
            **kwargs,
        )


class IntegerMatrix(Matrix):
    """A mobject that displays a matrix with integer entries on the screen.

    Examples
    --------

    .. manim:: IntegerMatrixExample
        :save_last_frame:

        class IntegerMatrixExample(Scene):
            def construct(self):
                m0 = IntegerMatrix(
                    [[3.7, 2], [42.2, 12]],
                    left_bracket="(",
                    right_bracket=")")
                self.add(m0)
    """

    def __init__(self, matrix, element_to_mobject=Integer, **kwargs):
        """
        Will round if there are decimal entries in the matrix.

        Parameters
        ----------
        matrix : :class:`typing.Iterable`
            A numpy 2d array or list of lists
        element_to_mobject : :class:`~.Mobject`, optional
            Mobject to use, by default Integer
        """
        super().__init__(matrix, element_to_mobject=element_to_mobject, **kwargs)


class MobjectMatrix(Matrix):
    """A mobject that displays a matrix of mobject entries on the screen.

    Examples
    --------

    .. manim:: MobjectMatrixExample
        :save_last_frame:

        class MobjectMatrixExample(Scene):
            def construct(self):
                a = Circle().scale(0.3)
                b = Square().scale(0.3)
                c = MathTex("\\pi").scale(2)
                d = Star().scale(0.3)
                m0 = MobjectMatrix([[a, b], [c, d]])
                self.add(m0)
    """

    def __init__(self, matrix, element_to_mobject=lambda m: m, **kwargs):
        super().__init__(matrix, element_to_mobject=element_to_mobject, **kwargs)


def get_det_text(
    matrix,
    determinant=None,
    background_rect=False,
    initial_scale_factor=2,
):
    r"""Helper function to create determinant.

    Parameters
    ----------
    matrix : :class:`~.Matrix`
        The matrix whose determinant is to be created

    determinant : :class:`int|str`
        The value of the determinant of the matrix

    background_rect : :class:`bool`
        The background rectangle

    initial_scale_factor : :class:`float`
        The scale of the text `det` w.r.t the matrix

    Returns
    --------
    :class:`~.VGroup`
        A VGroup containing the determinant

    Examples
    --------

    .. manim:: DeterminantOfAMatrix
        :save_last_frame:

        class DeterminantOfAMatrix(Scene):
            def construct(self):
                matrix = Matrix([
                    [2, 0],
                    [-1, 1]
                ])

                # scaling down the `det` string
                det = get_det_text(matrix,
                            determinant=3,
                            initial_scale_factor=1)

                # must add the matrix
                self.add(matrix)
                self.add(det)
    """
    parens = MathTex("(", ")")
    parens.scale(initial_scale_factor)
    parens.stretch_to_fit_height(matrix.height)
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
