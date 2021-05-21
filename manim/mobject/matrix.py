r"""Mobjects representing matrices.

Examples
--------

.. manim:: MatrixExamples
    :save_last_frame:

    class MatrixExamples(Scene):
        def construct(self):
            m0 = Matrix([[2, 0], [-1, 1]])
            m1 = Matrix([[1, 0], [0, 1]],
                        left_bracket="\\big(",
                        right_bracket="\\big)")
            m2 = DecimalMatrix(
                [[3.456, 2.122], [33.2244, 12.33]],
                element_to_mobject_config={"num_decimal_places": 2},
                left_bracket="\\{",
                right_bracket="\\}")

            self.add(m0.shift(LEFT - (3, 0, 0)))
            self.add(m1)
            self.add(m2.shift(RIGHT + (3, 0, 0)))

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


import numpy as np

from ..constants import *
from ..mobject.numbers import DecimalNumber, Integer
from ..mobject.shape_matchers import BackgroundRectangle
from ..mobject.svg.tex_mobject import MathTex, Tex
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import WHITE


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


class Matrix(VMobject):
    """A mobject that displays a matrix on the screen."""

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
            the left bracket type, by default "\\\\big["
        right_bracket : :class:`str`, optional
            the right bracket type, by default "\\\\big]"

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
            for j, _ in enumerate(row):
                mob = matrix[i][j]
                mob.move_to(
                    i * self.v_buff * DOWN + j * self.h_buff * RIGHT,
                    self.element_alignment_corner,
                )
        return self

    def add_brackets(self, left="\\big[", right="\\big]"):
        """Add the brackets to the Matrix mobject

        See Latex document for various bracket types.

        Parameters
        ----------
        left : :class:`str`, optional
            the left bracket, by default "\\\\big["
        right : :class:`str`, optional
            the right bracket, by default "\\\\big]"

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
        """Return columns of the matrix as VGroups

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a column of the matrix.
        """
        return VGroup(
            *[VGroup(*self.mob_matrix[:, i]) for i in range(self.mob_matrix.shape[1])]
        )

    def set_column_colors(self, *colors):
        """Set individual colors for each columns of the matrix

        Parameters
        ----------
        colors : :class:`str`
            The list of colors; each color specified corresponds to a column.

        Returns
        -------
        :class:`Matrix`
            The current matrix object (self).
        """
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
        """Return the underlying mob matrix mobjects

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a row of the matrix.
        """
        return self.mob_matrix

    def get_entries(self):
        """Return the individual entries of the matrix

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing entries of the matrix
        """
        return VGroup(*self.get_mob_matrix().flatten())

    def get_brackets(self):
        """Return the bracket mobjects

        Returns
        --------
        List[:class:`~.VGroup`]
            Each VGroup contains a bracket
        """
        return self.brackets


class DecimalMatrix(Matrix):
    """A mobject that displays a matrix with decimal entries on the screen."""

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
        Matrix.__init__(
            self,
            matrix,
            element_to_mobject=element_to_mobject,
            element_to_mobject_config=element_to_mobject_config,
            **kwargs,
        )


class IntegerMatrix(Matrix):
    """A mobject that displays a matrix with integer entries on the screen."""

    def __init__(self, matrix, element_to_mobject=Integer, **kwargs):
        """
        Note- Will round if there are decimal entries in the matrix.

        Parameters
        ----------
        matrix : :class:`typing.Iterable`
            A numpy 2d array or list of lists
        element_to_mobject : :class:`~.Mobject`, optional
            Mobject to use, by default Integer
        """
        Matrix.__init__(self, matrix, element_to_mobject=element_to_mobject, **kwargs)


class MobjectMatrix(Matrix):
    """A mobject that displays a matrix of mobject entries on the screen."""

    def __init__(self, matrix, element_to_mobject=lambda m: m, **kwargs):
        Matrix.__init__(self, matrix, element_to_mobject=element_to_mobject, **kwargs)


def get_det_text(
    matrix, determinant=None, background_rect=False, initial_scale_factor=2
):
    r"""Helper function to create determinant

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
