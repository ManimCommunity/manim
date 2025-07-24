from __future__ import annotations

import numpy as np
import pytest

from manim.mobject.matrix import (
    DecimalMatrix,
    IntegerMatrix,
    Matrix,
)
from manim.mobject.text.tex_mobject import MathTex
from manim.mobject.types.vectorized_mobject import VGroup


class TestMatrix:
    @pytest.mark.parametrize(
        (
            "matrix_elements",
            "left_bracket",
            "right_bracket",
            "expected_rows",
            "expected_columns",
        ),
        [
            ([[1, 2], [3, 4]], "[", "]", 2, 2),
            ([[1, 2, 3]], "[", "]", 1, 3),
            ([[1], [2], [3]], "[", "]", 3, 1),
            ([[5]], "[", "]", 1, 1),
            ([[1, 0], [0, 1]], "(", ")", 2, 2),
            ([["a", "b"], ["c", "d"]], "[", "]", 2, 2),
            (np.array([[10, 20], [30, 40]]), "[", "]", 2, 2),
        ],
        ids=[
            "2x2_default",
            "1x3_default",
            "3x1_default",
            "1x1_default",
            "2x2_parentheses",
            "2x2_strings",
            "2x2_numpy",
        ],
    )
    def test_matrix_init_valid(
        self,
        matrix_elements,
        left_bracket,
        right_bracket,
        expected_rows,
        expected_columns,
    ):
        matrix = Matrix(
            matrix_elements, left_bracket=left_bracket, right_bracket=right_bracket
        )

        assert isinstance(matrix, Matrix)
        assert matrix.left_bracket == left_bracket
        assert matrix.right_bracket == right_bracket
        assert len(matrix.get_rows()) == expected_rows
        assert len(matrix.get_columns()) == expected_columns

    @pytest.mark.parametrize(
        ("invalid_elements", "expected_error"),
        [
            (10, TypeError),
            (10.4, TypeError),
            ([1, 2, 3], TypeError),
        ],
        ids=[
            "integer",
            "float",
            "flat_list",
        ],
    )
    def test_matrix_init_invalid(self, invalid_elements, expected_error):
        with pytest.raises(expected_error):
            Matrix(invalid_elements)

    @pytest.mark.parametrize(
        ("matrix_elements", "expected_columns"),
        [
            ([[1, 2], [3, 4]], 2),
            ([[1, 2, 3]], 3),
            ([[1], [2], [3]], 1),
        ],
        ids=["2x2", "1x3", "3x1"],
    )
    def test_get_columns(self, matrix_elements, expected_columns):
        matrix = Matrix(matrix_elements)

        assert isinstance(matrix, Matrix)
        assert len(matrix.get_columns()) == expected_columns
        for column in matrix.get_columns():
            assert isinstance(column, VGroup)

    @pytest.mark.parametrize(
        ("matrix_elements", "expected_rows"),
        [
            ([[1, 2], [3, 4]], 2),
            ([[1, 2, 3]], 1),
            ([[1], [2], [3]], 3),
        ],
        ids=["2x2", "1x3", "3x1"],
    )
    def test_get_rows(self, matrix_elements, expected_rows):
        matrix = Matrix(matrix_elements)

        assert isinstance(matrix, Matrix)
        assert len(matrix.get_rows()) == expected_rows
        for row in matrix.get_rows():
            assert isinstance(row, VGroup)

    @pytest.mark.parametrize(
        ("matrix_elements", "expected_entries_tex_string", "expected_entries_count"),
        [
            ([[1, 2], [3, 4]], ["1", "2", "3", "4"], 4),
            ([[1, 2, 3]], ["1", "2", "3"], 3),
        ],
        ids=["2x2", "1x3"],
    )
    def test_get_entries(
        self, matrix_elements, expected_entries_tex_string, expected_entries_count
    ):
        matrix = Matrix(matrix_elements)
        entries = matrix.get_entries()

        assert isinstance(matrix, Matrix)
        assert len(entries) == expected_entries_count
        for index_entry, entry in enumerate(entries):
            assert isinstance(entry, MathTex)
            assert expected_entries_tex_string[index_entry] == entry.tex_string

    @pytest.mark.parametrize(
        ("matrix_elements", "row", "column", "expected_value_str"),
        [
            ([[1, 2], [3, 4]], 0, 0, "1"),
            ([[1, 2], [3, 4]], 1, 1, "4"),
            ([[1, 2, 3]], 0, 2, "3"),
            ([[1], [2], [3]], 2, 0, "3"),
        ],
        ids=["2x2_00", "2x2_11", "1x3_02", "3x1_20"],
    )
    def test_get_element(self, matrix_elements, row, column, expected_value_str):
        matrix = Matrix(matrix_elements)

        assert isinstance(matrix.get_columns()[column][row], MathTex)
        assert isinstance(matrix.get_rows()[row][column], MathTex)
        assert matrix.get_columns()[column][row].tex_string == expected_value_str
        assert matrix.get_rows()[row][column].tex_string == expected_value_str

    @pytest.mark.parametrize(
        ("matrix_elements", "row", "column", "expected_error"),
        [
            ([[1, 2]], 1, 0, IndexError),
            ([[1, 2]], 0, 2, IndexError),
        ],
        ids=["row_out_of_bounds", "col_out_of_bounds"],
    )
    def test_get_element_invalid(self, matrix_elements, row, column, expected_error):
        matrix = Matrix(matrix_elements)

        with pytest.raises(expected_error):
            matrix.get_columns()[column][row]

        with pytest.raises(expected_error):
            matrix.get_rows()[row][column]


class TestDecimalMatrix:
    @pytest.mark.parametrize(
        ("matrix_elements", "num_decimal_places", "expected_elements"),
        [
            ([[1.234, 5.678], [9.012, 3.456]], 2, [[1.234, 5.678], [9.012, 3.456]]),
            ([[1.0, 2.0], [3.0, 4.0]], 0, [[1, 2], [3, 4]]),
            ([[1, 2.3], [4.567, 7]], 1, [[1.0, 2.3], [4.567, 7.0]]),
        ],
        ids=[
            "basic_2_decimal_points",
            "basic_0_decimal_points",
            "mixed_1_decimal_points",
        ],
    )
    def test_decimal_matrix_init(
        self, matrix_elements, num_decimal_places, expected_elements
    ):
        matrix = DecimalMatrix(
            matrix_elements,
            element_to_mobject_config={"num_decimal_places": num_decimal_places},
        )

        assert isinstance(matrix, DecimalMatrix)
        for column_index, column in enumerate(matrix.get_columns()):
            for row_index, element in enumerate(column):
                assert element.number == expected_elements[row_index][column_index]
                assert element.num_decimal_places == num_decimal_places


class TestIntegerMatrix:
    @pytest.mark.parametrize(
        ("matrix_elements", "expected_elements"),
        [
            ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
            ([[1.2, 2.8], [3.5, 4]], [[1.2, 2.8], [3.5, 4]]),
        ],
        ids=["basic_int", "mixed_float_int"],
    )
    def test_integer_matrix_init(self, matrix_elements, expected_elements):
        matrix = IntegerMatrix(matrix_elements)

        assert isinstance(matrix, IntegerMatrix)
        for row_index, row in enumerate(matrix.get_rows()):
            for column_index, element in enumerate(row):
                assert element.number == expected_elements[row_index][column_index]
