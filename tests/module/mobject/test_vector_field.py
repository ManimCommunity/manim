from __future__ import annotations

import numpy as np
import pytest

from manim import BLUE, ORIGIN, ArrowVectorField


def _nan_at_origin(pos):
    return pos if pos.any() else np.full(3, np.nan)


def _inf_at_origin(pos):
    return pos if pos.any() else np.array([np.inf, 0.0, 0.0])


@pytest.mark.parametrize("func", [_nan_at_origin, _inf_at_origin])
@pytest.mark.parametrize("kwargs", [{}, {"color": BLUE}])
def test_arrow_vector_field_with_undefined_value(func, kwargs):
    field = ArrowVectorField(func, x_range=[-1, 1, 1], y_range=[-1, 1, 1], **kwargs)

    assert len(field) == 9
    assert np.isfinite(field.get_all_points()).all()

    vector_at_origin = field[4]
    np.testing.assert_array_equal(vector_at_origin.get_start(), ORIGIN)
    np.testing.assert_array_equal(vector_at_origin.get_end(), ORIGIN)
