from __future__ import annotations

import inspect

import pytest

from manim import Square, VGroup
from manim._config.logger_utils import user_stacklevel
from manim.utils.deprecation import deprecated


def _current_line() -> int:
    """Return the line number of the caller."""
    return inspect.currentframe().f_back.f_lineno


def _only_record(manim_caplog: pytest.LogCaptureFixture):
    assert len(manim_caplog.records) == 1
    return manim_caplog.records[0]


def test_user_stacklevel_outside_manim_is_one() -> None:
    # Called from user code, the logging call itself is already user code.
    assert user_stacklevel() == 1


def test_duplicate_submobject_warning_points_at_vgroup_call(
    manim_caplog: pytest.LogCaptureFixture,
) -> None:
    square = Square()
    line = _current_line() + 1
    VGroup(square, square)

    record = _only_record(manim_caplog)
    assert record.filename == "test_user_stacklevel.py"
    assert record.lineno == line
    assert "test_user_stacklevel.py" in record.stack_info


def test_duplicate_submobject_warning_points_at_add_call(
    manim_caplog: pytest.LogCaptureFixture,
) -> None:
    # Fewer Manim frames lie between this call and the warning than for
    # VGroup(...), so a fixed stacklevel could not handle both cases.
    square = Square()
    group = VGroup()
    line = _current_line() + 1
    group.add(square, square)

    record = _only_record(manim_caplog)
    assert record.filename == "test_user_stacklevel.py"
    assert record.lineno == line


def test_deprecation_warning_points_at_caller(
    manim_caplog: pytest.LogCaptureFixture,
) -> None:
    @deprecated
    def old_function() -> None:
        pass

    line = _current_line() + 1
    old_function()

    record = _only_record(manim_caplog)
    assert record.filename == "test_user_stacklevel.py"
    assert record.lineno == line
    assert record.stack_info is not None
