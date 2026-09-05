"""Warnings caused by user code should be attributed to the caller.

Each ``logger.warning`` that a user's own call can trigger passes ``stacklevel``,
so the emitted record points at the line the user wrote rather than at the line
inside Manim where the warning happens to live.
"""

from __future__ import annotations

import inspect
import linecache
from logging import LogRecord

from manim import Mobject
from manim.utils.deprecation import deprecated, deprecated_params


@deprecated(since="v0.1.0", message="Use something else.")
def _deprecated_function() -> int:
    return 1


@deprecated_params(params="old", since="v0.1.0", message="Use new instead.")
def _function_with_deprecated_param(**kwargs: int) -> int:
    return 1


def _assert_blames_caller(record: LogRecord, source_fragment: str) -> None:
    """Assert the record points at the caller's own statement.

    Checked by source text rather than a line number so the test survives
    reformatting.
    """
    caller = inspect.currentframe().f_back.f_code.co_name
    assert record.pathname == __file__, (
        f"warning was attributed to {record.pathname}, expected the calling module"
    )
    assert record.funcName == caller, (
        f"warning was attributed to {record.funcName}(), expected {caller}()"
    )
    blamed_line = linecache.getline(record.pathname, record.lineno)
    assert source_fragment in blamed_line, (
        f"warning pointed at {blamed_line.strip()!r}, "
        f"expected the line containing {source_fragment!r}"
    )


def test_deprecated_function_warning_points_at_caller(manim_caplog):
    _deprecated_function()
    _assert_blames_caller(manim_caplog.records[0], "_deprecated_function()")


def test_deprecated_param_warning_points_at_caller(manim_caplog):
    _function_with_deprecated_param(old=2)
    _assert_blames_caller(
        manim_caplog.records[0], "_function_with_deprecated_param(old=2)"
    )


def test_duplicate_add_warning_points_at_caller(manim_caplog):
    parent, child = Mobject(), Mobject()
    parent.add(child, child)
    _assert_blames_caller(manim_caplog.records[0], "parent.add(child, child)")


def test_duplicate_add_to_back_warning_points_at_caller(manim_caplog):
    parent, child = Mobject(), Mobject()
    parent.add_to_back(child, child)
    _assert_blames_caller(manim_caplog.records[0], "parent.add_to_back(child, child)")
