import warnings

import pytest

from manim import logger
from manim.utils.deprecation import deprecated, deprecated_params


@deprecated
class Foo:
    """This is a class to deprecate."""

    def __init__(self):
        pass


def test_deprecate_class(caplog):
    """Test the deprecation of a class."""
    f = Foo()
    assert len(caplog.record_tuples) == 1
    logger_name, level, message = caplog.record_tuples[0]
    assert (
        message
        == "The class Foo has been deprecated and may be removed in a later version."
    )
