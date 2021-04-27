import warnings

import pytest

from manim import logger
from manim.utils.deprecation import deprecated, deprecated_params


@deprecated
class Foo:
    """This is a class to deprecate."""

    def __init__(self):
        pass


@deprecated(since="v0.6.0")
class Bar:
    def __init__(self):
        pass


@deprecated(until="06/01/2021")
class Baz:
    def __init__(self):
        pass


@deprecated(since="0.7.0", until="0.9.0-rc2")
class Qux:
    def __init__(self):
        pass


def test_deprecate_class_no_args(caplog):
    """Test the deprecation of a class (decorator with no arguments)."""
    f = Foo()
    assert len(caplog.record_tuples) == 1
    logger_name, level, message = caplog.record_tuples[0]
    assert (
        message
        == "The class Foo has been deprecated and may be removed in a later version."
    )


def test_deprecate_class_since(caplog):
    """Test the deprecation of a class (decorator with since argument)."""
    b = Bar()
    assert len(caplog.record_tuples) == 1
    logger_name, level, message = caplog.record_tuples[0]
    assert (
        message
        == "The class Bar has been deprecated since v0.6.0 and may be removed in a later version."
    )


def test_deprecate_class_until(caplog):
    """Test the deprecation of a class (decorator with until argument)."""
    bz = Baz()
    assert len(caplog.record_tuples) == 1
    logger_name, level, message = caplog.record_tuples[0]
    assert (
        message
        == "The class Baz has been deprecated and is expected to be removed after 06/01/2021."
    )


def test_deprecate_class_since_and_until(caplog):
    """Test the deprecation of a class (decorator with since and until arguments)."""
    qx = Qux()
    assert len(caplog.record_tuples) == 1
    logger_name, level, message = caplog.record_tuples[0]
    assert (
        message
        == "The class Qux has been deprecated since 0.7.0 and is expected to be removed after 0.9.0-rc2."
    )
