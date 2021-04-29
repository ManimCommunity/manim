import warnings

import pytest

from manim import logger
from manim.utils.deprecation import deprecated, deprecated_params


@deprecated
class Foo:
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


@deprecated(message="Use something else.")
class Quux:
    def __init__(self):
        pass


@deprecated(replacement="ReplaceQuuz")
class Quuz:
    def __init__(self):
        pass


class ReplaceQuuz:
    def __init__(self):
        pass


@deprecated(
    since="0.7.0",
    until="1.2.1",
    replacement="ReplaceQuuz",
    message="Don't use this please.",
)
class QuuzAll:
    def __init__(self):
        pass


def test_deprecate_class_no_args(caplog):
    """Test the deprecation of a class (decorator with no arguments)."""
    f = Foo()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The class Foo has been deprecated and may be removed in a later version."
    )


def test_deprecate_class_since(caplog):
    """Test the deprecation of a class (decorator with since argument)."""
    b = Bar()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The class Bar has been deprecated since v0.6.0 and may be removed in a later version."
    )


def test_deprecate_class_until(caplog):
    """Test the deprecation of a class (decorator with until argument)."""
    bz = Baz()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The class Baz has been deprecated and is expected to be removed after 06/01/2021."
    )


def test_deprecate_class_since_and_until(caplog):
    """Test the deprecation of a class (decorator with since and until arguments)."""
    qx = Qux()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The class Qux has been deprecated since 0.7.0 and is expected to be removed after 0.9.0-rc2."
    )


def test_deprecate_class_msg(caplog):
    """Test the deprecation of a class (decorator with msg argument)."""
    qu = Quux()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The class Quux has been deprecated and may be removed in a later version. Use something else."
    )


def test_deprecate_class_replacement(caplog):
    """Test the deprecation of a class (decorator with replacement argument)."""
    qz = Quuz()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The class Quuz has been deprecated and may be removed in a later version. Use ReplaceQuuz instead."
    )


def test_deprecate_class_all(caplog):
    """Test the deprecation of a class (decorator with all arguments)."""
    qza = QuuzAll()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The class QuuzAll has been deprecated since 0.7.0 and is expected to be removed after 1.2.1. Use ReplaceQuuz instead. Don't use this please."
    )


def _get_caplog_record_msg(caplog):
    logger_name, level, message = caplog.record_tuples[0]
    return message


@deprecated
def useless(**kwargs):
    pass


class Top:
    def __init__(self):
        pass

    @deprecated(since="0.8.0", message="This method is useless.")
    def mid_func(self):
        pass

    @deprecated(until="1.4.0", replacement="Top.NewNested")
    class Nested:
        def __init__(self):
            pass

    class NewNested:
        def __init__(self):
            pass

        @deprecated(since="1.0.0", until="12/25/2025")
        def nested_func(self):
            pass

    class Bottom:
        def __init__(self):
            pass

        def normal_func(self):
            @deprecated
            def nested_func(self):
                pass

            return nested_func

    @deprecated_params(params="a, b, c")
    def foo(self, **kwargs):
        pass

    @deprecated_params(params="a", since="v0.2", until="v0.4")
    def bar(self, **kwargs):
        pass

    @deprecated_params(redirections=[("old_param", "new_param")])
    def baz(self, **kwargs):
        return kwargs

    @deprecated_params(
        redirections=[lambda runtime_in_ms: {"run_time": runtime_in_ms / 1000}]
    )
    def qux(self, **kwargs):
        return kwargs


def test_deprecate_func_no_args(caplog):
    """Test the deprecation of a method (decorator with no arguments)."""
    useless()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The method useless has been deprecated and may be removed in a later version."
    )


def test_deprecate_func_in_class_since_and_message(caplog):
    """Test the deprecation of a method within a class (decorator with since and message arguments)."""
    t = Top()
    t.mid_func()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The method Top.mid_func has been deprecated since 0.8.0 and may be removed in a later version. This method is useless."
    )


def test_deprecate_nested_class_until_and_replacement(caplog):
    """Test the deprecation of a nested class (decorator with until and replacement arguments)."""
    n = Top().Nested()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The class Top.Nested has been deprecated and is expected to be removed after 1.4.0. Use Top.NewNested instead."
    )


def test_deprecate_nested_class_func_since_and_until(caplog):
    """Test the deprecation of a method within a nested class (decorator with since and until arguments)."""
    n = Top().NewNested()
    n.nested_func()
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The method Top.NewNested.nested_func has been deprecated since 1.0.0 and is expected to be removed after 12/25/2025."
    )


def test_deprecate_nested_func(caplog):
    """Test the deprecation of a nested method (decorator with no arguments)."""
    b = Top().Bottom()
    ans = b.normal_func()
    ans(1)
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The method Top.Bottom.normal_func.<locals>.nested_func has been deprecated and may be removed in a later version."
    )


def test_deprecate_func_params(caplog):
    """Test the deprecation of method parameters (decorator with params argument)."""
    t = Top()
    t.foo(a=2, b=3, z=4)
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The parameters a and b of method Top.foo have been deprecated and may be removed in a later version."
    )


def test_deprecate_func_single_param_since_and_until(caplog):
    """Test the deprecation of a single method parameter (decorator with since and until arguments)."""
    t = Top()
    t.bar(a=1, b=2)
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The parameter a of method Top.bar has been deprecated since v0.2 and is expected to be removed after v0.4."
    )


def test_deprecate_func_param_redirect_tuple(caplog):
    """Test the deprecation of a method parameter and redirecting it to a new one using tuple."""
    t = Top()
    obj = t.baz(x=1, old_param=2)
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The parameter old_param of method Top.baz has been deprecated and may be removed in a later version."
    )
    assert obj == {"x": 1, "new_param": 2}


def test_deprecate_func_param_redirect_lambda(caplog):
    """Test the deprecation of a method parameter and redirecting it to a new one using lambda function."""
    t = Top()
    obj = t.qux(runtime_in_ms=500)
    assert len(caplog.record_tuples) == 1
    msg = _get_caplog_record_msg(caplog)
    assert (
        msg
        == "The parameter runtime_in_ms of method Top.qux has been deprecated and may be removed in a later version."
    )
    assert obj == {"run_time": 0.5}
