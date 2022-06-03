from __future__ import annotations

import logging

import pytest

from manim.utils.deprecation import deprecated, deprecated_params


def _get_caplog_record_msg(warn_caplog_manim):
    logger_name, level, message = warn_caplog_manim.record_tuples[0]
    return message


@pytest.fixture()
def warn_caplog_manim(caplog):
    caplog.set_level(logging.WARNING, logger="manim")
    yield caplog


@deprecated
class Foo:
    def __init__(self):
        pass


@deprecated(since="v0.6.0")
class Bar:
    """The Bar class."""

    def __init__(self):
        pass


@deprecated(until="06/01/2021")
class Baz:
    """The Baz class."""

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


doc_admonition = "\n\n.. attention:: Deprecated\n  "


def test_deprecate_class_no_args(warn_caplog_manim):
    """Test the deprecation of a class (decorator with no arguments)."""
    f = Foo()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The class Foo has been deprecated and may be removed in a later version."
    )
    assert f.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_class_since(warn_caplog_manim):
    """Test the deprecation of a class (decorator with since argument)."""
    b = Bar()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The class Bar has been deprecated since v0.6.0 and may be removed in a later version."
    )
    assert b.__doc__ == f"The Bar class.{doc_admonition}{msg}"


def test_deprecate_class_until(warn_caplog_manim):
    """Test the deprecation of a class (decorator with until argument)."""
    bz = Baz()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The class Baz has been deprecated and is expected to be removed after 06/01/2021."
    )
    assert bz.__doc__ == f"The Baz class.{doc_admonition}{msg}"


def test_deprecate_class_since_and_until(warn_caplog_manim):
    """Test the deprecation of a class (decorator with since and until arguments)."""
    qx = Qux()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The class Qux has been deprecated since 0.7.0 and is expected to be removed after 0.9.0-rc2."
    )
    assert qx.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_class_msg(warn_caplog_manim):
    """Test the deprecation of a class (decorator with msg argument)."""
    qu = Quux()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The class Quux has been deprecated and may be removed in a later version. Use something else."
    )
    assert qu.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_class_replacement(warn_caplog_manim):
    """Test the deprecation of a class (decorator with replacement argument)."""
    qz = Quuz()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The class Quuz has been deprecated and may be removed in a later version. Use ReplaceQuuz instead."
    )
    doc_msg = "The class Quuz has been deprecated and may be removed in a later version. Use :class:`~.ReplaceQuuz` instead."
    assert qz.__doc__ == f"{doc_admonition}{doc_msg}"


def test_deprecate_class_all(warn_caplog_manim):
    """Test the deprecation of a class (decorator with all arguments)."""
    qza = QuuzAll()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The class QuuzAll has been deprecated since 0.7.0 and is expected to be removed after 1.2.1. Use ReplaceQuuz instead. Don't use this please."
    )
    doc_msg = "The class QuuzAll has been deprecated since 0.7.0 and is expected to be removed after 1.2.1. Use :class:`~.ReplaceQuuz` instead. Don't use this please."
    assert qza.__doc__ == f"{doc_admonition}{doc_msg}"


@deprecated
def useless(**kwargs):
    pass


class Top:
    def __init__(self):
        pass

    @deprecated(since="0.8.0", message="This method is useless.")
    def mid_func(self):
        """Middle function in Top."""

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
            """Nested function in Top.NewNested."""

            pass

    class Bottom:
        def __init__(self):
            pass

        def normal_func(self):
            @deprecated
            def nested_func(self):
                pass

            return nested_func

    @deprecated_params(params="a, b, c", message="Use something else.")
    def foo(self, **kwargs):
        pass

    @deprecated_params(params="a", since="v0.2", until="v0.4")
    def bar(self, **kwargs):
        pass

    @deprecated_params(redirections=[("old_param", "new_param")])
    def baz(self, **kwargs):
        return kwargs

    @deprecated_params(
        redirections=[lambda runtime_in_ms: {"run_time": runtime_in_ms / 1000}],
    )
    def qux(self, **kwargs):
        return kwargs

    @deprecated_params(
        redirections=[
            lambda point2D_x=1, point2D_y=1: {"point2D": (point2D_x, point2D_y)},
        ],
    )
    def quux(self, **kwargs):
        return kwargs

    @deprecated_params(
        redirections=[
            lambda point2D=1: {"x": point2D[0], "y": point2D[1]}
            if isinstance(point2D, tuple)
            else {"x": point2D, "y": point2D},
        ],
    )
    def quuz(self, **kwargs):
        return kwargs


def test_deprecate_func_no_args(warn_caplog_manim):
    """Test the deprecation of a method (decorator with no arguments)."""
    useless()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The function useless has been deprecated and may be removed in a later version."
    )
    assert useless.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_func_in_class_since_and_message(warn_caplog_manim):
    """Test the deprecation of a method within a class (decorator with since and message arguments)."""
    t = Top()
    t.mid_func()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The method Top.mid_func has been deprecated since 0.8.0 and may be removed in a later version. This method is useless."
    )
    assert t.mid_func.__doc__ == f"Middle function in Top.{doc_admonition}{msg}"


def test_deprecate_nested_class_until_and_replacement(warn_caplog_manim):
    """Test the deprecation of a nested class (decorator with until and replacement arguments)."""
    n = Top().Nested()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The class Top.Nested has been deprecated and is expected to be removed after 1.4.0. Use Top.NewNested instead."
    )
    doc_msg = "The class Top.Nested has been deprecated and is expected to be removed after 1.4.0. Use :class:`~.Top.NewNested` instead."
    assert n.__doc__ == f"{doc_admonition}{doc_msg}"


def test_deprecate_nested_class_func_since_and_until(warn_caplog_manim):
    """Test the deprecation of a method within a nested class (decorator with since and until arguments)."""
    n = Top().NewNested()
    n.nested_func()
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The method Top.NewNested.nested_func has been deprecated since 1.0.0 and is expected to be removed after 12/25/2025."
    )
    assert (
        n.nested_func.__doc__
        == f"Nested function in Top.NewNested.{doc_admonition}{msg}"
    )


def test_deprecate_nested_func(warn_caplog_manim):
    """Test the deprecation of a nested method (decorator with no arguments)."""
    b = Top().Bottom()
    answer = b.normal_func()
    answer(1)
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The method Top.Bottom.normal_func.<locals>.nested_func has been deprecated and may be removed in a later version."
    )
    assert answer.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_func_params(warn_caplog_manim):
    """Test the deprecation of method parameters (decorator with params argument)."""
    t = Top()
    t.foo(a=2, b=3, z=4)
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The parameters a and b of method Top.foo have been deprecated and may be removed in a later version. Use something else."
    )


def test_deprecate_func_single_param_since_and_until(warn_caplog_manim):
    """Test the deprecation of a single method parameter (decorator with since and until arguments)."""
    t = Top()
    t.bar(a=1, b=2)
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The parameter a of method Top.bar has been deprecated since v0.2 and is expected to be removed after v0.4."
    )


def test_deprecate_func_param_redirect_tuple(warn_caplog_manim):
    """Test the deprecation of a method parameter and redirecting it to a new one using tuple."""
    t = Top()
    obj = t.baz(x=1, old_param=2)
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The parameter old_param of method Top.baz has been deprecated and may be removed in a later version."
    )
    assert obj == {"x": 1, "new_param": 2}


def test_deprecate_func_param_redirect_lambda(warn_caplog_manim):
    """Test the deprecation of a method parameter and redirecting it to a new one using lambda function."""
    t = Top()
    obj = t.qux(runtime_in_ms=500)
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The parameter runtime_in_ms of method Top.qux has been deprecated and may be removed in a later version."
    )
    assert obj == {"run_time": 0.5}


def test_deprecate_func_param_redirect_many_to_one(warn_caplog_manim):
    """Test the deprecation of multiple method parameters and redirecting them to one."""
    t = Top()
    obj = t.quux(point2D_x=3, point2D_y=5)
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The parameters point2D_x and point2D_y of method Top.quux have been deprecated and may be removed in a later version."
    )
    assert obj == {"point2D": (3, 5)}


def test_deprecate_func_param_redirect_one_to_many(warn_caplog_manim):
    """Test the deprecation of one method parameter and redirecting it to many."""
    t = Top()
    obj1 = t.quuz(point2D=0)
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The parameter point2D of method Top.quuz has been deprecated and may be removed in a later version."
    )
    assert obj1 == {"x": 0, "y": 0}

    warn_caplog_manim.clear()

    obj2 = t.quuz(point2D=(2, 3))
    assert len(warn_caplog_manim.record_tuples) == 1
    msg = _get_caplog_record_msg(warn_caplog_manim)
    assert (
        msg
        == "The parameter point2D of method Top.quuz has been deprecated and may be removed in a later version."
    )
    assert obj2 == {"x": 2, "y": 3}
