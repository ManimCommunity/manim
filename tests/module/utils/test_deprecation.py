from __future__ import annotations

import pytest

from manim.utils.deprecation import deprecated, deprecated_params


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


def test_deprecate_class_no_args():
    """Test the deprecation of a class (decorator with no arguments)."""

    msg = "The class Foo has been deprecated and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        f = Foo()
    assert f.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_class_since():
    """Test the deprecation of a class (decorator with since argument)."""
    msg = "The class Bar has been deprecated since v0.6.0 and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        b = Bar()
    assert b.__doc__ == f"The Bar class.{doc_admonition}{msg}"


def test_deprecate_class_until():
    """Test the deprecation of a class (decorator with until argument)."""
    msg = "The class Baz has been deprecated and is expected to be removed after 06/01/2021."
    with pytest.warns(FutureWarning, match=msg):
        bz = Baz()
    assert bz.__doc__ == f"The Baz class.{doc_admonition}{msg}"


def test_deprecate_class_since_and_until():
    """Test the deprecation of a class (decorator with since and until arguments)."""
    msg = "The class Qux has been deprecated since 0.7.0 and is expected to be removed after 0.9.0-rc2."
    with pytest.warns(FutureWarning, match=msg):
        qx = Qux()
    assert qx.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_class_msg():
    """Test the deprecation of a class (decorator with msg argument)."""
    msg = "The class Quux has been deprecated and may be removed in a later version. Use something else."
    with pytest.warns(FutureWarning, match=msg):
        qu = Quux()
    assert qu.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_class_replacement():
    """Test the deprecation of a class (decorator with replacement argument)."""
    msg = "The class Quuz has been deprecated and may be removed in a later version. Use ReplaceQuuz instead."
    with pytest.warns(FutureWarning, match=msg):
        qz = Quuz()
    doc_msg = "The class Quuz has been deprecated and may be removed in a later version. Use :class:`~.ReplaceQuuz` instead."
    assert qz.__doc__ == f"{doc_admonition}{doc_msg}"


def test_deprecate_class_all():
    """Test the deprecation of a class (decorator with all arguments)."""
    msg = "The class QuuzAll has been deprecated since 0.7.0 and is expected to be removed after 1.2.1. Use ReplaceQuuz instead. Don't use this please."
    with pytest.warns(FutureWarning, match=msg):
        qza = QuuzAll()
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
            lambda point2D=1: (
                {"x": point2D[0], "y": point2D[1]}
                if isinstance(point2D, tuple)
                else {"x": point2D, "y": point2D}
            ),
        ],
    )
    def quuz(self, **kwargs):
        return kwargs


def test_deprecate_func_no_args():
    """Test the deprecation of a method (decorator with no arguments)."""
    msg = "The function useless has been deprecated and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        useless()
    assert useless.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_func_in_class_since_and_message():
    """Test the deprecation of a method within a class (decorator with since and message arguments)."""
    t = Top()
    msg = "The method Top.mid_func has been deprecated since 0.8.0 and may be removed in a later version. This method is useless."
    with pytest.warns(FutureWarning, match=msg):
        t.mid_func()
    assert t.mid_func.__doc__ == f"Middle function in Top.{doc_admonition}{msg}"


def test_deprecate_nested_class_until_and_replacement():
    """Test the deprecation of a nested class (decorator with until and replacement arguments)."""
    msg = "The class Top.Nested has been deprecated and is expected to be removed after 1.4.0. Use Top.NewNested instead."
    with pytest.warns(FutureWarning, match=msg):
        n = Top().Nested()
    doc_msg = "The class Top.Nested has been deprecated and is expected to be removed after 1.4.0. Use :class:`~.Top.NewNested` instead."
    assert n.__doc__ == f"{doc_admonition}{doc_msg}"


def test_deprecate_nested_class_func_since_and_until():
    """Test the deprecation of a method within a nested class (decorator with since and until arguments)."""
    n = Top().NewNested()
    msg = "The method Top.NewNested.nested_func has been deprecated since 1.0.0 and is expected to be removed after 12/25/2025."
    with pytest.warns(FutureWarning, match=msg):
        n.nested_func()
    assert (
        n.nested_func.__doc__
        == f"Nested function in Top.NewNested.{doc_admonition}{msg}"
    )


def test_deprecate_nested_func():
    """Test the deprecation of a nested method (decorator with no arguments)."""
    b = Top().Bottom()
    answer = b.normal_func()
    msg = "The method Top.Bottom.normal_func.<locals>.nested_func has been deprecated and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        answer(1)
    assert answer.__doc__ == f"{doc_admonition}{msg}"


def test_deprecate_func_params():
    """Test the deprecation of method parameters (decorator with params argument)."""
    t = Top()
    msg = "The parameters a and b of method Top.foo have been deprecated and may be removed in a later version. Use something else."
    with pytest.warns(FutureWarning, match=msg):
        t.foo(a=2, b=3, z=4)


def test_deprecate_func_single_param_since_and_until():
    """Test the deprecation of a single method parameter (decorator with since and until arguments)."""
    t = Top()
    msg = "The parameter a of method Top.bar has been deprecated since v0.2 and is expected to be removed after v0.4."
    with pytest.warns(FutureWarning, match=msg):
        t.bar(a=1, b=2)


def test_deprecate_func_param_redirect_tuple():
    """Test the deprecation of a method parameter and redirecting it to a new one using tuple."""
    t = Top()
    msg = "The parameter old_param of method Top.baz has been deprecated and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        obj = t.baz(x=1, old_param=2)
    assert obj == {"x": 1, "new_param": 2}


def test_deprecate_func_param_redirect_lambda():
    """Test the deprecation of a method parameter and redirecting it to a new one using lambda function."""
    t = Top()
    msg = "The parameter runtime_in_ms of method Top.qux has been deprecated and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        obj = t.qux(runtime_in_ms=500)
    assert obj == {"run_time": 0.5}


def test_deprecate_func_param_redirect_many_to_one():
    """Test the deprecation of multiple method parameters and redirecting them to one."""
    t = Top()
    msg = "The parameters point2D_x and point2D_y of method Top.quux have been deprecated and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        obj = t.quux(point2D_x=3, point2D_y=5)
    assert obj == {"point2D": (3, 5)}


def test_deprecate_func_param_redirect_one_to_many():
    """Test the deprecation of one method parameter and redirecting it to many."""
    t = Top()
    msg = "The parameter point2D of method Top.quuz has been deprecated and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        obj1 = t.quuz(point2D=0)
    assert obj1 == {"x": 0, "y": 0}

    msg = "The parameter point2D of method Top.quuz has been deprecated and may be removed in a later version."
    with pytest.warns(FutureWarning, match=msg):
        obj2 = t.quuz(point2D=(2, 3))
    assert obj2 == {"x": 2, "y": 3}
