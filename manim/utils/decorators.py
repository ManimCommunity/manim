__all__ = ["deprecated", "deprecated_params"]


from manim.constants import SMALL_BUFF
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import re

from numpy import sinc

# from .. import logger


def get_callable_description(callable: Callable) -> Tuple[str, str]:
    what = type(callable).__name__
    name = callable.__qualname__
    if what == "function" and name[0].isupper():  # TODO a bit hacky but works
        what = "method"
    elif what == "type":
        what = "class"
    return (what, name)


def deprecation_text_component(
    since: Optional[str], until: Optional[str], message: str
) -> str:
    since = "" if since is None else f"since {since} "
    until = "may be deleted soon" if until is None else f"will be deleted after {until}"
    return f"deprecated {since}and {until}. {message}"


def deprecated(
    func: Optional[Callable] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    replacement: Optional[str] = None,
    message: str = "",
) -> Callable:
    def warning_msg(func):
        what, name = get_callable_description(func)
        message_ = message
        if replacement is not None:
            message_ = f"Use {replacement} instead. {message}"
        deprecated = deprecation_text_component(since, until, message_)

        return f"The {what} {name} is {deprecated}"

    def decorator(func):
        def deprecated_func(*args, **kwargs):
            print(warning_msg(func))
            return func(*args, **kwargs)

        deprecated_func.__qualname__ = func.__qualname__
        return deprecated_func

    if func is None:
        return decorator
    else:
        return decorator(func)


def deprecated_params(
    params: Union[str, Iterable[str]] = [],
    since: Optional[str] = None,
    until: Optional[str] = None,
    message: str = "",
    redirections: "Iterable[Union[Tuple[str, str], Callable[..., dict[str, Any]]]]" = [],
) -> Callable:
    # Check if decorator is used without parenthesis
    if callable(params):
        raise ValueError("deprecate_parameters requires arguments to be specified.")

    # Construct params list
    params = re.split("[,\s]+", params) if isinstance(params, str) else list(params)

    # Add params which are only implicitly given via redirections
    for redirector in redirections:
        if isinstance(redirector, tuple):
            params.append(redirector[0])
        else:
            params.extend(redirector.__code__.co_varnames)

    # remove duplicates
    prams = list(set(params))

    # Make sure prams only contains valid identifiers
    identifier = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
    if not all(re.match(identifier, param) for param in params):
        raise ValueError("Given parameter values are invalid.")

    redirections = list(redirections)

    def warning_msg(func, used):
        what, name = get_callable_description(func)
        plural = len(used) > 1
        prameter_s = "s" if plural else ""
        used_ = ", ".join(used[:-1]) + " and " + used[-1] if plural else used[0]
        is_are = "are" if plural else "is"
        deprecated = deprecation_text_component(since, until, message)
        return (
            f"The parameter{prameter_s} {used_} of {what} {name} {is_are} {deprecated}"
        )

    def decorator(func):
        def deprecated_func(*args, **kwargs):
            used = []
            for param in params:
                if param in kwargs:
                    used.append(param)
            if len(used) > 0:
                print(warning_msg(func, used))

                for redirector in redirections:
                    if isinstance(redirector, tuple):
                        old_param, new_param = redirector

                        if old_param in used:
                            kwargs[new_param] = kwargs.pop(old_param)
                    else:
                        redirector_params = redirector.__code__.co_varnames
                        redirector_args = {}
                        for r_param in redirector_params:
                            if r_param in used:
                                redirector_args[r_param] = kwargs.pop(r_param)
                        if len(redirector_args) > 0:
                            kwargs.update(redirector(**redirector_args))

            return func(*args, **kwargs)

        deprecated_func.__qualname__ = func.__qualname__
        return deprecated_func

    return decorator


# @redirect_params(
#     ("n_cols", "cols"),  # equivalent to second line
#     lambda n_cols: {"cols": n_cols},
#     lambda buff_x=SMALL_BUFF, buff_y=SMALL_BUFF: {"buff": (buff_x, buff_y)},
#     lambda buff: {"buff_x": buff, "buff_y": buff},
# )

# vorgehen:
# Einen nach dem anderen durchgehen, gucken ob min ein param passt
# Falls ja:
#   Ausführen (muss halt im Zweifelsfall defaults haben)


# @deprecated
# @deprecated(since="0.2")
# @deprecated(until="0.5")
# @deprecated(since="0.2", until="0.5")
# @deprecated(replacement="bar", message="It's better.")
# def foo():
#     pass


# @deprecated
# class Foo:
#     @deprecated
#     def bar(self):
#         pass


# @deprecated_params(redirections=[
#         ("n_rows", "rows"),
#         lambda n_cols: {"cols": n_cols}, # equivalent to ("n_cols", "cols"),
#         lambda buff_x=SMALL_BUFF, buff_y=SMALL_BUFF: {"buff": (buff_x, buff_y)}
# ])
# def baz(rows=None, cols=None, buff=None):
#     print(f"rows={rows}, cols={cols}, buff={buff}")


# @deprecated_params(
#     since="0.2",
#     until="0.5",
#     redirections=[
#         lambda buff: {"buff_x": buff[0], "buff_y": buff[1]}
#         if isinstance(buff, tuple)
#         else {"buff_x": buff, "buff_y": buff}
#     ],
# )
# def zab(buff_x=None, buff_y=None):
#     print(f"buff_x={buff_x}, buff_y={buff_y}")


# print("")
# # zab(buff=2)


# # foo()

# baz(n_rows=2)
# baz(n_cols=2, n_rows=4)
# baz(buff_x=1)
# baz(buff_x=1, buff_y=2)
# print("")
# # baz(n_cols=2)
# # print("")
# # baz(n_rows=5, n_cols=2)
# # print("")

# # print("")


# # # print("")

# # a = Foo()

# # print("")

# # a.bar()

# # # print("")

# # @deprecated_params(params="useless_param, other_useless_param", until="0.5")
# # def foo(useless_param=None, other_useless_param=None):
# #     pass

# # foo()
# # foo(useless_param=2)
# # foo(useless_param=2, other_useless_param=0)


# @deprecated_params(redirections=[
#         lambda buff: {"buff_x": buff[0], "buff_y": buff[1]} if isinstance(buff, tuple)
#                 else {"buff_x": buff, "buff_y": buff}
# ])
# def zab(buff_x, buff_y):
#     print(f"buff_x={buff_x}, buff_y={buff_y}")

# print("")

# zab(buff=1)
# zab(buff=(2,3))