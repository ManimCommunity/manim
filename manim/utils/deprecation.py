"""Decorators for deprecating classes, functions and function parameters."""

from __future__ import annotations

__all__ = ["deprecated", "deprecated_params"]


import inspect
import re
from typing import Any, Callable, Iterable

from decorator import decorate, decorator

from .. import logger


def _get_callable_info(callable: Callable) -> tuple[str, str]:
    """Returns type and name of a callable.

    Parameters
    ----------
    callable
        The callable

    Returns
    -------
    Tuple[str, str]
        The type and name of the callable. Type can can be one of "class", "method" (for
        functions defined in classes) or "function"). For methods, name is Class.method.
    """
    what = type(callable).__name__
    name = callable.__qualname__
    if what == "function" and "." in name:
        what = "method"
    elif what != "function":
        what = "class"
    return (what, name)


def _deprecation_text_component(
    since: str | None,
    until: str | None,
    message: str,
) -> str:
    """Generates a text component used in deprecation messages.

    Parameters
    ----------
    since
        The version or date since deprecation
    until
        The version or date until removal of the deprecated callable
    message
        The reason for why the callable has been deprecated

    Returns
    -------
    str
        The deprecation message text component.
    """
    since = f"since {since} " if since else ""
    until = (
        f"is expected to be removed after {until}"
        if until
        else "may be removed in a later version"
    )
    msg = " " + message if message else ""
    return f"deprecated {since}and {until}.{msg}"


def deprecated(
    func: Callable = None,
    since: str | None = None,
    until: str | None = None,
    replacement: str | None = None,
    message: str | None = "",
) -> Callable:
    """Decorator to mark a callable as deprecated.

    The decorated callable will cause a warning when used. The docstring of the
    deprecated callable is adjusted to indicate that this callable is deprecated.

    Parameters
    ----------
    func
        The function to be decorated. Should not be set by the user.
    since
        The version or date since deprecation.
    until
        The version or date until removal of the deprecated callable.
    replacement
        The identifier of the callable replacing the deprecated one.
    message
        The reason for why the callable has been deprecated.

    Returns
    -------
    Callable
        The decorated callable.

    Examples
    --------
    Basic usage::

        from manim.utils.deprecation import deprecated

        @deprecated
        def foo(**kwargs):
            pass

        @deprecated
        class Bar:
            def __init__(self):
                pass

            @deprecated
            def baz(self):
                pass

        foo()
        # WARNING  The function foo has been deprecated and may be removed in a later version.

        a = Bar()
        # WARNING  The class Bar has been deprecated and may be removed in a later version.

        a.baz()
        # WARNING  The method Bar.baz has been deprecated and may be removed in a later version.

    You can specify additional information for a more precise warning::

        from manim.utils.deprecation import deprecated

        @deprecated(
            since="v0.2",
            until="v0.4",
            replacement="bar",
            message="It is cooler."
        )
        def foo():
            pass

        foo()
        # WARNING  The function foo has been deprecated since v0.2 and is expected to be removed after v0.4. Use bar instead. It is cooler.

    You may also use dates instead of versions::

        from manim.utils.deprecation import deprecated

        @deprecated(since="05/01/2021", until="06/01/2021")
        def foo():
            pass

        foo()
        # WARNING  The function foo has been deprecated since 05/01/2021 and is expected to be removed after 06/01/2021.

    """
    # If used as factory:
    if func is None:
        return lambda func: deprecated(func, since, until, replacement, message)

    what, name = _get_callable_info(func)

    def warning_msg(for_docs: bool = False) -> str:
        """Generate the deprecation warning message.

        Parameters
        ----------
        for_docs
            Whether or not to format the message for use in documentation.

        Returns
        -------
        str
            The deprecation message.
        """
        msg = message
        if replacement is not None:
            repl = replacement
            if for_docs:
                mapper = {"class": "class", "method": "meth", "function": "func"}
                repl = f":{mapper[what]}:`~.{replacement}`"
            msg = f"Use {repl} instead.{' ' + message if message else ''}"
        deprecated = _deprecation_text_component(since, until, msg)
        return f"The {what} {name} has been {deprecated}"

    def deprecate_docs(func: Callable):
        """Adjust docstring to indicate the deprecation.

        Parameters
        ----------
        func
            The callable whose docstring to adjust.
        """
        warning = warning_msg(True)
        doc_string = func.__doc__ or ""
        func.__doc__ = f"{doc_string}\n\n.. attention:: Deprecated\n  {warning}"

    def deprecate(func: Callable, *args, **kwargs):
        """The actual decorator used to extend the callables behavior.

        Logs a warning message.

        Parameters
        ----------
        func
            The callable to decorate.
        args
            The arguments passed to the given callable.
        kwargs
            The keyword arguments passed to the given callable.

        Returns
        -------
        Any
            The return value of the given callable when being passed the given
            arguments.
        """
        logger.warning(warning_msg())
        return func(*args, **kwargs)

    if type(func).__name__ != "function":
        deprecate_docs(func)
        func.__init__ = decorate(func.__init__, deprecate)
        return func

    func = decorate(func, deprecate)
    deprecate_docs(func)
    return func


def deprecated_params(
    params: str | Iterable[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    message: str | None = "",
    redirections: None | (
        Iterable[tuple[str, str] | Callable[..., dict[str, Any]]]
    ) = None,
) -> Callable:
    """Decorator to mark parameters of a callable as deprecated.

    It can also be used to automatically redirect deprecated parameter values to their
    replacements.

    Parameters
    ----------
    params
        The parameters to be deprecated. Can consist of:

        * An iterable of strings, with each element representing a parameter to deprecate
        * A single string, with parameter names separated by commas or spaces.
    since
        The version or date since deprecation.
    until
        The version or date until removal of the deprecated callable.
    message
        The reason for why the callable has been deprecated.
    redirections
        A list of parameter redirections. Each redirection can be one of the following:

        * A tuple of two strings. The first string defines the name of the deprecated
          parameter; the second string defines the name of the parameter to redirect to,
          when attempting to use the first string.

        * A function performing the mapping operation. The parameter names of the
          function determine which parameters are used as input. The function must
          return a dictionary which contains the redirected arguments.

        Redirected parameters are also implicitly deprecated.

    Returns
    -------
    Callable
        The decorated callable.

    Raises
    ------
    ValueError
        If no parameters are defined (neither explicitly nor implicitly).
    ValueError
        If defined parameters are invalid python identifiers.

    Examples
    --------
    Basic usage::

        from manim.utils.deprecation import deprecated_params

        @deprecated_params(params="a, b, c")
        def foo(**kwargs):
            pass

        foo(x=2, y=3, z=4)
        # No warning

        foo(a=2, b=3, z=4)
        # WARNING  The parameters a and b of method foo have been deprecated and may be removed in a later version.

    You can also specify additional information for a more precise warning::

        from manim.utils.deprecation import deprecated_params

        @deprecated_params(
            params="a, b, c",
            since="v0.2",
            until="v0.4",
            message="The letters x, y, z are cooler."
        )
        def foo(**kwargs):
            pass

        foo(a=2)
        # WARNING  The parameter a of method foo has been deprecated since v0.2 and is expected to be removed after v0.4. The letters x, y, z are cooler.

    Basic parameter redirection::

        from manim.utils.deprecation import deprecated_params

        @deprecated_params(redirections=[
            # Two ways to redirect one parameter to another:
            ("old_param", "new_param"),
            lambda old_param2: {"new_param22": old_param2}
        ])
        def foo(**kwargs):
            return kwargs

        foo(x=1, old_param=2)
        # WARNING  The parameter old_param of method foo has been deprecated and may be removed in a later version.
        # returns {"x": 1, "new_param": 2}

    Redirecting using a calculated value::

        from manim.utils.deprecation import deprecated_params

        @deprecated_params(redirections=[
            lambda runtime_in_ms: {"run_time": runtime_in_ms / 1000}
        ])
        def foo(**kwargs):
            return kwargs

        foo(runtime_in_ms=500)
        # WARNING  The parameter runtime_in_ms of method foo has been deprecated and may be removed in a later version.
        # returns {"run_time": 0.5}

    Redirecting multiple parameter values to one::

        from manim.utils.deprecation import deprecated_params

        @deprecated_params(redirections=[
            lambda buff_x=1, buff_y=1: {"buff": (buff_x, buff_y)}
        ])
        def foo(**kwargs):
            return kwargs

        foo(buff_x=2)
        # WARNING  The parameter buff_x of method foo has been deprecated and may be removed in a later version.
        # returns {"buff": (2, 1)}

    Redirect one parameter to multiple::

        from manim.utils.deprecation import deprecated_params

        @deprecated_params(redirections=[
            lambda buff=1: {"buff_x": buff[0], "buff_y": buff[1]} if isinstance(buff, tuple)
                    else {"buff_x": buff,    "buff_y": buff}
        ])
        def foo(**kwargs):
            return kwargs

        foo(buff=0)
        # WARNING  The parameter buff of method foo has been deprecated and may be removed in a later version.
        # returns {"buff_x": 0, buff_y: 0}

        foo(buff=(1,2))
        # WARNING  The parameter buff of method foo has been deprecated and may be removed in a later version.
        # returns {"buff_x": 1, buff_y: 2}


    """
    # Check if decorator is used without parenthesis
    if callable(params):
        raise ValueError("deprecate_parameters requires arguments to be specified.")

    if params is None:
        params = []

    # Construct params list
    params = re.split(r"[,\s]+", params) if isinstance(params, str) else list(params)

    # Add params which are only implicitly given via redirections
    if redirections is None:
        redirections = []
    for redirector in redirections:
        if isinstance(redirector, tuple):
            params.append(redirector[0])
        else:
            params.extend(list(inspect.signature(redirector).parameters))
    # Keep ordering of params so that warning message is consistently the same
    # This will also help pass unit testing
    params = list(dict.fromkeys(params))

    # Make sure params only contains valid identifiers
    identifier = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
    if not all(re.match(identifier, param) for param in params):
        raise ValueError("Given parameter values are invalid.")

    redirections = list(redirections)

    def warning_msg(func: Callable, used: list[str]):
        """Generate the deprecation warning message.

        Parameters
        ----------
        func
            The callable with deprecated parameters.
        used
            The list of deprecated parameters used in a call.

        Returns
        -------
        str
            The deprecation message.
        """
        what, name = _get_callable_info(func)
        plural = len(used) > 1
        parameter_s = "s" if plural else ""
        used_ = ", ".join(used[:-1]) + " and " + used[-1] if plural else used[0]
        has_have_been = "have been" if plural else "has been"
        deprecated = _deprecation_text_component(since, until, message)
        return f"The parameter{parameter_s} {used_} of {what} {name} {has_have_been} {deprecated}"

    def redirect_params(kwargs: dict, used: list[str]):
        """Adjust the keyword arguments as defined by the redirections.

        Parameters
        ----------
        kwargs
            The keyword argument dictionary to be updated.
        used
            The list of deprecated parameters used in a call.
        """
        for redirector in redirections:
            if isinstance(redirector, tuple):
                old_param, new_param = redirector
                if old_param in used:
                    kwargs[new_param] = kwargs.pop(old_param)
            else:
                redirector_params = list(inspect.signature(redirector).parameters)
                redirector_args = {}
                for redirector_param in redirector_params:
                    if redirector_param in used:
                        redirector_args[redirector_param] = kwargs.pop(redirector_param)
                if len(redirector_args) > 0:
                    kwargs.update(redirector(**redirector_args))

    def deprecate_params(func, *args, **kwargs):
        """The actual decorator function used to extend the callables behavior.

        Logs a warning message when a deprecated parameter is used and redirects it if
        specified.

        Parameters
        ----------
        func
            The callable to decorate.
        args
            The arguments passed to the given callable.
        kwargs
            The keyword arguments passed to the given callable.

        Returns
        -------
        Any
            The return value of the given callable when being passed the given
            arguments.

        """
        used = []
        for param in params:
            if param in kwargs:
                used.append(param)

        if len(used) > 0:
            logger.warning(warning_msg(func, used))
            redirect_params(kwargs, used)
        return func(*args, **kwargs)

    return decorator(deprecate_params)
