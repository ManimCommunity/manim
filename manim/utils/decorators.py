__all__ = ["deprecated", "deprecated_params"]


from typing import Any, Callable, Iterable, Optional, Tuple, Union
import re
from functools import wraps

from .. import logger


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
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            logger.warn(warning_msg(func))
            return func(*args, **kwargs)

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
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            used = []
            for param in params:
                if param in kwargs:
                    used.append(param)
            if len(used) > 0:
                logger.warn(warning_msg(func, used))

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

        return deprecated_func

    return decorator