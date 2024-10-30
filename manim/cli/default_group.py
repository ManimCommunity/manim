"""``DefaultGroup`` allows a subcommand to act as the main command.

In particular, this class is what allows ``manim`` to act as ``manim render``.

.. note::
    This is a vendored version of https://github.com/click-contrib/click-default-group/
    under the BSD 3-Clause "New" or "Revised" License.

    This library isn't used as a dependency, as we need to inherit from
    :class:`cloup.Group` instead of :class:`click.Group`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable

import cloup

from manim.utils.deprecation import deprecated

__all__ = ["DefaultGroup"]

if TYPE_CHECKING:
    from click import Command, Context


class DefaultGroup(cloup.Group):
    """Invokes a subcommand marked with ``default=True`` if any subcommand is not
    chosen.

    Parameters
    ----------
    *args
        Positional arguments to forward to :class:`cloup.Group`.
    **kwargs
        Keyword arguments to forward to :class:`cloup.Group`. The keyword
        ``ignore_unknown_options`` must be set to ``False``.

    Attributes
    ----------
    default_cmd_name : str | None
        The name of the default command, if specified through the ``default``
        keyword argument. Otherwise, this is set to ``None``.
    default_if_no_args : bool
        Whether to include or not the default command, if no command arguments
        are supplied. This can be specified through the ``default_if_no_args``
        keyword argument. Default is ``False``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        # To resolve as the default command.
        if not kwargs.get("ignore_unknown_options", True):
            raise ValueError("Default group accepts unknown options")
        self.ignore_unknown_options = True
        self.default_cmd_name: str | None = kwargs.pop("default", None)
        self.default_if_no_args: bool = kwargs.pop("default_if_no_args", False)
        super().__init__(*args, **kwargs)

    def set_default_command(self, command: Command) -> None:
        """Sets a command function as the default command.

        Parameters
        ----------
        command
            The command to set as default.
        """
        cmd_name = command.name
        self.add_command(command)
        self.default_cmd_name = cmd_name

    def parse_args(self, ctx: Context, args: list[str]) -> list[str]:
        """Parses the list of ``args`` by forwarding it to
        :meth:`cloup.Group.parse_args`. Before doing so, if
        :attr:`default_if_no_args` is set to ``True`` and ``args`` is empty,
        this function appends to it the name of the default command specified
        by :attr:`default_cmd_name`.

        Parameters
        ----------
        ctx
            The Click context.
        args
            A list of arguments. If it's empty and :attr:`default_if_no_args`
            is ``True``, append the name of the default command to it.

        Returns
        -------
        list[str]
            The parsed arguments.
        """
        if not args and self.default_if_no_args and self.default_cmd_name:
            args.insert(0, self.default_cmd_name)
        parsed_args: list[str] = super().parse_args(ctx, args)
        return parsed_args

    def get_command(self, ctx: Context, cmd_name: str) -> Command | None:
        """Get a command function by its name, by forwarding the arguments to
        :meth:`cloup.Group.get_command`. If ``cmd_name`` does not match any of
        the command names in :attr:`commands`, attempt to get the default command
        instead.

        Parameters
        ----------
        ctx
            The Click context.
        cmd_name
            The name of the command to get.

        Returns
        -------
        :class:`click.Command` | None
            The command, if found. Otherwise, ``None``.
        """
        if cmd_name not in self.commands and self.default_cmd_name:
            # No command name matched.
            ctx.meta["arg0"] = cmd_name
            cmd_name = self.default_cmd_name
        return super().get_command(ctx, cmd_name)

    def resolve_command(
        self, ctx: Context, args: list[str]
    ) -> tuple[str | None, Command | None, list[str]]:
        """Given a list of ``args`` given by a CLI, find a command which
        matches the first element, and return its name (``cmd_name``), the
        command function itself (``cmd``) and the rest of the arguments which
        shall be passed to the function (``cmd_args``). If not found, return
        ``None``, ``None`` and the rest of the arguments.

        After resolving the command, if the Click context given by ``ctx``
        contains an ``arg0`` attribute in its :attr:`click.Context.meta`
        dictionary, insert it as the first element of the returned
        ``cmd_args``.

        Parameters
        ----------
        ctx
            The Click context.
        cmd_name
            The name of the command to get.

        Returns
        -------
        cmd_name : str | None
            The command name, if found. Otherwise, ``None``.
        cmd : :class:`click.Command` | None
            The command, if found. Otherwise, ``None``.
        cmd_args : list[str]
            The rest of the arguments to be passed to ``cmd``.
        """
        cmd_name, cmd, args = super().resolve_command(ctx, args)
        if "arg0" in ctx.meta:
            args.insert(0, ctx.meta["arg0"])
            if cmd is not None:
                cmd_name = cmd.name
        return cmd_name, cmd, args

    @deprecated
    def command(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., object]], Command]:
        """Return a decorator which converts any function into the default
        subcommand for this :class:`DefaultGroup`.

        .. warning::
            This method is deprecated. Use the ``default`` parameter of
            :class:`DefaultGroup` or :meth:`set_default_command` instead.

        Parameters
        ----------
        *args
            Positional arguments to pass to :meth:`cloup.Group.command`.
        **kwargs
            Keyword arguments to pass to :meth:`cloup.Group.command`.

        Returns
        -------
        Callable[[Callable[..., object]], click.Command]
            A decorator which transforms its input into this
            :class:`DefaultGroup`'s default subcommand.
        """
        default = kwargs.pop("default", False)
        decorator: Callable[[Callable[..., object]], Command] = super().command(
            *args, **kwargs
        )
        if not default:
            return decorator
        warnings.warn(
            "Use default param of DefaultGroup or set_default_command() instead",
            DeprecationWarning,
            stacklevel=1,
        )

        def _decorator(f: Callable) -> Command:
            cmd = decorator(f)
            self.set_default_command(cmd)
            return cmd

        return _decorator
