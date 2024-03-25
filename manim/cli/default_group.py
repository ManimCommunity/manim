"""``DefaultGroup`` allows a subcommand to act as the main command.

In particular, this class is what allows ``manim`` to act as ``manim render``.

.. note::
    This is a vendored version of https://github.com/click-contrib/click-default-group/
    under the BSD 3-Clause "New" or "Revised" License.

    This library isn't used as a dependency as we need to inherit from ``cloup.Group`` instead
    of ``click.Group``.
"""

import warnings

import cloup

__all__ = ["DefaultGroup"]


class DefaultGroup(cloup.Group):
    """Invokes a subcommand marked with ``default=True`` if any subcommand not
    chosen.
    """

    def __init__(self, *args, **kwargs):
        # To resolve as the default command.
        if not kwargs.get("ignore_unknown_options", True):
            raise ValueError("Default group accepts unknown options")
        self.ignore_unknown_options = True
        self.default_cmd_name = kwargs.pop("default", None)
        self.default_if_no_args = kwargs.pop("default_if_no_args", False)
        super().__init__(*args, **kwargs)

    def set_default_command(self, command):
        """Sets a command function as the default command."""
        cmd_name = command.name
        self.add_command(command)
        self.default_cmd_name = cmd_name

    def parse_args(self, ctx, args):
        if not args and self.default_if_no_args:
            args.insert(0, self.default_cmd_name)
        return super().parse_args(ctx, args)

    def get_command(self, ctx, cmd_name):
        if cmd_name not in self.commands:
            # No command name matched.
            ctx.arg0 = cmd_name
            cmd_name = self.default_cmd_name
        return super().get_command(ctx, cmd_name)

    def resolve_command(self, ctx, args):
        base = super()
        cmd_name, cmd, args = base.resolve_command(ctx, args)
        if hasattr(ctx, "arg0"):
            args.insert(0, ctx.arg0)
            cmd_name = cmd.name
        return cmd_name, cmd, args

    def command(self, *args, **kwargs):
        default = kwargs.pop("default", False)
        decorator = super().command(*args, **kwargs)
        if not default:
            return decorator
        warnings.warn(
            "Use default param of DefaultGroup or set_default_command() instead",
            DeprecationWarning,
        )

        def _decorator(f):
            cmd = decorator(f)
            self.set_default_command(cmd)
            return cmd

        return _decorator
