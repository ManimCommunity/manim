"""Manim's cfg subcommand.

Manim's cfg subcommand is accessed in the command-line interface via ``manim
cfg``. Here you can specify options, subcommands, and subgroups for the cfg
group.

"""

from __future__ import annotations

import contextlib
from ast import literal_eval
from pathlib import Path
from typing import Any, cast

import cloup
from rich.errors import StyleSyntaxError
from rich.style import Style

from manim._config import cli_ctx_settings, console
from manim._config.utils import config_file_paths, make_config_parser
from manim.constants import EPILOG
from manim.utils.file_ops import guarantee_existence, open_file

RICH_COLOUR_INSTRUCTIONS: str = """
[red]The default colour is used by the input statement.
If left empty, the default colour will be used.[/red]
[magenta] For a full list of styles, visit[/magenta] [green]https://rich.readthedocs.io/en/latest/style.html[/green]
"""
RICH_NON_STYLE_ENTRIES: list[str] = ["log.width", "log.height", "log.timestamps"]

__all__ = [
    "value_from_string",
    "value_from_string",
    "is_valid_style",
    "replace_keys",
    "cfg",
    "write",
    "show",
    "export",
]


def value_from_string(value: str) -> str | int | bool:
    """Extract the literal of proper datatype from a ``value`` string.

    Parameters
    ----------
    value
        The value to check get the literal from.

    Returns
    -------
    :class:`str` | :class:`int` | :class:`bool`
        The literal of appropriate datatype.
    """
    with contextlib.suppress(SyntaxError, ValueError):
        value = literal_eval(value)
    return value


def _is_expected_datatype(
    value: str, expected: str, validate_style: bool = False
) -> bool:
    """Check whether the literal from ``value`` is the same datatype as the
    literal from ``expected``. If ``validate_style`` is ``True``, also check if
    the style given by ``value`` is valid, according to ``rich``.

    Parameters
    ----------
    value
        The string of the value to check, obtained from reading the user input.
    expected
        The string of the literal datatype which must be matched by ``value``.
        This is obtained from reading the ``cfg`` file.
    validate_style
        Whether or not to confirm if ``value`` is a valid style, according to
        ``rich``. Default is ``False``.

    Returns
    -------
    :class:`bool`
        Whether or not the literal from ``value`` matches the datatype of the
        literal from ``expected``.
    """
    value_literal = value_from_string(value)
    ExpectedLiteralType = type(value_from_string(expected))

    return isinstance(value_literal, ExpectedLiteralType) and (
        (isinstance(value_literal, str) and is_valid_style(value_literal))
        if validate_style
        else True
    )


def is_valid_style(style: str) -> bool:
    """Checks whether the entered color style is valid, according to ``rich``.

    Parameters
    ----------
    style
        The style to check whether it is valid.

    Returns
    -------
    :class:`bool`
        Whether the color style is valid or not, according to ``rich``.
    """
    try:
        Style.parse(style)
        return True
    except StyleSyntaxError:
        return False


def replace_keys(default: dict[str, Any]) -> dict[str, Any]:
    """Replace ``_`` with ``.`` and vice versa in a dictionary's keys for
    ``rich``.

    Parameters
    ----------
    default
        The dictionary whose keys will be checked and replaced.

    Returns
    -------
    :class:`dict`
        The dictionary whose keys are modified by replacing ``_`` with ``.``
        and vice versa.
    """
    for key in default:
        if "_" in key:
            temp = default[key]
            del default[key]
            key = key.replace("_", ".")
            default[key] = temp
        else:
            temp = default[key]
            del default[key]
            key = key.replace(".", "_")
            default[key] = temp
    return default


@cloup.group(
    context_settings=cli_ctx_settings,
    invoke_without_command=True,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages Manim configuration files.",
)
@cloup.pass_context
def cfg(ctx: cloup.Context) -> None:
    """Responsible for the cfg subcommand."""
    pass


@cfg.command(context_settings=cli_ctx_settings, no_args_is_help=True)
@cloup.option(
    "-l",
    "--level",
    type=cloup.Choice(["user", "cwd"], case_sensitive=False),
    default="cwd",
    help="Specify if this config is for user or the working directory.",
)
@cloup.option("-o", "--open", "openfile", is_flag=True)
def write(level: str | None = None, openfile: bool = False) -> None:
    config_paths = config_file_paths()
    console.print(
        "[yellow bold]Manim Configuration File Writer[/yellow bold]",
        justify="center",
    )

    USER_CONFIG_MSG = f"""A configuration file at [yellow]{config_paths[1]}[/yellow] has been created with your required changes.
This will be used when running the manim command. If you want to override this config,
you will have to create a manim.cfg in the local directory, where you want those changes to be overridden."""

    CWD_CONFIG_MSG = f"""A configuration file at [yellow]{config_paths[2]}[/yellow] has been created.
To save your config please save that file and place it in your current working directory, from where you run the manim command."""

    parser = make_config_parser()
    if not openfile:
        action = "save this as"
        for category in parser:
            console.print(f"{category}", style="bold green underline")
            default = cast(dict[str, Any], parser[category])
            if category == "logger":
                console.print(RICH_COLOUR_INSTRUCTIONS)
                default = replace_keys(default)

            for key in default:
                # All the cfg entries for logger need to be validated as styles,
                # as long as they aren't setting the log width or height etc
                if category == "logger" and key not in RICH_NON_STYLE_ENTRIES:
                    desc = "style"
                    style = default[key]
                else:
                    desc = "value"
                    style = None

                console.print(f"Enter the {desc} for {key} ", style=style, end="")
                if category != "logger" or key in RICH_NON_STYLE_ENTRIES:
                    defaultval = (
                        repr(default[key])
                        if isinstance(value_from_string(default[key]), str)
                        else default[key]
                    )
                    console.print(f"(defaults to {defaultval}) :", end="")
                try:
                    temp = input()
                except EOFError:
                    raise Exception(
                        """Not enough values in input.
You may have added a new entry to default.cfg, in which case you will have to
modify write_cfg_subcmd_input to account for it.""",
                    ) from None
                if temp:
                    while temp and not _is_expected_datatype(
                        temp,
                        default[key],
                        bool(style),
                    ):
                        console.print(
                            f"[red bold]Invalid {desc}. Try again.[/red bold]",
                        )
                        console.print(
                            f"Enter the {desc} for {key}:",
                            style=style,
                            end="",
                        )
                        temp = input()

                    default[key] = temp.replace("%", "%%")

            default = replace_keys(default) if category == "logger" else default

            parser[category] = {
                i: v.replace("%", "%%") for i, v in dict(default).items()
            }

    else:
        action = "open"

    if level is None:
        console.print(
            f"Do you want to {action} the default config for this User?(y/n)[[n]]",
            style="dim purple",
            end="",
        )
        action_to_userpath = input()
    else:
        action_to_userpath = ""

    if action_to_userpath.lower() == "y" or level == "user":
        cfg_file_path = config_paths[1]
        guarantee_existence(config_paths[1].parents[0])
        console.print(USER_CONFIG_MSG)
    else:
        cfg_file_path = config_paths[2]
        guarantee_existence(config_paths[2].parents[0])
        console.print(CWD_CONFIG_MSG)
    with cfg_file_path.open("w") as fp:
        parser.write(fp)
    if openfile:
        open_file(cfg_file_path)


@cfg.command(context_settings=cli_ctx_settings)
def show() -> None:
    parser = make_config_parser()
    rich_non_style_entries = [a.replace(".", "_") for a in RICH_NON_STYLE_ENTRIES]
    for category in parser:
        console.print(f"{category}", style="bold green underline")
        for entry in parser[category]:
            if category == "logger" and entry not in rich_non_style_entries:
                console.print(f"{entry} :", end="")
                console.print(
                    f" {parser[category][entry]}",
                    style=parser[category][entry],
                )
            else:
                console.print(f"{entry} : {parser[category][entry]}")
        console.print("\n")


@cfg.command(context_settings=cli_ctx_settings)
@cloup.option("-d", "--directory", default=Path.cwd())
@cloup.pass_context
def export(ctx: cloup.Context, directory: str) -> None:
    directory_path = Path(directory)
    if directory_path.absolute == Path.cwd().absolute:
        console.print(
            """You are reading the config from the same directory you are exporting to.
This means that the exported config will overwrite the config for this directory.
Are you sure you want to continue? (y/n)""",
            style="red bold",
            end="",
        )
        proceed = input().lower() == "y"
    else:
        proceed = True
    if proceed:
        if not directory_path.is_dir():
            console.print(f"Creating folder: {directory}.", style="red bold")
            directory_path.mkdir(parents=True)

        ctx.invoke(write)
        from_path = Path.cwd() / "manim.cfg"
        to_path = directory_path / "manim.cfg"

        console.print(f"Exported final Config at {from_path} to {to_path}.")
    else:
        console.print("Aborted...", style="red bold")
