"""Manim's cfg subcommand.

Manim's cfg subcommand is accessed in the command-line interface via ``manim
cfg``. Here you can specify options, subcommands, and subgroups for the cfg
group.

"""
import os
from ast import literal_eval
from typing import Union

import click
from rich.errors import StyleSyntaxError
from rich.style import Style

from ... import config, console
from ..._config.utils import config_file_paths, make_config_parser
from ...constants import CONTEXT_SETTINGS, EPILOG
from ...utils.file_ops import guarantee_existence, open_file

RICH_COLOUR_INSTRUCTIONS: str = """
[red]The default colour is used by the input statement.
If left empty, the default colour will be used.[/red]
[magenta] For a full list of styles, visit[/magenta] [green]https://rich.readthedocs.io/en/latest/style.html[/green]
"""
RICH_NON_STYLE_ENTRIES: str = ["log.width", "log.height", "log.timestamps"]


def value_from_string(value: str) -> Union[str, int, bool]:
    """Extracts the literal of proper datatype from a string.
    Parameters
    ----------
    value : :class:`str`
        The value to check get the literal from.

    Returns
    -------
    Union[:class:`str`, :class:`int`, :class:`bool`]
        Returns the literal of appropriate datatype.
    """
    try:
        value = literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    return value


def _is_expected_datatype(value: str, expected: str, style: bool = False) -> bool:
    """Checks whether `value` is the same datatype as `expected`,
    and checks if it is a valid `style` if `style` is true.

    Parameters
    ----------
    value : :class:`str`
        The string of the value to check (obtained from reading the user input).
    expected : :class:`str`
        The string of the literal datatype must be matched by `value`. Obtained from
        reading the cfg file.
    style : :class:`bool`, optional
        Whether or not to confirm if `value` is a style, by default False

    Returns
    -------
    :class:`bool`
        Whether or not `value` matches the datatype of `expected`.
    """
    value = value_from_string(value)
    expected = type(value_from_string(expected))

    return isinstance(value, expected) and (is_valid_style(value) if style else True)


def is_valid_style(style: str) -> bool:
    """Checks whether the entered color is a valid color according to rich
    Parameters
    ----------
    style : :class:`str`
        The style to check whether it is valid.
    Returns
    -------
    Boolean
        Returns whether it is valid style or not according to rich.
    """
    try:
        Style.parse(style)
        return True
    except StyleSyntaxError:
        return False


def replace_keys(default: dict) -> dict:
    """Replaces _ to . and vice versa in a dictionary for rich
    Parameters
    ----------
    default : :class:`dict`
        The dictionary to check and replace
    Returns
    -------
    :class:`dict`
        The dictionary which is modified by replacing _ with . and vice versa
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


@click.group(
    context_settings=CONTEXT_SETTINGS,
    invoke_without_command=True,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages Manim configuration files.",
)
@click.pass_context
def cfg(ctx):
    """Responsible for the cfg subcommand."""
    pass


@cfg.command(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.option(
    "-l",
    "--level",
    type=click.Choice(["user", "cwd"], case_sensitive=False),
    default="cwd",
    help="Specify if this config is for user or the working directory.",
)
@click.option("-o", "--open", "openfile", is_flag=True)
def write(level: str = None, openfile: bool = False) -> None:
    config_paths = config_file_paths()
    console.print(
        "[yellow bold]Manim Configuration File Writer[/yellow bold]", justify="center"
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
            default = parser[category]
            if category == "logger":
                console.print(RICH_COLOUR_INSTRUCTIONS)
                default = replace_keys(default)

            for key in default:
                # All the cfg entries for logger need to be validated as styles,
                # as long as they arent setting the log width or height etc
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
modify write_cfg_subcmd_input to account for it."""
                    )
                if temp:
                    while temp and not _is_expected_datatype(
                        temp, default[key], bool(style)
                    ):
                        console.print(
                            f"[red bold]Invalid {desc}. Try again.[/red bold]"
                        )
                        console.print(
                            f"Enter the {desc} for {key}:", style=style, end=""
                        )
                        temp = input()
                    else:
                        default[key] = temp

            default = replace_keys(default) if category == "logger" else default

            parser[category] = dict(default)

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
    with open(cfg_file_path, "w") as fp:
        parser.write(fp)
    if openfile:
        open_file(cfg_file_path)


@cfg.command(context_settings=CONTEXT_SETTINGS)
def show():
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


@cfg.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--directory", default=os.getcwd())
@click.pass_context
def export(ctx, directory):
    if os.path.abspath(directory) == os.path.abspath(os.getcwd()):
        console.print(
            """You are reading the config from the same directory you are exporting to.
This means that the exported config will overwrite the config for this directory.
Are you sure you want to continue? (y/n)""",
            style="red bold",
            end="",
        )
        proceed = True if input().lower() == "y" else False
    else:
        proceed = True
    if proceed:
        if not os.path.isdir(directory):
            console.print(f"Creating folder: {directory}.", style="red bold")
            os.mkdir(directory)
        with open(os.path.join(directory, "manim.cfg"), "w") as outpath:
            ctx.invoke(write)
            from_path = os.path.join(os.getcwd(), "manim.cfg")
            to_path = os.path.join(directory, "manim.cfg")
        console.print(f"Exported final Config at {from_path} to {to_path}.")
    else:
        console.print("Aborted...", style="red bold")
