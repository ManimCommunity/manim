"""Manim's project initialization and management subcommands.

init -  The init subcommand is a quick and easy way to initialize a project
        Ite copies 2 files and pastes them in the current working dir

new  -  The new command group has 2 commands. these commands handle project creation
        and scene creation or insertion

        project -   The project subcommand is used for project creation
                    This command is similar to init but different in a way
                    that it asks for project name and template name.
                    init command initializes a new project in the current dir
                    while project command creates a new folder and creates the
                    project there

        scene   -   The scene subcommand command is used for inserting new scenes.
                    scene command can create new files and insert new scenes in there.
                    It can also insert new scenes in an already existing python file.
"""

import configparser
import curses
from pathlib import Path
from shutil import copyfile

import click

from ... import console
from ...constants import CONTEXT_SETTINGS, EPILOG, QUALITIES

CFG_DEFAULTS = {
    "frame_rate": 60,
    "background_color": "BLACK",
    "background_opacity": 1,
    "scene_names": "default",
    "resolution": (1280, 720),
}

cfg_resolutions = []


def update_cfg(cfg, project_cfg_path):
    """Updates the manim.cfg file in the newly created project
    Parameters
    ----------
    cfg : :dictionary:
        Look at `CFG_DEFAULTS`
        dictionary of values that will be edited in the manim.cfg file
    project_cfg_path : :Path:
        path of the project manim.cfg file
    """
    config = configparser.ConfigParser()
    config.read(project_cfg_path)
    cli_config = config["CLI"]
    for key, value in cfg.items():
        if key == "resolution":
            cli_config["pixel_height"] = str(value[0])
            cli_config["pixel_width"] = str(value[1])
        else:
            cli_config[key] = str(value)

    with open(project_cfg_path, "w") as conf:
        config.write(conf)


def copy_template_files(project_dir=Path("."), template_name="default"):
    """Copies template files for the new project creation from the templates
    Parameters
    ----------
    project_dir : class : Path
        Directory where template files will be copied
    template_name : :class:`str`, optional
        if template_name is not given or is given but does not exist `default.py` will be used as template for main.py
    """
    template_cfg_path = Path.resolve(Path(__file__).parent / "templates/template.cfg")
    template_scene_path = Path.resolve(
        Path(__file__).parent / f"templates/{template_name}.py"
    )

    if not template_scene_path.exists():
        template_scene_path = Path.resolve(
            Path(__file__).parent / "templates/default.py"
        )

    copyfile(template_cfg_path, Path.resolve(project_dir / "manim.cfg"))
    console.print("\n...copied [green]manim.cfg[/green]\n")

    copyfile(template_scene_path, Path.resolve(project_dir / "main.py"))
    console.print("\n...copied [green]main.py[/green]\n")


def resolution_to_tuple(str):
    resolution = str.split("x", 1)
    return (resolution[0], resolution[1])


@click.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,
    epilog=EPILOG,
    help="Initialize New Project",
)
@click.argument("project_name", type=Path, required=False)
@click.argument("template_name", required=False)
@click.option(
    "-d",
    "--default",
    "default_settings",
    is_flag=True,
    help="Default settings for Project Initialization",
    nargs=1,
)
def project(default_settings, **args):
    if args["project_name"]:
        project_name = args["project_name"]
    else:
        project_name = click.prompt("Project Name", type=Path)

    if project_name.is_dir():
        console.print(
            f"\nFolder [red]{project_name}[/red] exists. Please type another name\n"
        )
    else:
        project_name.mkdir()
        new_cfg = dict()

        if not default_settings:
            for key, value in CFG_DEFAULTS.items():
                if key == "scene_names":
                    if args["template_name"]:
                        new_cfg[key] = args["template_name"]
                    else:
                        new_cfg[key] = value
                elif key == "resolution":
                    resolution = click.prompt("\nResolution", default="1280x720")
                    new_cfg[key] = resolution_to_tuple(resolution)
                else:
                    new_cfg[key] = click.prompt(f"\n{key}", default=value)

            copy_template_files(project_name, args["template_name"])
            update_cfg(new_cfg, Path.resolve(project_name / "manim.cfg"))
        else:
            copy_template_files(project_name, args["template_name"])
            update_cfg(CFG_DEFAULTS, Path.resolve(project_name / "manim.cfg"))
            console.print(default_settings)


@click.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,
    epilog=EPILOG,
    help="Quickly setup a basic project",
)
@click.argument("scene_name", type=str, required=False)
def scene():
    pass


@click.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,
    epilog=EPILOG,
    help="Quickly setup a basic project",
)
def init():
    """Initialize a new project in the current working directory"""
    cfg = Path("manim.cfg")
    if cfg.exists():
        console.print("\t[red]manim.cfg exists[/red]\n")
    else:
        copy_template_files()


@click.group(
    context_settings=CONTEXT_SETTINGS,
    invoke_without_command=True,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Create Project or Scene.",
)
@click.pass_context
def new(ctx):
    pass


new.add_command(project)
