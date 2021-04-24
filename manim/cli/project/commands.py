"""Manim's project initialization and management subcommands.

init -  The init subcommand is a quick and easy way to initialize a project
        Ite copies 2 files and pastes them in the current working dir

new  -  The new command group has 2 commands. these commands handle project creation
        and scene creation

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
from pathlib import Path
from shutil import copyfile

import click

from ... import console
from ...constants import CONTEXT_SETTINGS, EPILOG

cfg_vars = [
    "frame_rate",
    "resolution",
    "background_color",
    "background_opacity",
    "scene_names",
]


def update_cfg(vars, values, project_cfg_path):
    """Updates the manim.cfg file in the newly created project
    Parameters
    ----------
    vars : list
        list of variables that are to be edited after files are copied.
    values : list
        list of values that will be edited in the manim.cfg file
    """
    config = configparser.ConfigParser()
    config.read(project_cfg_path)
    cli_config = config["CLI"]
    for num, var in enumerate(vars, start=0):
        cli_config[var] = values[num]

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
    Note:
        In the future templates may be downloaded over the internet
        and users may be able to create and share templates
    """
    init_default_cfg_path = Path.resolve(
        Path(__file__).parent / "templates/template.cfg"
    )
    default_template = Path.resolve(
        Path(__file__).parent / f"templates/{template_name}.py"
    )

    copyfile(init_default_cfg_path, Path.resolve(project_dir / "manim.cfg"))
    console.print("...copied [green]manim.cfg[/green]\n")

    copyfile(default_template, Path.resolve(project_dir / "main.py"))
    console.print("...copied [green]main.py[/green]\n")
