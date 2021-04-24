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
from pathlib import Path
from shutil import copyfile

import click

from ... import console
from ...constants import CONTEXT_SETTINGS, EPILOG, QUALITIES

cfg_vars = {
    "frame_rate": 60,
    "resolution": (1920, 1200),
    "background_color": "BLACK",
    "background_opacity": 1,
    "scene_names": "DefaultScene",
}

cfg_resolutions = []


def update_cfg(cfg_vars, project_cfg_path):
    """Updates the manim.cfg file in the newly created project
    Parameters
    ----------
    values : dictionary : Look at `cfg_vars`
        dictionary of values that will be edited in the manim.cfg file
    """
    config = configparser.ConfigParser()
    config.read(project_cfg_path)
    cli_config = config["CLI"]
    for key, value in cfg_vars.items():
        if key == "resolution":
            cli_config["pixel_height"] = value[0]
            cli_config["pixel_width"] = value[1]
        else:
            cli_config[key] = value

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
    template_cfg_path = Path.resolve(Path(__file__).parent / "templates/template.cfg")
    template_scene_path = Path.resolve(
        Path(__file__).parent / f"templates/{template_name}.py"
    )

    if not template_scene_path.exists():
        template_scene_path = Path.resolve(
            Path(__file__).parent / f"templates/{template_name}.py"
        )

    copyfile(template_cfg_path, Path.resolve(project_dir / "manim.cfg"))
    console.print("...copied [green]manim.cfg[/green]\n")

    copyfile(template_scene_path, Path.resolve(project_dir / "main.py"))
    console.print("...copied [green]main.py[/green]\n")


def select_resolution(stdscr):
    for key, quality in QUALITIES.items():
        cfg_resolutions.append((key, quality))

    attributes = {}
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    attributes["normal"] = curses.color_pair(1)

    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
    attributes["highlighted"] = curses.color_pair(2)

    c = 0  # last character read
    option = 0  # the current option that is marked
    while c != 10:  # Enter in ascii
        stdscr.erase()
        stdscr.addstr("Select rendring quality?\n", curses.A_UNDERLINE)
        for i in range(len(cfg_resolutions)):
            if i == option:
                attr = attributes["highlighted"]
            else:
                attr = attributes["normal"]
            stdscr.addstr("{0}. ".format(i + 1))
            stdscr.addstr(cfg_resolutions[i][0] + "\n", attr)
        c = stdscr.getch()
        if c == curses.KEY_UP and option > 0:
            option -= 1
        elif c == curses.KEY_DOWN and option < len(cfg_resolutions) - 1:
            option += 1

    stdscr.addstr("You chose {0}".format(cfg_resolutions[option][0]))
    stdscr.getch()


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
        console.print("Project Name Exists")
    else:
        project_name.mkdir()
        new_cfg_vars = dict()

        if not default_settings:
            for key, value in cfg_vars.items():
                if key == "scene_names":
                    pass
                elif key == "resolution":
                    pass
                else:
                    new_cfg_vars[key] = click.prompt(f"{key}")

            copy_template_files(project_name)
            update_cfg(new_cfg_vars, Path.resolve(project_name / "manim.cfg"))

        else:
            console.print(default_settings)
