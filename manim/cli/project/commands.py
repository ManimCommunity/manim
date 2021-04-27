"""Manim's project initialization and management subcommands.

init -  The init subcommand is a quick and easy way to initialize a project
        It copies files from templates dir and pastes them in the current working dir

new  -  The new command group has 2 commands. these commands handle project creation
        and scene creation or insertion

        project -   The project subcommand is used for project creation
                    This command is similar to init but different in a way
                    that it asks for project_name and template_name.
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

# when -d flag is passed in 'project new' command CFG_DEFAULTS is passed to update_cfg() function
CFG_DEFAULTS = {
    "frame_rate": 30,
    "background_color": "BLACK",
    "background_opacity": 1,
    "scene_names": "default",
    "resolution": (854, 480),
}

"""\
    utility functions are helper functions that provide some core functionality
    to do project management

    functions:
    1 - update_cfg:
            used for updating manim.cfg file when -d flag is not provided in
            the 'manim new project' command

    2 - copy_template_files:
            copies template file upon successful project creation.
            it is used in both 'manim init' and 'manim project new' commands

    3 - select_resolution:
            when 'manim project new' command is issued without the -d flag.
            this function does conversions of 'QUALITIES' constant in to a
            workable format for the click.Choice prompt. after the prompt it
            returns a tuple containing pixel width and height based on the
            result from the prompt.
"""


def update_cfg(cfg_dict, project_cfg_path):
    """Updates the manim.cfg file after reading it from the project_cfg_path.

    Args:
        cfg (dict): values used to update manim.cfg found project_cfg_path
        project_cfg_path (Path): Path of manim.cfg file
    """
    config = configparser.ConfigParser()
    config.read(project_cfg_path)
    cli_config = config["CLI"]
    for key, value in cfg_dict.items():
        if key == "resolution":
            cli_config["pixel_height"] = str(value[0])
            cli_config["pixel_width"] = str(value[1])
        else:
            cli_config[key] = str(value)

    with open(project_cfg_path, "w") as conf:
        config.write(conf)


def add_class_name(file, template_name, class_name="Main", isNewFile=True):
    with open(file, "r+") as f:
        import_line = ""
        if isNewFile:
            import_line += "from manim import *"
        content = f.read()

        content = content.replace(f"{template_name}", class_name, 1)
        f.seek(0, 0)
        f.write(import_line.rstrip("\r\n") + "\n\n\n" + content)


def copy_template_files(project_dir=Path("."), template_name="default"):
    """Copies template files from templates dir to project_dir.

    Args:
        project_dir (Path, optional): [description]. Defaults to Path(".").
        template_name (str, optional): [description]. Defaults to "default".
    """
    template_cfg_path = Path.resolve(Path(__file__).parent / "templates/template.cfg")
    if not template_cfg_path.exists():
        raise FileNotFoundError(f"{template_cfg_path} does not exist")

    template_scene_path = Path.resolve(
        Path(__file__).parent / f"templates/{template_name}.py"
    )
    if not template_scene_path.exists():
        template_scene_path = Path.resolve(
            Path(__file__).parent / "templates/default.py"
        )

    copyfile(template_cfg_path, Path.resolve(project_dir / "manim.cfg"))
    console.print("\n\t[green]copied[/green] [blue]manim.cfg[/blue]\n")
    copyfile(template_scene_path, Path.resolve(project_dir / "main.py"))
    console.print("\n\t[green]copied[/green] [blue]main.py[/blue]\n")
    add_class_name(Path.resolve(project_dir / "main.py"), template_name)


# select_resolution() called inside project command
def select_resolution():
    """prompts input of type click.Choice from user

    Returns:
        tuple: tuple containing height and width
    """
    resolution_options = []
    for quality in QUALITIES.items():
        resolution_options.append(
            (quality[1]["pixel_height"], quality[1]["pixel_width"])
        )
    resolution_options.pop()
    choice = click.prompt(
        "\nSelect resolution:\n",
        type=click.Choice([f"{i[0]}p" for i in resolution_options]),
        show_default=False,
        default="480p",
    )
    return [res for res in resolution_options if f"{res[0]}p" == choice][0]


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

    if args["template_name"]:
        template_name = args["template_name"]
    else:
        # in the future when implementing a full template system. Choices are going to be saved in some sort of config file for templates
        template_name = click.prompt(
            "Template",
            type=click.Choice(["default", "Graph", "MovingCamera"], False),
            default="default",
        )

    if project_name.is_dir():
        console.print(
            f"\nFolder [red]{project_name}[/red] exists. Please type another name\n"
        )
    else:
        project_name.mkdir()
        new_cfg = dict()
        new_cfg_path = Path.resolve(project_name / "manim.cfg")

        if not default_settings:
            for key, value in CFG_DEFAULTS.items():
                if key == "scene_names":
                    if args["template_name"]:
                        new_cfg[key] = args["template_name"]
                    else:
                        new_cfg[key] = value
                elif key == "resolution":
                    new_cfg[key] = select_resolution()
                else:
                    new_cfg[key] = click.prompt(f"\n{key}", default=value)

            console.print("\n", new_cfg)
            if click.confirm("Do you want to continue?", default=True, abort=True):
                copy_template_files(project_name, template_name)
                update_cfg(new_cfg, new_cfg_path)
        else:
            copy_template_files(project_name, template_name)
            update_cfg(CFG_DEFAULTS, new_cfg_path)


@click.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Add a scene to an existing file or a new file",
)
@click.argument("scene_name", type=str, required=True)
@click.argument("template_name", type=str, required=False)
@click.argument("file_name", type=Path, required=False)
def scene(**args):
    if args["template_name"]:
        console.print(f"template name given{args['template_name']}")
    else:
        console.print("template name not given")
    pass


@click.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=False,
    epilog=EPILOG,
    help="Quickly setup a project",
)
def init():
    """Initialize a new project in the current working directory"""
    cfg = Path("manim.cfg")
    if cfg.exists():
        raise FileExistsError(f"\t{cfg} exists\n")
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
new.add_command(scene)
