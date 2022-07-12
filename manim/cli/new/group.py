from __future__ import annotations

import configparser
from pathlib import Path

import click
import cloup

from ... import console
from ...constants import CONTEXT_SETTINGS, EPILOG, QUALITIES
from ...utils.file_ops import (
    add_import_statement,
    copy_template_files,
    get_template_names,
    get_template_path,
)

CFG_DEFAULTS = {
    "frame_rate": 30,
    "background_color": "BLACK",
    "background_opacity": 1,
    "scene_names": "Default",
    "resolution": (854, 480),
}


def select_resolution():
    """Prompts input of type click.Choice from user. Presents options from QUALITIES constant.

    Returns
    -------
        :class:`tuple`
            Tuple containing height and width.
    """
    resolution_options = []
    for quality in QUALITIES.items():
        resolution_options.append(
            (quality[1]["pixel_height"], quality[1]["pixel_width"]),
        )
    resolution_options.pop()
    choice = click.prompt(
        "\nSelect resolution:\n",
        type=click.Choice([f"{i[0]}p" for i in resolution_options]),
        show_default=False,
        default="480p",
    )
    return [res for res in resolution_options if f"{res[0]}p" == choice][0]


def update_cfg(cfg_dict, project_cfg_path):
    """Updates the manim.cfg file after reading it from the project_cfg_path.

    Parameters
    ----------
    cfg : :class:`dict`
        values used to update manim.cfg found project_cfg_path.
    project_cfg_path : :class:`Path`
        Path of manim.cfg file.
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


@cloup.command(
    context_settings=CONTEXT_SETTINGS,
    epilog=EPILOG,
)
@cloup.argument("project_name", type=Path, required=False)
@cloup.option(
    "-d",
    "--default",
    "default_settings",
    is_flag=True,
    help="Default settings for project creation.",
    nargs=1,
)
def project(default_settings, **args):
    """Creates a new project.

    PROJECT_NAME is the name of the folder in which the new project will be initialized.
    """
    if args["project_name"]:
        project_name = args["project_name"]
    else:
        project_name = click.prompt("Project Name", type=Path)

    # in the future when implementing a full template system. Choices are going to be saved in some sort of config file for templates
    template_name = click.prompt(
        "Template",
        type=click.Choice(get_template_names(), False),
        default="Default",
    )

    if project_name.is_dir():
        console.print(
            f"\nFolder [red]{project_name}[/red] exists. Please type another name\n",
        )
    else:
        project_name.mkdir()
        new_cfg = {}
        new_cfg_path = Path.resolve(project_name / "manim.cfg")

        if not default_settings:
            for key, value in CFG_DEFAULTS.items():
                if key == "scene_names":
                    new_cfg[key] = template_name + "Template"
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


@cloup.command(
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
    epilog=EPILOG,
)
@cloup.argument("scene_name", type=str, required=True)
@cloup.argument("file_name", type=str, required=False)
def scene(**args):
    """Inserts a SCENE to an existing FILE or creates a new FILE.

    SCENE is the name of the scene that will be inserted.

    FILE is the name of file in which the SCENE will be inserted.
    """
    if not Path("main.py").exists():
        raise FileNotFoundError(f"{Path('main.py')} : Not a valid project directory.")

    template_name = click.prompt(
        "template",
        type=click.Choice(get_template_names(), False),
        default="Default",
    )
    scene = ""
    with open(Path.resolve(get_template_path() / f"{template_name}.mtp")) as f:
        scene = f.read()
        scene = scene.replace(template_name + "Template", args["scene_name"], 1)

    if args["file_name"]:
        file_name = Path(args["file_name"] + ".py")

        if file_name.is_file():
            # file exists so we are going to append new scene to that file
            with open(file_name, "a") as f:
                f.write("\n\n\n" + scene)
        else:
            # file does not exist so we create a new file, append the scene and prepend the import statement
            with open(file_name, "w") as f:
                f.write("\n\n\n" + scene)

            add_import_statement(file_name)
    else:
        # file name is not provided so we assume it is main.py
        # if main.py does not exist we do not continue
        with open(Path("main.py"), "a") as f:
            f.write("\n\n\n" + scene)


@cloup.group(
    context_settings=CONTEXT_SETTINGS,
    invoke_without_command=True,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Create a new project or insert a new scene.",
    deprecated=True,
)
@cloup.pass_context
def new(ctx):
    pass


new.add_command(project)
new.add_command(scene)
