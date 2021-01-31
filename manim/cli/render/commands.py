import os
import click

from manim.constants import EPILOG
from manim.constants import CONTEXT_SETTINGS

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup


@click.group(
    invoke_without_command=True,
    context_settings=CONTEXT_SETTINGS,
    epilog=EPILOG,
)
@click.argument("file", required=False)
@click.argument("scenes", required=False, nargs=-1)
@optgroup.group("Global options")
@optgroup.option(
    "--config_file", type=click.File(), help="Specify the configuration file."
)
@optgroup.option(
    "--custom_folders",
    is_flag=True,
    help="Use the folders defined in the [custom_folders] section of the config file to define the output folder structure.",
)
@optgroup.option(
    "--disable_caching",
    is_flag=True,
    help="Disable the use of the cache (still generates cache files).",
)
@optgroup.option(
    "--tex_template", type=click.File(), help="Specify a custom TeX template file."
)
@optgroup.option(
    "-v",
    "--verbose",
    count=True,
    show_default=True,
    help="""
    Verbosity counter (-vv...). Changes ffmpeg log level unless 5+.
   
    {0:NONE,1:DEBUG,2:INFO,3:WARNING,4:ERROR,5+:CRITICAL}
    """,
)
@optgroup.group("Output options")
@optgroup.option(
    "-o",
    "--output",
    multiple=True,
    help="Specify the filename(s) of the rendered scene(s).",
)
@optgroup.option(
    "--media_dir", type=click.Path(), help="Path to store rendered videos and latex."
)
@optgroup.option("--log_dir", type=click.Path(), help="Path to store render logs.")
@optgroup.option(
    "--log_to_file",
    default=True,
    show_default=True,
    is_flag=True,
    help="Log terminal output to file",
)
@optgroup.group("Rendering Options")
@optgroup.option(
    "-n",
    "--from_animation_number",
    nargs=2,
    type=int,
    help="Start rendering from n_0 until n_1. If n_1 is left unspecified, renders all scenes after n_0.",
)
@optgroup.option(
    "-f",
    "--format",
    default="mp4",
    type=click.Choice(
        [
            "png",
            "gif",
            "mp4",
        ],
        case_sensitive=False,
    ),
)
@optgroup.option(
    "-q",
    "--quality",
    default="p",
    type=click.Choice(
        [
            "l",
            "m",
            "h",
            "p",
            "k",
        ],
        case_sensitive=False,
    ),
    help="""
    Render quality at the follow resolution framerates, respectively:
    854x480 30FPS, 
    1280x720 30FPS,
    1920x1080 60FPS,
    2560x1440 60FPS,
    3840x2160 60FPS
    """,
)
@optgroup.option(
    "-r",
    "--resolution",
    nargs=2,
    type=int,
    help="Resolution in (W,H) for when 16:9 aspect ratio isn't possible.",
)
@optgroup.option(
    "--fps",
    default=30,
    show_default=True,
    type=float,
    help="Render at this frame rate.",
)
@optgroup.option(
    "--webgl_renderer",
    show_default=True,
    default=os.getcwd(),
    type=click.Path(),
    help="Render scenes using the WebGL frontend. Requires a path to the WebGL frontend."
)
@optgroup.option(
    "-t", "--transparent", is_flag=True, help="Render scenes with alpha channel."
)
@optgroup.option(
    "-c",
    "--background_color",
    show_default=True,
    default="000000",
    help="Render scenes with background color.",
)
@optgroup.group("Ease of access options")
@optgroup.option(
    "--progress_bar",
    default="display",
    show_default=True,
    type=click.Choice(
        [
            "display",
            "leave",
            "none",
        ],
        case_sensitive=False,
    ),
    help="Display progress bars and/or keep them displayed.",
)
@optgroup.option(
    "-p",
    "--preview",
    is_flag=True,
    help="Preview the rendered file(s) in default player.",
)
@optgroup.option(
    "-f",
    "--show_in_file_browser",
    is_flag=True,
    help="Show the output file in the file browser.",
)
@optgroup.option("--sound", is_flag=True, help="Play a success/failure sound.")
@click.pass_context
def render(
    ctx,
    file,
    scenes,
    config_file,
    custom_folders,
    disable_caching,
    tex_template,
    verbose,
    output,
    media_dir,
    log_dir,
    log_to_file,
    from_animation_number,
    format,
    quality,
    resolution,
    fps,
    webgl_renderer,
    transparent,
    background_color,
    progress_bar,
    preview,
    show_in_file_browser,
    sound,
):
    """Render SCENE(S) from the input FILE.

    FILE is the file path of the script.

    SCENES is an optional list of scenes in the file.
    """
    click.echo("render")
    print(
        ctx,
        file,
        scenes,
        config_file,
        custom_folders,
        disable_caching,
        tex_template,
        verbose,
        output,
        media_dir,
        log_dir,
        log_to_file,
        from_animation_number,
        format,
        quality,
        resolution,
        fps,
        webgl_renderer,
        transparent,
        background_color,
        progress_bar,
        preview,
        show_in_file_browser,
        sound,
        sep="\n",
    )


@render.command(
    context_settings=CONTEXT_SETTINGS, help="Remove cached partial movie files."
)
def clear():
    click.echo("clearing cache")
