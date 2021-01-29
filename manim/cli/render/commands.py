import click

from manim.constants import EPILOG
from manim.constants import CONTEXT_SETTINGS


@click.command(
    context_settings={"max_content_width": 2000}.update(CONTEXT_SETTINGS),
    help="Renders scene(s) from the input file",
    epilog=EPILOG,
)
@click.argument("file", required=False)
@click.argument("scenes", required=False, nargs=-1)
@click.option(
    "-o",
    "--output",
    multiple=True,
    help="Specify the filename(s) of the output scene(s)"
)
@click.option(
    "-p",
    "--preview",
    is_flag=True,
    help="Automatically open the file(s) when rendered",
)
@click.option(
    "-f",
    "--show_in_file_browser",
    is_flag=True,
    help="Show the output file in the File Browse",
)
@click.option("--sound", is_flag=True, help="Play a success/failure sound")
# @click.option("--leave_progress_bars", help="Leave progress bars displayed in terminal")
@click.option(
    "-a", "--write_all", is_flag=True, help="Write all the scenes from a file"
)
@click.option(
    "-w",
    "--write_to_movie",
    is_flag=True,
    default=True,
    help="Render the scene as a movie file",
)
@click.option(
    "-s",
    "--save_last_frame",
    is_flag=True,
    help="Save only the last frame",
)
@click.option("-g", "--save_pngs", is_flag=True, help="Save each frame as a png")
@click.option("-i", "--save_as_gif", is_flag=True, help="Save the video as gif")
@click.option(
    "--disable_caching",
    is_flag=True,
    help="Disable caching (will generate partial-movie-files anyway)",
)
@click.option(
    "--flush_cache", is_flag=True, help="Remove all cached partial-movie-files"
)
@click.option("--log_to_file", is_flag=True, help="Log terminal output to file")
@click.option("-c", "--background_color", help="Specify background color")
@click.option("--media_dir", help="Directory to store media (including video files)")
@click.option("--log_dir", help="Directory to store log files")
@click.option("--tex_template", help="Specify a custom TeX template file")
@click.option(
    "--dry_run",
    is_flag=True,
    help="Do a dry run (render scenes but generate no output files)",
)
@click.option(
    "-t", "--transparent", is_flag=True, help="Render a scene with an alpha channel"
)
@click.option(
    "-q",
    "--quality",
    help="Render at specific quality, short form of the --*_quality flags",
)
# @click.option("--low_quality", help="Renderatlowquality")
# @click.option("--medium_quality", help="Renderatmediumquality")
# @click.option("--high_quality", help="Renderathighquality")
# @click.option("--production_quality", help="Renderatdefaultproductionquality")
# @click.option("--fourk_quality", help="Renderat4Kquality")
# @click.option("-l", help="DEPRECATED:USE-qlor--qualityl")
# @click.option("-m", help="DEPRECATED:USE-qmor--qualitym")
# @click.option("-e", help="DEPRECATED:USE-qhor--qualityh")
# @click.option("-k", help="DEPRECATED:USE-qkor--qualityk")
@click.option(
    "-r",
    "--resolution",
    help="Resolution, passed as 'height,width'. Overrides the -l, -m, -e, and -k flags, if present",
)
@click.option(
    "-n",
    "--from_animation_number",
    help="Start rendering at the specified animation index, instead of the first animation. If you pass in two comma separated values, e.g. '3,6', it will end the rendering at the second value ",
)
@click.option(
    "--use_js_renderer",
    is_flag=True,
    help="Render animations using the javascript frontend",
)
@click.option("--js_renderer_path", help="Path to the javascript frontend")
@click.option("--config_file", help="Specify the configuration file")
@click.option(
    "--custom_folders",
    help="Use the folders defined in the [custom_folders] section of the config file to define the output folder structure",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    show_default=True,
    help="""
    Verbosity counter (-vv...). Changes ffmpeg log level unless 5+\n
    {0:NONE,1:DEBUG,2:INFO,3:WARNING,4:ERROR,5+:CRITICAL}
    """,
)
@click.option(
    "--progress_bar",
    default="display",
    show_default=True,
    type=click.Choice(
        [
            "display",
            "leave",
        ],
        case_sensitive=False,
    ),
    help="Display progress bars, and/or keep them displayed",
)
def render(
    file,
    scenes,
    output,
    preview,
    show_in_file_browser,
    sound,
    write_all,
    write_to_movie,
    save_last_frame,
    save_pngs,
    save_as_gif,
    disable_caching,
    flush_cache,
    log_to_file,
    background_color,
    media_dir,
    log_dir,
    tex_template,
    dry_run,
    transparent,
    quality,
    resolution,
    from_animation_number,
    use_js_renderer,
    js_renderer_path,
    config_file,
    custom_folders,
    verbose,
    progress_bar,
):
    click.echo("render")
    print(
        file,
        scenes,
        output,
        preview,
        show_in_file_browser,
        sound,
        write_all,
        write_to_movie,
        save_last_frame,
        save_pngs,
        save_as_gif,
        disable_caching,
        flush_cache,
        log_to_file,
        background_color,
        media_dir,
        log_dir,
        tex_template,
        dry_run,
        transparent,
        quality,
        resolution,
        from_animation_number,
        use_js_renderer,
        js_renderer_path,
        config_file,
        custom_folders,
        verbose,
        progress_bar,
    )
