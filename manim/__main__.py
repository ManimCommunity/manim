import os
import sys
import traceback
import click

from manim import logger, config
from manim.utils.module_ops import (
    get_module,
    get_scene_classes_from_module,
    get_scenes_to_render,
    scene_classes_from_file,
)
from manim._config import cfg_subcmds
from manim.plugins.plugins_flags import list_plugins
from manim.utils.file_ops import open_file as open_media_file
from manim._config.main_utils import parse_args

try:
    from manim.grpc.impl import frame_server_impl
except ImportError:
    frame_server_impl = None


def open_file_if_needed(file_writer):
    if config["verbosity"] != "DEBUG":
        curr_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    open_file = any([config["preview"], config["show_in_file_browser"]])

    if open_file:
        file_paths = []

        if config["save_last_frame"]:
            file_paths.append(file_writer.image_file_path)
        if config["write_to_movie"] and not config["save_as_gif"]:
            file_paths.append(file_writer.movie_file_path)
        if config["save_as_gif"]:
            file_paths.append(file_writer.gif_file_path)

        for file_path in file_paths:
            if config["show_in_file_browser"]:
                open_media_file(file_path, True)
            if config["preview"]:
                open_media_file(file_path, False)

    if config["verbosity"] != "DEBUG":
        sys.stdout.close()
        sys.stdout = curr_stdout


def main():
    args = parse_args(sys.argv)

    if hasattr(args, "cmd"):
        if args.cmd == "cfg":
            if args.subcmd:
                from manim._config import cfg_subcmds

                if args.subcmd == "write":
                    cfg_subcmds.write(args.level, args.open)
                elif args.subcmd == "show":
                    cfg_subcmds.show()
                elif args.subcmd == "export":
                    cfg_subcmds.export(args.dir)
            else:
                logger.error("No subcommand provided; Exiting...")

        elif args.cmd == "plugins":
            from manim.plugins import plugins_flags

            if args.list:
                plugins_flags.list_plugins()
            elif not args.list:
                logger.error("No flag provided; Exiting...")

        # elif args.cmd == "some_other_cmd":
        #     something_else_here()

    else:
        config.digest_args(args)
        input_file = config.get_dir("input_file")
        if config["use_js_renderer"]:
            try:
                if frame_server_impl is None:
                    raise ImportError("Dependencies for JS renderer is not installed.")
                server = frame_server_impl.get(input_file)
                server.start()
                server.wait_for_termination()
            except Exception:
                print("\n\n")
                traceback.print_exc()
                print("\n\n")
        else:
            for SceneClass in scene_classes_from_file(input_file):
                try:
                    scene = SceneClass()
                    scene.render()
                    open_file_if_needed(scene.renderer.file_writer)
                except Exception:
                    print("\n\n")
                    traceback.print_exc()
                    print("\n\n")


class SkipArg(click.Group):
    def parse_args(self, ctx, args):
        if not args:
            return click.Group().parse_args(ctx, [])
        if args[0] in self.commands:
            if len(args) == 1 or args[1] not in self.commands:
                args.insert(0, "")
        super(SkipArg, self).parse_args(ctx, args)


EPILOG = "Made with <3 by the manim community devs"
HELP_OPTIONS = dict(help_option_names=["-h", "--help"])


@click.group(
    cls=SkipArg,
    invoke_without_command=True,
    context_settings=HELP_OPTIONS,
    no_args_is_help=True,
    help="Animation engine for explanatory math videos",
    epilog=EPILOG,
)
@click.argument("file", required=False)
@click.version_option()
@click.pass_context
def cli(ctx, file):
    """The main entry point for the manim command."""
    print("main", ctx, file)


@cli.command(
    context_settings=HELP_OPTIONS,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages plugins",
)
@click.option("-l", "--list", is_flag=True, help="List available plugins")
def plugins(list):
    if list:
        list_plugins()


@cli.group(
    context_settings=HELP_OPTIONS,
    invoke_without_command=True,
    no_args_is_help=True,
    epilog=EPILOG,
    help="Manages config files",
)
def cfg():
    pass


@cfg.command(context_settings=HELP_OPTIONS, no_args_is_help=True)
@click.option(
    "-l",
    "--level",
    type=click.Choice(["user", "cwd"], case_sensitive=False),
    default="cwd",
    help="Specify if this config is for user or the working directory.",
)
@click.option("-o", "--open", is_flag=True)
def write(level, open):
    pass


@cfg.command(context_settings=HELP_OPTIONS)
def show():
    cfg_subcmds.show()


@cfg.command(context_settings=HELP_OPTIONS)
@click.option("-d", "--dir", default=os.getcwd())
def export(dir):
    cfg_subcmds.export(dir)


if __name__ == "__main__":
    main()
