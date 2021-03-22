import os
import sys
import traceback

from manim import logger, console, config, __version__
from manim.utils.module_ops import (
    get_module,
    get_scene_classes_from_module,
    get_scenes_to_render,
    scene_classes_from_file,
)

from manim._config.main_utils import parse_args


def main():
    console.print(f"Manim Community [green]v{__version__}[/green]")
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

        if config["use_opengl_renderer"]:
            from manim.renderer.opengl_renderer import OpenGLRenderer

            for SceneClass in scene_classes_from_file(input_file):
                try:
                    renderer = OpenGLRenderer()
                    scene = SceneClass(renderer)
                    scene.render()
                except Exception:
                    console.print_exception()
        elif config["use_webgl_renderer"]:
            try:
                from manim.grpc.impl import frame_server_impl

                server = frame_server_impl.get(input_file)
                server.start()
                server.wait_for_termination()
            except ModuleNotFoundError:
                console.print(
                    "Dependencies for the WebGL render are missing. Run "
                    "pip install manim[webgl_renderer] to install them."
                )
                console.print_exception()
        else:
            for SceneClass in scene_classes_from_file(input_file):
                try:
                    scene = SceneClass()
                    scene.render()
                except Exception:
                    console.print_exception()


if __name__ == "__main__":
    main()
