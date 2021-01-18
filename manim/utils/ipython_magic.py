"""Utilities for using Manim with IPython (in particular: Jupyter notebooks)"""

import mimetypes
from pathlib import Path

from IPython.core.magic import Magics, magics_class, line_cell_magic, needs_local_scope
from IPython.display import display, Image, Video

from manim import config, tempconfig

from .._config.main_utils import parse_args


@magics_class
class ManimMagic(Magics):
    @needs_local_scope
    @line_cell_magic
    def manim(self, line, cell=None, local_ns=None):
        r"""Render Manim scenes contained in IPython cells.
        Works as a line or cell magic.

        .. note::

            This line and cell magic works best when used in a JupyterLab environment:
            while all of the functionality is available for classic Jupyter notebooks
            as well, it is possible that videos sometimes don't update on repeated
            execution of the same cell if the scene name stays the same.

            This problem does not occur when using JupyterLab.

        Please refer to `<https://jupyter.org/>`_ for more information about JupyterLab
        and Jupyter notebooks.

        Usage in line mode::

            %manim MyAwesomeScene [CLI options]

        Usage in cell mode::

            %%manim MyAwesomeScene [CLI options]

            class MyAweseomeScene(Scene):
                def construct(self):
                    ...

        Run ``%manim -h`` for possible command line interface options.
        """
        if cell:
            exec(cell, local_ns)

        cli_args = ["manim", ""] + line.split()
        if len(cli_args) == 2:
            # empty line.split(): no commands have been passed, call with -h
            cli_args.append("-h")

        try:
            args = parse_args(cli_args)
        except SystemExit:
            return  # probably manim -h was called, process ended preemptively

        with tempconfig(local_ns.get("config", {})):
            config.digest_args(args)

            exec(f"{config['scene_names'][0]}().render()", local_ns)
            local_path = Path(config["output_file"]).relative_to(Path.cwd())

            file_type = mimetypes.guess_type(Path(config["output_file"]))[0]
            if file_type.startswith("image"):
                display(Image(filename=config["output_file"]))
                return

            display(
                Video(
                    local_path,
                    html_attributes='controls autoplay loop style="max-width: 100%;"',
                )
            )
