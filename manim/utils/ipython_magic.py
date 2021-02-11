"""Utilities for using Manim with IPython (in particular: Jupyter notebooks)"""

import mimetypes
from pathlib import Path

from manim.__main__ import main
from click.testing import CliRunner

try:
    from IPython.core.magic import (
        Magics,
        magics_class,
        line_cell_magic,
        needs_local_scope,
    )
    from IPython.display import display, Image, Video
except ImportError:
    pass
else:

    @magics_class
    class ManimMagic(Magics):
        def __init__(self, shell):
            super(ManimMagic, self).__init__(shell)
            self.rendered_files = dict()

        @needs_local_scope
        @line_cell_magic
        def manim(self, line, cell=None, local_ns=None):
            r"""Render Manim scenes contained in IPython cells.
            Works as a line or cell magic.

            .. note::

                This line and cell magic works best when used in a JupyterLab
                environment: while all of the functionality is available for
                classic Jupyter notebooks as well, it is possible that videos
                sometimes don't update on repeated execution of the same cell
                if the scene name stays the same.

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
            args = line.split()
            if not len(args) or "-h" in args or "--help" in args or "--version" in args:
                main.main(args, standalone_mode=False)
                return

            runner = CliRunner()  # This runs the command.
            result = runner.invoke(main, args, input=cell)

            config = main.main(
                ["--jupyter"] + args, standalone_mode=False
            )  # This runs the render subcommand, but returns config
            file = Path(config.output_file)

            file_type = mimetypes.guess_type(file)[0]
            if file_type.startswith("image"):
                display(Image(filename=config["output_file"]))
                return

            display(
                Video(
                    file,
                    html_attributes='controls autoplay loop style="max-width: 100%;"',
                    embed=True,
                )
            )
