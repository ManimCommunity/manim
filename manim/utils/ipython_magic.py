"""Utilities for using Manim with IPython (in particular: Jupyter notebooks)"""

import hashlib
import mimetypes
import os
import shutil
from pathlib import Path

from manim import config, tempconfig
from manim.__main__ import main

try:
    from IPython import get_ipython
    from IPython.core.magic import (
        Magics,
        line_cell_magic,
        magics_class,
        needs_local_scope,
    )
    from IPython.display import Image, Video, display
except ImportError:
    pass
else:

    @magics_class
    class ManimMagic(Magics):
        def __init__(self, shell):
            super(ManimMagic, self).__init__(shell)
            self.rendered_files = {}

        @needs_local_scope
        @line_cell_magic
        def manim(self, line, cell=None, local_ns=None):
            r"""Render Manim scenes contained in IPython cells.
            Works as a line or cell magic.

            .. hint::

                This line and cell magic works best when used in a JupyterLab
                environment: while all of the functionality is available for
                classic Jupyter notebooks as well, it is possible that videos
                sometimes don't update on repeated execution of the same cell
                if the scene name stays the same.

                This problem does not occur when using JupyterLab.

            Please refer to `<https://jupyter.org/>`_ for more information about JupyterLab
            and Jupyter notebooks.

            Usage in line mode::

                %manim [CLI options] MyAwesomeScene

            Usage in cell mode::

                %%manim [CLI options] MyAwesomeScene

                class MyAweseomeScene(Scene):
                    def construct(self):
                        ...

            Run ``%manim -h`` and ``%manim render -h`` for possible command line interface options.

            .. note::

                The maximal width of the rendered videos that are displayed in the notebook can be
                configured via the ``media_width`` configuration option. The default is set to ``25vw``,
                which is 25% of your current viewport width. To allow the output to become as large
                as possible, set ``config.media_width = "100%"``.

            Examples
            --------

            First make sure to put ``import manim``, or even ``from manim import *``
            in a cell and evaluate it. Then, a typical Jupyter notebook cell for Manim
            could look as follows::

                %%manim -v WARNING --disable_caching -qm BannerExample

                config.media_width = "75%"

                class BannerExample(Scene):
                    def construct(self):
                        self.camera.background_color = "#ece6e2"
                        banner_large = ManimBanner(dark_theme=False).scale(0.7)
                        self.play(banner_large.create())
                        self.play(banner_large.expand())

            Evaluating this cell will render and display the ``BannerExample`` scene defined in the body of the cell.

            .. note::

                In case you want to hide the red box containing the output progress bar, the ``progress_bar`` config
                option should be set to ``None``. This can also be done by passing ``--progress_bar None`` as a
                CLI flag.

            """
            if cell:
                exec(cell, local_ns)

            args = line.split()
            if not len(args) or "-h" in args or "--help" in args or "--version" in args:
                main(args, standalone_mode=False, prog_name="manim")
                return
            modified_args = ["--jupyter"] + args[:-1] + [""] + [args[-1]]
            args = main(modified_args, standalone_mode=False, prog_name="manim")
            with tempconfig(local_ns.get("config", {})):
                config.digest_args(args)
                exec(f"{config['scene_names'][0]}().render()", local_ns)
                local_path = Path(config["output_file"]).relative_to(Path.cwd())
                tmpfile = (
                    Path(config["media_dir"])
                    / "jupyter"
                    / f"{_video_hash(local_path)}{local_path.suffix}"
                )

                if local_path in self.rendered_files:
                    self.rendered_files[local_path].unlink()
                self.rendered_files[local_path] = tmpfile
                os.makedirs(tmpfile.parent, exist_ok=True)
                shutil.copy(local_path, tmpfile)

                file_type = mimetypes.guess_type(config["output_file"])[0]
                if file_type.startswith("image"):
                    display(Image(filename=config["output_file"]))
                    return

                # videos need to be embedded when running in google colab
                video_embed = "google.colab" in str(get_ipython())

                display(
                    Video(
                        tmpfile,
                        html_attributes=f'controls autoplay loop style="max-width: {config["media_width"]};"',
                        embed=video_embed,
                    )
                )


def _video_hash(path):
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()
