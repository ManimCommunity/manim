"""Utilities for using Manim with IPython (in particular: Jupyter notebooks)"""

from __future__ import annotations

import mimetypes
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from manim import Group, config, logger, tempconfig
from manim.__main__ import main
from manim.renderer.shader import shader_program_cache

try:
    from IPython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell
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
        def __init__(self, shell: InteractiveShell) -> None:
            super().__init__(shell)
            self.rendered_files = {}

        @needs_local_scope
        @line_cell_magic
        def manim(
            self,
            line: str,
            cell: str = None,
            local_ns: dict[str, Any] = None,
        ) -> None:
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

            Run ``%manim --help`` and ``%manim render --help`` for possible command line interface options.

            .. note::

                The maximal width of the rendered videos that are displayed in the notebook can be
                configured via the ``media_width`` configuration option. The default is set to ``25vw``,
                which is 25% of your current viewport width. To allow the output to become as large
                as possible, set ``config.media_width = "100%"``.

                The ``media_embed`` option will embed the image/video output in the notebook. This is
                generally undesirable as it makes the notebooks very large, but is required on some
                platforms (notably Google's CoLab, where it is automatically enabled unless suppressed
                by ``config.embed = False``) and needed in cases when the notebook (or converted HTML
                file) will be moved relative to the video locations. Use-cases include building
                documentation with Sphinx and JupyterBook. See also the :mod:`manim directive for Sphinx
                <manim.utils.docbuild.manim_directive>`.

            Examples
            --------

            First make sure to put ``import manim``, or even ``from manim import *``
            in a cell and evaluate it. Then, a typical Jupyter notebook cell for Manim
            could look as follows::

                %%manim -v WARNING --disable_caching -qm BannerExample

                config.media_width = "75%"
                config.media_embed = True

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

            modified_args = self.add_additional_args(args)
            args = main(modified_args, standalone_mode=False, prog_name="manim")
            with tempconfig(local_ns.get("config", {})):
                config.digest_args(args)

                renderer = None
                if config.renderer == "opengl":
                    # Check if the imported mobjects extend the OpenGLMobject class
                    # meaning ConvertToOpenGL did its job
                    if "OpenGLMobject" in [
                        parent_class.__name__ for parent_class in Group.mro()
                    ]:
                        from manim.renderer.opengl_renderer import OpenGLRenderer

                        renderer = OpenGLRenderer()
                    else:
                        logger.warning(
                            "Renderer must be set to OpenGL in the configuration file "
                            "before importing Manim! Using cairo renderer instead.",
                        )
                        config.renderer = "cairo"

                try:
                    SceneClass = local_ns[config["scene_names"][0]]
                    scene = SceneClass(renderer=renderer)
                    scene.render()
                finally:
                    # Shader cache becomes invalid as the context is destroyed
                    shader_program_cache.clear()

                    # Close OpenGL window here instead of waiting for the main thread to
                    # finish causing the window to stay open and freeze
                    if renderer is not None and renderer.window is not None:
                        renderer.window.close()

                if config["output_file"] is None:
                    logger.info("No output file produced")
                    return

                local_path = Path(config["output_file"]).relative_to(Path.cwd())
                tmpfile = (
                    Path(config["media_dir"])
                    / "jupyter"
                    / f"{_generate_file_name()}{local_path.suffix}"
                )

                if local_path in self.rendered_files:
                    self.rendered_files[local_path].unlink()
                self.rendered_files[local_path] = tmpfile
                os.makedirs(tmpfile.parent, exist_ok=True)
                shutil.copy(local_path, tmpfile)

                file_type = mimetypes.guess_type(config["output_file"])[0]
                embed = config["media_embed"]
                if embed is None:
                    # videos need to be embedded when running in google colab.
                    # do this automatically in case config.media_embed has not been
                    # set explicitly.
                    embed = "google.colab" in str(get_ipython())

                if file_type.startswith("image"):
                    result = Image(filename=config["output_file"])
                else:
                    result = Video(
                        tmpfile,
                        html_attributes=f'controls autoplay loop style="max-width: {config["media_width"]};"',
                        embed=embed,
                    )

                display(result)

        def add_additional_args(self, args: list[str]) -> list[str]:
            additional_args = ["--jupyter"]
            # Use webm to support transparency
            if "-t" in args and "--format" not in args:
                additional_args += ["--format", "webm"]
            return additional_args + args[:-1] + [""] + [args[-1]]


def _generate_file_name() -> str:
    return config["scene_names"][0] + "@" + datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
