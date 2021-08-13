"""Utilities for using Manim with IPython (in particular: Jupyter notebooks)"""

import mimetypes
import os
import shutil
import webbrowser
from datetime import datetime
from pathlib import Path

from manim import Group, config, logger, tempconfig
from manim.__main__ import main
from manim.renderer.shader import shader_program_cache


def init_buttons():
    b0 = Button(
        description="",
        icon="fa-picture-o",
        button_style="",
        layout=Layout(height="30px", width="40px"),
    )
    b1 = Button(
        description="",
        icon="fa-expand",
        button_style="",
        layout=Layout(height="30px", width="40px"),
    )
    b2 = Button(
        description="",
        icon="fa-download",
        button_style="",
        layout=Layout(height="30px", width="40px"),
    )
    return b0, b1, b2


def image_viewer(image_path, small_width, large_width):

    original1t1_width = f"{config.frame_size[0]}px"
    file = open(image_path, "rb")
    image = file.read()
    dis_img = widgets.Image(value=image, format="png")

    button_list = GridspecLayout(3, 1, height="120px")
    b0, b1, b2 = init_buttons()
    button_list[0, 0] = b0
    button_list[1, 0] = b1
    button_list[2, 0] = b2
    dis_img.width = large_width  # # default width

    def on_button_image1t1_clicked(b):
        dis_img.width = original1t1_width

    def on_button_expand_clicked(b):
        if dis_img.width != small_width:
            dis_img.width = small_width
        else:
            dis_img.width = large_width

    def on_button_download_clicked(b):
        url = (Path.cwd() / image_path).as_uri()
        webbrowser.open(url)

    b0.on_click(on_button_image1t1_clicked)
    b1.on_click(on_button_expand_clicked)
    b2.on_click(on_button_download_clicked)
    return AppLayout(
        left_sidebar=button_list, center=dis_img, pane_widths=["50px", 1, 0]
    )


def video_viewer(video_path, small_width, large_width):

    original1t1_width = f"{config.frame_size[0]}px"

    dis_video = Video.from_file(video_path)
    dis_video.controls = True

    dis_but = widgets.ToggleButton(
        value=False,
        description="Fullscreen",
        disabled=False,
        button_style="",
        tooltip="Description",
    )

    b0, b1, b2 = init_buttons()

    button_list = GridspecLayout(3, 1, height="120px")
    button_list[0, 0] = b0
    button_list[1, 0] = b1
    button_list[2, 0] = b2

    dis_video.width = large_width  # default width

    def on_button_image1t1_clicked(b):
        dis_video.width = original1t1_width

    def on_button_expand_clicked(b):
        if dis_video.width != small_width:
            dis_video.width = small_width
        else:
            dis_video.width = large_width

    def on_button_download_clicked(b):
        url = (Path.cwd() / video_path).as_uri()
        webbrowser.open(url)

    b0.on_click(on_button_image1t1_clicked)
    b1.on_click(on_button_expand_clicked)
    b2.on_click(on_button_download_clicked)
    return AppLayout(
        left_sidebar=button_list, center=dis_video, pane_widths=["50px", 1, 0]
    )


try:
    from IPython import get_ipython
    from IPython.core.magic import (
        Magics,
        line_cell_magic,
        magics_class,
        needs_local_scope,
    )
    from IPython.display import display
    from ipywidgets import AppLayout, Button, GridspecLayout, Layout, Video, widgets

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

            Run ``%manim --help`` and ``%manim render --help`` for possible command line interface options.

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
            modified_args = self.add_additional_args(args)
            args = main(modified_args, standalone_mode=False, prog_name="manim")
            with tempconfig(local_ns.get("config", {})):
                config.digest_args(args)

                renderer = None
                if config.renderer == "opengl":
                    # Check if the imported mobjects extend the OpenGLMobject class
                    # meaning ConvertToOpenGL did its job
                    if "OpenGLMobject" in map(lambda cls: cls.__name__, Group.mro()):
                        from manim.renderer.opengl_renderer import OpenGLRenderer

                        renderer = OpenGLRenderer()
                    else:
                        logger.warning(
                            "Renderer must be set to OpenGL in the configuration file "
                            "before importing Manim! Using cairo renderer instead."
                        )
                        config.renderer = "cairo"

                SceneClass = local_ns[config["scene_names"][0]]
                scene = SceneClass(renderer)
                scene.render()

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

                if file_type.startswith("image"):
                    display(image_viewer(tmpfile, "350px", "900px"))
                else:
                    display(video_viewer(tmpfile, "350px", "900px"))

        def add_additional_args(self, args):
            additional_args = ["--jupyter"]
            # Use webm to support transparency
            if "-t" in args and "--format" not in args:
                additional_args += ["--format", "webm"]
            return additional_args + args[:-1] + [""] + [args[-1]]


def _generate_file_name():
    return config["scene_names"][0] + "@" + datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
