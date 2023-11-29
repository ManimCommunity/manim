r"""
A directive for including Manim videos in a Sphinx document
===========================================================

When rendering the HTML documentation, the ``.. manim::`` directive
implemented here allows to include rendered videos.

Its basic usage that allows processing **inline content**
looks as follows::

    .. manim:: MyScene

        class MyScene(Scene):
            def construct(self):
                ...

It is required to pass the name of the class representing the
scene to be rendered to the directive.

As a second application, the directive can also be used to
render scenes that are defined within doctests, for example::

    .. manim:: DirectiveDoctestExample
        :ref_classes: Dot

        >>> from manim import Create, Dot, RED, Scene
        >>> dot = Dot(color=RED)
        >>> dot.color
        ManimColor('#FC6255')
        >>> class DirectiveDoctestExample(Scene):
        ...     def construct(self):
        ...         self.play(Create(dot))


Options
-------

Options can be passed as follows::

    .. manim:: <Class name>
        :<option name>: <value>

The following configuration options are supported by the
directive:

    hide_source
        If this flag is present without argument,
        the source code is not displayed above the rendered video.

    no_autoplay
        If this flag is present without argument,
        the video will not autoplay.

    quality : {'low', 'medium', 'high', 'fourk'}
        Controls render quality of the video, in analogy to
        the corresponding command line flags.

    save_as_gif
        If this flag is present without argument,
        the scene is rendered as a gif.

    save_last_frame
        If this flag is present without argument,
        an image representing the last frame of the scene will
        be rendered and displayed, instead of a video.

    ref_classes
        A list of classes, separated by spaces, that is
        rendered in a reference block after the source code.

    ref_functions
        A list of functions, separated by spaces,
        that is rendered in a reference block after the source code.

    ref_methods
        A list of methods, separated by spaces,
        that is rendered in a reference block after the source code.

"""
from __future__ import annotations

import csv
import itertools as it
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path
from timeit import timeit

import jinja2
from docutils import nodes
from docutils.parsers.rst import Directive, directives  # type: ignore
from docutils.statemachine import StringList

from manim import QUALITIES
from manim import __version__ as manim_version

classnamedict = {}


class SkipManimNode(nodes.Admonition, nodes.Element):
    """Auxiliary node class that is used when the ``skip-manim`` tag is present
    or ``.pot`` files are being built.

    Skips rendering the manim directive and outputs a placeholder instead.
    """

    pass


def visit(self, node, name=""):
    self.visit_admonition(node, name)
    if not isinstance(node[0], nodes.title):
        node.insert(0, nodes.title("skip-manim", "Example Placeholder"))


def depart(self, node):
    self.depart_admonition(node)


def process_name_list(option_input: str, reference_type: str) -> list[str]:
    r"""Reformats a string of space separated class names
    as a list of strings containing valid Sphinx references.

    Tests
    -----

    ::

        >>> process_name_list("Tex TexTemplate", "class")
        [':class:`~.Tex`', ':class:`~.TexTemplate`']
        >>> process_name_list("Scene.play Mobject.rotate", "func")
        [':func:`~.Scene.play`', ':func:`~.Mobject.rotate`']
    """
    return [f":{reference_type}:`~.{name}`" for name in option_input.split()]


class ManimDirective(Directive):
    r"""The manim directive, rendering videos while building
    the documentation.

    See the module docstring for documentation.
    """
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    option_spec = {
        "hide_source": bool,
        "no_autoplay": bool,
        "quality": lambda arg: directives.choice(
            arg,
            ("low", "medium", "high", "fourk"),
        ),
        "save_as_gif": bool,
        "save_last_frame": bool,
        "ref_modules": lambda arg: process_name_list(arg, "mod"),
        "ref_classes": lambda arg: process_name_list(arg, "class"),
        "ref_functions": lambda arg: process_name_list(arg, "func"),
        "ref_methods": lambda arg: process_name_list(arg, "meth"),
    }
    final_argument_whitespace = True

    def run(self):
        # Rendering is skipped if the tag skip-manim is present,
        # or if we are making the pot-files
        should_skip = (
            "skip-manim" in self.state.document.settings.env.app.builder.tags.tags
            or self.state.document.settings.env.app.builder.name == "gettext"
        )
        if should_skip:
            clsname = self.arguments[0]
            node = SkipManimNode()
            self.state.nested_parse(
                StringList(
                    [
                        f"Placeholder block for ``{clsname}``.",
                        "",
                        ".. code-block:: python",
                        "",
                    ]
                    + ["    " + line for line in self.content]
                    + [
                        "",
                        ".. raw:: html",
                        "",
                        f'    <pre data-manim-binder data-manim-classname="{clsname}">',
                    ]
                    + ["    " + line for line in self.content]
                    + ["    </pre>"],
                ),
                self.content_offset,
                node,
            )
            return [node]

        from manim import config, tempconfig

        global classnamedict

        clsname = self.arguments[0]
        if clsname not in classnamedict:
            classnamedict[clsname] = 1
        else:
            classnamedict[clsname] += 1

        hide_source = "hide_source" in self.options
        no_autoplay = "no_autoplay" in self.options
        save_as_gif = "save_as_gif" in self.options
        save_last_frame = "save_last_frame" in self.options
        assert not (save_as_gif and save_last_frame)

        ref_content = (
            self.options.get("ref_modules", [])
            + self.options.get("ref_classes", [])
            + self.options.get("ref_functions", [])
            + self.options.get("ref_methods", [])
        )
        if ref_content:
            ref_block = "References: " + " ".join(ref_content)

        else:
            ref_block = ""

        if "quality" in self.options:
            quality = f'{self.options["quality"]}_quality'
        else:
            quality = "example_quality"
        frame_rate = QUALITIES[quality]["frame_rate"]
        pixel_height = QUALITIES[quality]["pixel_height"]
        pixel_width = QUALITIES[quality]["pixel_width"]

        state_machine = self.state_machine
        document = state_machine.document

        source_file_name = Path(document.attributes["source"])
        source_rel_name = source_file_name.relative_to(setup.confdir)
        source_rel_dir = source_rel_name.parents[0]
        dest_dir = Path(setup.app.builder.outdir, source_rel_dir).absolute()
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)

        source_block = [
            ".. code-block:: python",
            "",
            "    from manim import *\n",
            *("    " + line for line in self.content),
            "",
            ".. raw:: html",
            "",
            f'    <pre data-manim-binder data-manim-classname="{clsname}">',
            *("    " + line for line in self.content),
            "",
            "    </pre>",
        ]
        source_block = "\n".join(source_block)

        config.media_dir = (Path(setup.confdir) / "media").absolute()
        config.images_dir = "{media_dir}/images"
        config.video_dir = "{media_dir}/videos/{quality}"
        output_file = f"{clsname}-{classnamedict[clsname]}"
        config.assets_dir = Path("_static")
        config.progress_bar = "none"
        config.verbosity = "WARNING"

        example_config = {
            "frame_rate": frame_rate,
            "no_autoplay": no_autoplay,
            "pixel_height": pixel_height,
            "pixel_width": pixel_width,
            "save_last_frame": save_last_frame,
            "write_to_movie": not save_last_frame,
            "output_file": output_file,
        }
        if save_last_frame:
            example_config["format"] = None
        if save_as_gif:
            example_config["format"] = "gif"

        user_code = self.content
        if user_code[0].startswith(">>> "):  # check whether block comes from doctest
            user_code = [
                line[4:] for line in user_code if line.startswith((">>> ", "... "))
            ]

        code = [
            "from manim import *",
            *user_code,
            f"{clsname}().render()",
        ]

        try:
            with tempconfig(example_config):
                run_time = timeit(lambda: exec("\n".join(code), globals()), number=1)
                video_dir = config.get_dir("video_dir")
                images_dir = config.get_dir("images_dir")
        except Exception as e:
            raise RuntimeError(f"Error while rendering example {clsname}") from e

        _write_rendering_stats(
            clsname,
            run_time,
            self.state.document.settings.env.docname,
        )

        # copy video file to output directory
        if not (save_as_gif or save_last_frame):
            filename = f"{output_file}.mp4"
            filesrc = video_dir / filename
            destfile = Path(dest_dir, filename)
            shutil.copyfile(filesrc, destfile)
        elif save_as_gif:
            filename = f"{output_file}.gif"
            filesrc = video_dir / filename
        elif save_last_frame:
            filename = f"{output_file}.png"
            filesrc = images_dir / filename
        else:
            raise ValueError("Invalid combination of render flags received.")
        rendered_template = jinja2.Template(TEMPLATE).render(
            clsname=clsname,
            clsname_lowercase=clsname.lower(),
            hide_source=hide_source,
            filesrc_rel=Path(filesrc).relative_to(setup.confdir).as_posix(),
            no_autoplay=no_autoplay,
            output_file=output_file,
            save_last_frame=save_last_frame,
            save_as_gif=save_as_gif,
            source_block=source_block,
            ref_block=ref_block,
        )
        state_machine.insert_input(
            rendered_template.split("\n"),
            source=document.attributes["source"],
        )

        return []


rendering_times_file_path = Path("../rendering_times.csv")


def _write_rendering_stats(scene_name, run_time, file_name):
    with rendering_times_file_path.open("a") as file:
        csv.writer(file).writerow(
            [
                re.sub(r"^(reference\/)|(manim\.)", "", file_name),
                scene_name,
                "%.3f" % run_time,
            ],
        )


def _log_rendering_times(*args):
    if rendering_times_file_path.exists():
        with rendering_times_file_path.open() as file:
            data = list(csv.reader(file))
        if len(data) == 0:
            sys.exit()

        print("\nRendering Summary\n-----------------\n")

        # filter out empty lists caused by csv reader
        data = [row for row in data if row]

        max_file_length = max(len(row[0]) for row in data)
        for key, group in it.groupby(data, key=lambda row: row[0]):
            key = key.ljust(max_file_length + 1, ".")
            group = list(group)
            if len(group) == 1:
                row = group[0]
                print(f"{key}{row[2].rjust(7, '.')}s {row[1]}")
                continue
            time_sum = sum(float(row[2]) for row in group)
            print(
                f"{key}{f'{time_sum:.3f}'.rjust(7, '.')}s  => {len(group)} EXAMPLES",
            )
            for row in group:
                print(f"{' '*(max_file_length)} {row[2].rjust(7)}s {row[1]}")
        print("")


def _delete_rendering_times(*args):
    if rendering_times_file_path.exists():
        rendering_times_file_path.unlink()


def setup(app):
    app.add_node(SkipManimNode, html=(visit, depart))

    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.add_directive("manim", ManimDirective)

    app.connect("builder-inited", _delete_rendering_times)
    app.connect("build-finished", _log_rendering_times)

    app.add_js_file("manim-binder.min.js")
    app.add_js_file(
        None,
        body=textwrap.dedent(
            f"""\
                window.initManimBinder({{branch: "v{manim_version}"}})
            """
        ).strip(),
    )

    metadata = {"parallel_read_safe": False, "parallel_write_safe": True}
    return metadata


TEMPLATE = r"""
{% if not hide_source %}
.. raw:: html

    <div id="{{ clsname_lowercase }}" class="admonition admonition-manim-example">
    <p class="admonition-title">Example: {{ clsname }} <a class="headerlink" href="#{{ clsname_lowercase }}">¶</a></p>

{% endif %}

{% if not (save_as_gif or save_last_frame) %}
.. raw:: html

    <video
        class="manim-video"
        controls
        loop
        {{ '' if no_autoplay else 'autoplay' }}
        src="./{{ output_file }}.mp4">
    </video>

{% elif save_as_gif %}
.. image:: /{{ filesrc_rel }}
    :align: center

{% elif save_last_frame %}
.. image:: /{{ filesrc_rel }}
    :align: center

{% endif %}
{% if not hide_source %}
{{ source_block }}

{{ ref_block }}

.. raw:: html

    </div>

{% endif %}
"""
