from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2
from docutils import nodes
from docutils.parsers.rst.directives import choice, flag
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri
from sphinx.util.typing import ExtensionMetadata

from manim import __version__ as manim_version
from manim._config import tempconfig
from manim.constants import QUALITIES
from manim.scene.scene import Scene

__all__ = ["ManimDirective"]


class ManimDirective(SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        "hide_source": flag,
        "no_autoplay": flag,
        "quality": lambda arg: choice(
            arg,
            [q.removesuffix("_quality") for q in QUALITIES],
        ),
        "save_as_gif": flag,
        "save_last_frame": flag,
        "ref_modules": lambda args: [f":mod:`~.{arg}`" for arg in args.split()],
        "ref_classes": lambda args: [f":class:`~.{arg}`" for arg in args.split()],
        "ref_functions": lambda args: [f":func:`~.{arg}`" for arg in args.split()],
        "ref_methods": lambda args: [f":meth:`~.{arg}`" for arg in args.split()],
    }

    def run(self) -> list[nodes.Element]:
        # OPTIONS
        clsname: str = self.arguments[0].strip()
        quality: str = self.options.get("quality", "example") + "_quality"
        save_as_gif: bool = "save_as_gif" in self.options
        save_last_frame: bool = "save_last_frame" in self.options
        no_autoplay: bool = "no_autoplay" in self.options
        hide_source: bool = "hide_source" in self.options
        ref_modules: list[str] = self.options.get("ref_modules", [])
        ref_classes: list[str] = self.options.get("ref_classes", [])
        ref_functions: list[str] = self.options.get("ref_functions", [])
        ref_methods: list[str] = self.options.get("ref_methods", [])

        assert not (save_as_gif and save_last_frame)

        source_code = "\n".join(self.content)
        identifier = Path(self.env.docname, clsname)
        build_dir = self.env.app.outdir.parent
        media_dir = Path(build_dir, "media", identifier)
        output_file = Path(self.env.app.outdir, "_manim", identifier).absolute()
        format_ = "png" if save_last_frame else "gif" if save_as_gif else "mp4"
        references = ref_modules + ref_classes + ref_functions + ref_methods

        def get_code() -> str:
            if self.content[0].startswith(">>> "):
                return "\n".join(
                    [
                        line[4:]
                        for line in self.content
                        if line.startswith((">>> ", "... "))
                    ]
                )
            return "\n".join(self.content)

        def render() -> str:
            try:
                with tempconfig(
                    {
                        "quality": quality,
                        "media_dir": media_dir,
                        "verbosity": "WARNING",
                        "progress_bar": "none",
                        "save_last_frame": save_last_frame,
                        "output_file": output_file,
                        "format": format_,
                    }
                ):
                    namespace: dict[str, Any] = {}
                    exec("from manim import *", namespace)
                    exec(code, namespace)
                    scene_cls: type[Scene] = namespace[clsname]
                    scene = scene_cls()
                    scene.render()

                    final_file_path = Path(scene.renderer.file_writer.final_file_path)
                    target = final_file_path.relative_to(
                        self.env.app.outdir.absolute()
                    ).as_posix()
                    page_uri = self.env.app.builder.get_target_uri(self.env.docname)
                    return relative_uri(page_uri, target)

            except Exception as e:
                msg = f"Failed to render scene: {clsname}"
                raise RuntimeError(msg) from e

        def insert() -> None:
            rst = _TEMPLATE.render(
                id=clsname.lower(),
                clsname=clsname,
                uri=uri,
                format=format_,
                autoplay=not no_autoplay,
                source_code=source_code,
                code=code,
                references=references,
                hide_source=hide_source,
            )
            self.state_machine.insert_input(rst.splitlines(), source=str(identifier))

        code = get_code()
        uri = render()
        insert()
        return []


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("manim", ManimDirective)

    app.add_js_file("manim-binder.min.js")
    app.add_js_file(
        None,
        body=f'window.initManimBinder({{branch: "v{manim_version}"}})',
    )
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


_TEMPLATE = jinja2.Template(r"""
{% if not hide_source %}
.. raw:: html

    <div id="{{ id }}" class="admonition admonition-manim-example">
    <p class="admonition-title">Example: {{ clsname }}<a class="headerlink" href="#{{ id }}">¶</a></p>

{% endif %}
.. raw:: html

{% if format == "mp4" %}
    <video class="manim-video" controls loop {{ " autoplay " if autoplay }} src="{{ uri }}" alt="{{ clsname }}"></video>
{% else %}
    <img src="{{ uri }}" alt="{{ clsname }}">
{% endif %}

{% if not hide_source %}
.. code-block:: python

    from manim import *

{{ source_code | indent(4, first=True) }}

.. raw:: html

    <pre data-manim-binder data-manim-classname="{{ clsname }}">
{{ code | e | indent(4, first=True) }}
    </pre>

{% endif %}

{% if not hide_source and references %}
References: {{ references | join(" ") }}

{% endif %}

{% if not hide_source %}
.. raw:: html

    </div>

{% endif %}
""")
