r"""A Sphinx directive that renders Manim scenes into the documentation.

Usage::

    .. manim:: MyScene
        :save_last_frame:

        class MyScene(Scene):
            def construct(self):
                ...

The argument is the name of the scene class to render. The content is either
plain code or a doctest, whose ``>>>`` and ``...`` lines are run.

Options
-------

``hide_source``
    Show only the rendered output, not the code.
``header``
    The title of the example, ``Example: MyScene`` when omitted. ``None`` or
    ``False`` leave it out, e.g. when the page already shows a title of its
    own.
``no_autoplay``
    Do not start the video automatically.
``quality``
    One of ``low``, ``medium``, ``high``, ``fourk``, ``production`` or
    ``example`` (the default), as in :data:`~.constants.QUALITIES`.
``save_as_gif``
    Render a GIF instead of a video.
``save_last_frame``
    Render only the last frame, as an image.
``ref_modules``, ``ref_classes``, ``ref_functions``, ``ref_methods``
    Space-separated names listed as references below the code.

Configuration
-------------

The directive reads two values from ``conf.py``; both can be overridden on
the command line with ``-D``:

``manim_config``
    Manim config applied to every example, as with :func:`.tempconfig`.
    The options of an example are applied on top of it. Single entries can
    be overridden, e.g. ``-D manim_config.quality=low_quality``.
``manim_render``
    Whether to render examples that are not cached yet (default ``True``).
    With ``-D manim_render=0``, such examples are shown as their code only;
    this is how doctests and translation templates are built.

Rendered examples are cached in ``_manim`` in the build's output directory,
named after a hash of the scene, its code and its config, so an example is
only rendered again when one of these changes. A build can thus be split into
two passes: one that renders, without writing any pages, and one that writes
the pages from the cache::

    sphinx-build -b dummy source build/dummy -d build/doctrees
    sphinx-build -b html source build/html -d build/doctrees

Both passes need the same output directory for ``_manim``, which is where the
HTML pages look for the rendered files; pass ``-D manim_cache=<dir>`` to the
first one if it writes elsewhere.

At the end of every build, a summary of the slowest examples is printed, and
the render time of every example is written to ``manim_render_times.csv`` in
the output directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
from docutils import nodes
from docutils.parsers.rst.directives import choice, flag
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri
from sphinx.util.typing import ExtensionMetadata

from manim import __version__ as manim_version
from manim._config import tempconfig
from manim.constants import QUALITIES
from manim.scene.scene import Scene

__all__ = ["ManimDirective"]

logger = logging.getLogger(__name__)

_SUFFIXES = {"png": ".png", "gif": ".gif", "mp4": ".mp4"}


@dataclass
class _Render:
    """How long an example took to render; ``None`` when it was cached."""

    docname: str
    clsname: str
    seconds: float | None


def _references(role: str, names: str | None) -> list[str]:
    return [f":{role}:`~.{name}`" for name in (names or "").split()]


def _renders(env: BuildEnvironment) -> list[_Render]:
    """The examples of this build, kept with the environment (see setup)."""
    renders: list[_Render] = env.domaindata.setdefault("manim", {}).setdefault(
        "renders", []
    )
    return renders


class ManimDirective(SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        "hide_source": flag,
        "header": lambda arg: None if arg == "None" else arg,
        "no_autoplay": flag,
        "quality": lambda arg: choice(
            arg or "",
            [q.removesuffix("_quality") for q in QUALITIES],
        ),
        "save_as_gif": flag,
        "save_last_frame": flag,
        "ref_modules": lambda arg: _references("mod", arg),
        "ref_classes": lambda arg: _references("class", arg),
        "ref_functions": lambda arg: _references("func", arg),
        "ref_methods": lambda arg: _references("meth", arg),
    }

    def run(self) -> list[nodes.Element]:
        clsname: str = self.arguments[0].strip()
        save_as_gif = "save_as_gif" in self.options
        save_last_frame = "save_last_frame" in self.options
        if save_as_gif and save_last_frame:
            raise self.error("save_as_gif and save_last_frame exclude each other")
        format_ = "png" if save_last_frame else "gif" if save_as_gif else "mp4"
        references = [
            reference
            for option in ("ref_modules", "ref_classes", "ref_functions", "ref_methods")
            for reference in self.options.get(option, [])
        ]

        if self.content and self.content[0].startswith(">>> "):
            code = "\n".join(
                line[4:] for line in self.content if line.startswith((">>> ", "... "))
            )
        else:
            code = "\n".join(self.content)

        example_config = {
            **self.config.manim_config,
            "quality": self.options.get("quality", "example") + "_quality",
            "format": format_,
            "save_last_frame": save_last_frame,
        }
        output = self._cached_output(clsname, code, example_config)
        if output.exists():
            self._record(clsname, None)
        elif self.config.manim_render:
            self._render(clsname, code, example_config, output)

        uri = None
        if output.exists():
            # The cache is the ``_manim`` directory of the site, also when a
            # render-only pass writes it from elsewhere (``manim_cache``). The
            # link is relative to the page's HTML file, and does not depend on
            # the builder, as a later HTML build may reuse this page's doctree.
            uri = relative_uri(f"{self.env.docname}.html", f"_manim/{output.name}")

        rst = _TEMPLATE.render(
            id=clsname.lower(),
            clsname=clsname,
            uri=uri,
            format=format_,
            autoplay="no_autoplay" not in self.options,
            source_code="\n".join(self.content),
            code=code,
            references=references,
            hide_source="hide_source" in self.options and uri is not None,
            header=self.options.get("header", f"Example: {clsname}"),
        )
        self.state_machine.insert_input(
            rst.splitlines(),
            source=self.state_machine.get_source_and_line(self.lineno)[0] or "",
        )
        return []

    def _cached_output(
        self, clsname: str, code: str, example_config: dict[str, Any]
    ) -> Path:
        """Return the path of the example's output, named after what it shows."""
        key = json.dumps(
            [manim_version, clsname, code, example_config], sort_keys=True, default=str
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        suffix = _SUFFIXES[example_config["format"]]
        return _cache_dir(self.env) / f"{clsname}-{digest}{suffix}"

    def _render(
        self, clsname: str, code: str, example_config: dict[str, Any], output: Path
    ) -> None:
        # Each example renders in a media directory of its own, next to the
        # build's output directory, and its output is moved into the cache
        # when done: parallel processes never share files (the same example
        # can be on pages read by different processes), and an interrupted
        # render leaves nothing in the cache.
        media_dir = Path(
            self.env.app.outdir, "..", "manim_media", self.env.docname, output.stem
        ).resolve()
        partial = media_dir / output.stem
        config = {
            "media_dir": media_dir,
            "assets_dir": Path(self.env.app.srcdir) / "_static",
            "progress_bar": "none",
            "verbosity": "WARNING",
            **example_config,
            "output_file": partial,
        }
        start = time.perf_counter()
        try:
            with tempconfig(config):
                namespace: dict[str, Any] = {}
                exec("from manim import *", namespace)
                exec(code, namespace)
                scene: Scene = namespace[clsname]()
                scene.render()
        except Exception as error:
            raise self.error(
                f"Failed to render {clsname}: {type(error).__name__}: {error}"
            ) from error
        seconds = time.perf_counter() - start
        output.parent.mkdir(parents=True, exist_ok=True)
        # The cache can be on another file system (see ``manim_cache``).
        temporary = output.with_name(f".{output.name}")
        shutil.move(partial.with_suffix(output.suffix), temporary)
        temporary.replace(output)
        self._record(clsname, seconds)

    def _record(self, clsname: str, seconds: float | None) -> None:
        _renders(self.env).append(_Render(self.env.docname, clsname, seconds))


def _cache_dir(env: BuildEnvironment) -> Path:
    cache = env.config.manim_cache
    return Path(cache or Path(env.app.outdir, "_manim")).absolute()


def _merge_renders(
    app: Sphinx, env: BuildEnvironment, docnames: set[str], other: BuildEnvironment
) -> None:
    """Collect the examples of the documents a parallel reader process read."""
    _renders(env).extend(
        render for render in _renders(other) if render.docname in docnames
    )


def _reset_renders(app: Sphinx, env: BuildEnvironment, docnames: list[str]) -> None:
    _renders(env).clear()


def _report_renders(app: Sphinx, exception: Exception | None) -> None:
    examples = _renders(app.env)
    rendered = sorted(
        (example for example in examples if example.seconds is not None),
        key=lambda example: example.seconds or 0.0,
        reverse=True,
    )
    if exception is not None or not examples:
        return
    report = Path(app.outdir, "manim_render_times.csv")
    with report.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["document", "scene", "seconds"])
        for example in rendered:
            writer.writerow(
                [example.docname, example.clsname, f"{example.seconds:.3f}"]
            )
    total = sum(example.seconds or 0.0 for example in rendered)
    lines = [
        f"Manim examples: {len(rendered)} rendered in {total:.1f}s (summed over "
        f"processes), {len(examples) - len(rendered)} already in the cache."
    ]
    if rendered:
        lines.append("Slowest:")
        lines += [
            f"  {example.seconds:7.2f}s  {example.clsname} ({example.docname})"
            for example in rendered[:10]
        ]
        lines.append(f"All render times: {report}")
    logger.info("\n".join(lines))


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_config_value("manim_config", {}, "env", dict)
    app.add_config_value("manim_render", True, "env", bool)
    app.add_config_value("manim_cache", "", "", str)
    app.add_directive("manim", ManimDirective)
    app.connect("env-before-read-docs", _reset_renders)
    app.connect("env-merge-info", _merge_renders)
    app.connect("build-finished", _report_renders)

    app.add_js_file("manim-binder.min.js")
    app.add_js_file(
        None,
        body=f'window.initManimBinder({{branch: "v{manim_version}"}})',
    )
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


# Values are HTML-escaped unless marked ``safe``: the reST parts of the
# template (code blocks, references) must be inserted as they are.
_TEMPLATE = jinja2.Template(
    r"""
{% if not hide_source %}
.. raw:: html

    <div id="{{ id }}" class="admonition admonition-manim-example">
{% if header is not none %}
    <p class="admonition-title">{{ header }}<a class="headerlink" href="#{{ id }}">¶</a></p>
{% endif %}

{% endif %}
{% if uri is none %}
.. raw:: html

    <p class="manim-example-not-rendered">This example was not rendered in this build.</p>

{% elif format == "mp4" %}
.. raw:: html

    <video class="manim-video" controls loop {{ " autoplay " if autoplay }} src="{{ uri }}" alt="{{ clsname }}"></video>

{% else %}
.. raw:: html

    <img src="{{ uri }}" alt="{{ clsname }}">

{% endif %}
{% if not hide_source %}
.. code-block:: python

    from manim import *

{{ source_code | indent(4, first=True) | safe }}

.. raw:: html

    <pre data-manim-binder data-manim-classname="{{ clsname }}">
{{ code | indent(4, first=True) }}
    </pre>

{% endif %}
{% if not hide_source and references %}
References: {{ references | join(" ") | safe }}

{% endif %}
{% if not hide_source %}
.. raw:: html

    </div>

{% endif %}
""",
    autoescape=True,
)
