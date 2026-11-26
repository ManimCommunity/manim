from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("sphinx")

from sphinx.application import Sphinx

import manim.utils.docbuild.manim_directive as manim_directive

_PAGE = """\
Page
====

.. manim:: Example
    :save_last_frame:
{options}
    class Example(Scene):
        def construct(self):
            self.add(Square())
"""


def _build(
    tmp_path: Path, options: str, monkeypatch, warnings: io.StringIO | None = None
) -> dict:
    """Build a one-page project and return the config Example rendered with."""
    seen: dict = {}

    def render(self, clsname, code, example_config, output):
        seen.update(example_config)

    monkeypatch.setattr(manim_directive.ManimDirective, "_render", render)
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "conf.py").write_text(
        'extensions = ["manim.utils.docbuild.manim_directive"]\n'
    )
    (source / "index.rst").write_text(_PAGE.format(options=options))
    app = Sphinx(
        source,
        source,
        tmp_path / "out",
        tmp_path / "doctrees",
        "dummy",
        status=None,
        warning=warnings,
        freshenv=True,
    )
    app.build()
    return seen


def test_examples_render_with_the_configured_renderer_by_default(tmp_path, monkeypatch):
    assert "renderer" not in _build(tmp_path, "", monkeypatch)


def test_an_example_can_choose_its_renderer(tmp_path, monkeypatch):
    config = _build(tmp_path, "    :renderer: opengl\n", monkeypatch)
    assert config["renderer"] == "opengl"


def test_the_renderer_is_part_of_the_cache_key(tmp_path, monkeypatch):
    paths = []

    def cached_output(self, clsname, code, example_config):
        path = original(self, clsname, code, example_config)
        paths.append(path.name)
        return path

    original = manim_directive.ManimDirective._cached_output
    monkeypatch.setattr(manim_directive.ManimDirective, "_cached_output", cached_output)
    _build(tmp_path / "a", "", monkeypatch)
    _build(tmp_path / "b", "    :renderer: opengl\n", monkeypatch)
    assert paths[0] != paths[1]


def test_unknown_renderers_are_rejected(tmp_path, monkeypatch):
    warnings = io.StringIO()
    config = _build(tmp_path, "    :renderer: povray\n", monkeypatch, warnings)
    assert "povray" in warnings.getvalue()
    assert not config
