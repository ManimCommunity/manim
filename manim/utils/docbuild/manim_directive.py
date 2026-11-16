from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TypedDict

import sphinx.ext.autosummary
from docutils.nodes import Node
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


def setup(app: Sphinx) -> SetupMetadata:
    app.add_directive("manim", ManimDirective)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


if hasattr(sphinx.ext.autosummary, "FakeDirective"):
    sphinx.ext.autosummary.FakeDirective = functools.cache(
        sphinx.ext.autosummary.FakeDirective
    )


class ManimDirective(SphinxDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True

    def run(self) -> Sequence[Node]:
        return self.parse_text_to_nodes()  # type: ignore[no-any-return]





class SetupMetadata(TypedDict):
    parallel_read_safe: bool
    parallel_write_safe: bool
