from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from docutils.nodes import Node
from sphinx import directives
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from docutils.parsers.rst import directives


def setup(app: Sphinx) -> SetupMetadata:
    print("Adding manim directive")
    app.add_directive("manim", ManimDirective)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


class ManimDirective(SphinxDirective):
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

    def run(self) -> Sequence[Node]:
        return self.parse_content_to_nodes()


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


class SetupMetadata(TypedDict):
    parallel_read_safe: bool
    parallel_write_safe: bool
