from __future__ import annotations

import re
from typing import TYPE_CHECKING

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList

from manim.utils.docbuild.module_parsing import get_manim_typing_docs

if TYPE_CHECKING:
    from sphinx.application import Sphinx

__all__ = ["TypingModuleDocumenter"]


MANIM_TYPING_DOCS = get_manim_typing_docs()
ALIAS_LIST = [
    alias_name
    for category_dict in MANIM_TYPING_DOCS.values()
    for alias_name in category_dict.keys()
]


def setup(app: Sphinx) -> None:
    app.add_directive("autotypingmodule", TypingModuleDocumenter)


class TypingModuleDocumenter(Directive):
    objtype = "autotypingmodule"
    required_arguments = 0
    has_content = True

    def run(self) -> list[nodes.Element]:
        content = nodes.container()
        for category_name, category_dict in MANIM_TYPING_DOCS.items():
            category_section = nodes.section(
                ids=[category_name.lower().replace(" ", "_")]
            )
            category_section += nodes.title(text=category_name)
            category_alias_container = nodes.container()
            category_section += category_alias_container

            for alias_name, alias_dict in category_dict.items():
                unparsed = ViewList(
                    [
                        f".. class:: {alias_name}",
                        "",
                        "    .. code-block::",
                        "",
                        f"        {alias_dict['definition']}",
                        "",
                    ]
                )

                if "doc" in alias_dict:
                    alias_doc = alias_dict["doc"]
                    for A in ALIAS_LIST:
                        alias_doc = alias_doc.replace(f"`{A}`", f":class:`~.{A}`")
                    doc_lines = alias_doc.split("\n")
                    if (
                        len(doc_lines) >= 2
                        and doc_lines[0].startswith("``shape:")
                        and doc_lines[1].strip() != ""
                    ):
                        doc_lines.insert(1, "")
                    unparsed.extend(ViewList([f"    {line}" for line in doc_lines]))

                # https://www.sphinx-doc.org/en/master/extdev/markupapi.html#parsing-directive-content-as-rest
                alias_section = nodes.container()
                self.state.nested_parse(unparsed, 0, alias_section)
                category_alias_container += alias_section

            content += category_section

        return [content]
