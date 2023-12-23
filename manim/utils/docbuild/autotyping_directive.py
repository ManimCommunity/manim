from __future__ import annotations

import inspect

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx

from manim import ManimColor
from manim.utils.docbuild.module_parsing import get_typing_docs

__all__ = ["TypingModuleDocumenter"]


def setup(app: Sphinx) -> None:
    app.add_directive("autotypingmodule", TypingModuleDocumenter)


class TypingModuleDocumenter(Directive):
    objtype = "autotypingmodule"
    required_arguments = 0
    has_content = True

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

    def run(self) -> List[nodes.Element]:
        content = nodes.container()

        typing_docs_dict = get_typing_docs()
        for category_name, category_dict in typing_docs_dict.items():
            category_section = nodes.section(
                ids=[category_name.lower().replace(" ", "_")]
            )
            category_section += nodes.title(text=category_name)
            category_alias_container = nodes.container()
            category_section += category_alias_container

            for alias_name, alias_dict in category_dict.items():
                alias_section = nodes.topic(ids=[alias_name.lower().replace(" ", "_")])
                category_alias_container += alias_section
                alias_section += nodes.title(text=alias_name)
                alias_section += nodes.paragraph(text=alias_dict["definition"])
                if "doc" in alias_dict:
                    alias_section += nodes.paragraph(text=alias_dict["doc"])
            content += category_section

        return [content]
