from __future__ import annotations

import inspect

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx

from manim import ManimColor

__all__ = ["TypingModuleDocumenter"]


def setup(app: Sphinx) -> None:
    app.add_directive("autotypingmodule", TypingModuleDocumenter)


class TypingModuleDocumenter(Directive):
    objtype = "autotypingmodule"
    required_arguments = 1
    has_content = True

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

    def run(self) -> List[nodes.Element]:
        module_name = self.arguments[0]
        try:
            import importlib

            module = importlib.import_module(module_name)
        except ImportError:
            return [
                nodes.error(
                    None,
                    nodes.paragraph(text="Failed to import module '%s'" % module_name),
                )
            ]

        content = nodes.container()
        for category_name, category in module.manim_type_aliases.items():
            category_section = nodes.section(
                ids=[category_name.lower().replace(" ", "_")]
            )
            category_section += nodes.title(text=category_name)
            category_alias_container = nodes.container()
            category_section += category_alias_container

            for alias_name in category:
                alias = getattr(module, alias_name)
                alias_section = nodes.topic(ids=[alias_name.lower().replace(" ", "_")])
                category_alias_container += alias_section
                alias_section += nodes.title(text=alias_name)
                alias_section += nodes.paragraph(text=str(alias))
            content += category_section

        return [content]
