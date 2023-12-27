from __future__ import annotations

from typing import TYPE_CHECKING

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList

from manim.utils.docbuild.module_parsing import parse_module_attributes

if TYPE_CHECKING:
    from sphinx.application import Sphinx

__all__ = ["AliasAttrDocumenter"]


ALIAS_DOCS_DICT, DATA_DICT = parse_module_attributes()
ALIAS_LIST = [
    alias_name
    for module_dict in ALIAS_DOCS_DICT.values()
    for category_dict in module_dict.values()
    for alias_name in category_dict.keys()
]


def setup(app: Sphinx) -> None:
    app.add_directive("autoaliasattr", AliasAttrDocumenter)


class AliasAttrDocumenter(Directive):
    objtype = "autotype"
    required_arguments = 1
    has_content = True

    def run(self) -> list[nodes.Element]:
        module_name = self.arguments[0]
        module_alias_dict = ALIAS_DOCS_DICT.get(module_name[6:], None)
        module_attrs_list = DATA_DICT.get(module_name[6:], None)

        content = nodes.container()

        # Add "Type Aliases" section
        if module_alias_dict is not None:
            module_alias_section = nodes.section(ids=[f"{module_name}.alias"])
            content += module_alias_section

            module_alias_section += nodes.rubric(text="Type Aliases")

            for category_name, category_dict in module_alias_dict.items():
                category_section = nodes.section(
                    ids=[category_name.lower().replace(" ", "_")]
                )
                module_alias_section += category_section
                if category_name:
                    category_section += nodes.title(text=category_name)

                category_alias_container = nodes.container()
                category_section += category_alias_container

                for alias_name, alias_subdict in category_dict.items():
                    unparsed = ViewList(
                        [
                            f".. class:: {alias_name}",
                            "",
                            "    .. code-block::",
                            "",
                            f"        {alias_subdict['definition']}",
                            "",
                        ]
                    )

                    if "doc" in alias_subdict:
                        alias_doc = alias_subdict["doc"]
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
                    alias_container = nodes.container()
                    self.state.nested_parse(unparsed, 0, alias_container)
                    category_alias_container += alias_container

        if module_attrs_list is not None:
            module_attrs_section = nodes.section(ids=[f"{module_name}.data"])
            content += module_attrs_section

            module_attrs_section += nodes.rubric(text="Module Attributes")
            unparsed = ViewList(
                [
                    ".. autosummary::",
                    *(f"    {attr}" for attr in module_attrs_list),
                ]
            )
            data_container = nodes.container()
            self.state.nested_parse(unparsed, 0, data_container)
            module_attrs_section += data_container

        return [content]
