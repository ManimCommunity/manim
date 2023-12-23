from __future__ import annotations

from docutils import nodes
from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx

from manim.utils.docbuild.module_parsing import get_typing_docs

__all__ = ["TypingModuleDocumenter"]


def setup(app: Sphinx) -> None:
    app.add_directive("autotypingmodule", TypingModuleDocumenter)


class TypingModuleDocumenter(Directive):
    objtype = "autotypingmodule"
    required_arguments = 0
    has_content = True

    def run(self) -> list[nodes.Element]:
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
                result = ViewList()
                alias_section = nodes.topic(ids=[alias_name.lower().replace(" ", "_")])
                category_alias_container += alias_section
                alias_section += nodes.title(text=alias_name)

                alias_section += nodes.literal_block(text=alias_dict["definition"])

                doc = nodes.paragraph() # one paragraph for both
                # add | to keep on different lines
                # TODO: Figure out where to log sphinx errors (/tmp/sphinx-errs not availiable for non-linux)
                # result.append(f'| ``{alias_dict["definition"]}``', "/tmp/sphinx-errs.log", 10)
                if "doc" in alias_dict:
                    result.append("| %s" % alias_dict["doc"], "/tmp/sphinx-errs.log", 10)

                # https://www.sphinx-doc.org/en/master/extdev/markupapi.html#parsing-directive-content-as-rest
                self.state.nested_parse(result, 0, doc)
                alias_section += doc
            content += category_section

        return [content]
