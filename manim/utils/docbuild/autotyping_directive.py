from __future__ import annotations

from typing import TYPE_CHECKING

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList

from manim.utils.docbuild.module_parsing import get_manim_typing_docs

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment

__all__ = ["TypingModuleDocumenter", "resolve_type_aliases"]


MANIM_TYPING_DOCS = get_manim_typing_docs()
MANIM_TYPE_ALIASES = [
    alias_name
    for category_dict in MANIM_TYPING_DOCS.values()
    for alias_name in category_dict.keys()
]
# This is to prevent issues when replacing text
MANIM_TYPE_ALIASES.sort(key=lambda alias: len(alias), reverse=True)


def setup(app: Sphinx) -> None:
    app.connect("missing-reference", resolve_type_aliases)
    app.add_directive("autotypingmodule", TypingModuleDocumenter)


def resolve_type_aliases(
    app: Sphinx,
    env: BuildEnvironment,
    node: nodes.Element,
    contnode: nodes.Element,
) -> nodes.Element | None:
    """Resolve :class: references to type aliases as :attr: instead.
    From https://github.com/sphinx-doc/sphinx/issues/10785
    """
    if (
        node["refdomain"] == "py"
        and node["reftype"] == "class"
        and node["reftarget"] in MANIM_TYPE_ALIASES
    ):
        return app.env.get_domain("py").resolve_xref(
            env,
            node["refdoc"],
            app.builder,
            "attr",
            node["reftarget"],
            node,
            contnode,
        )


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
                alias_section = nodes.section(ids=[alias_name])
                category_alias_container += alias_section

                alias_section += nodes.title(text=alias_name)
                alias_paragraph = nodes.paragraph()
                alias_section += alias_paragraph

                alias_def = alias_dict["definition"]
                # for A in MANIM_TYPE_ALIASES:
                #     alias_def = alias_def.replace(A, f":ref:`{A}`")
                alias_paragraph += nodes.literal_block(text=alias_def)
                result = ViewList()
                if "doc" in alias_dict:
                    alias_doc = alias_dict["doc"]
                    # for A in MANIM_TYPE_ALIASES:
                    #     alias_doc = alias_doc.replace(A, f":ref:`{A}`")
                    # add | to keep on different lines
                    result.append("| %s" % alias_doc, "/tmp/sphinx-errs.log", 10)

                # https://www.sphinx-doc.org/en/master/extdev/markupapi.html#parsing-directive-content-as-rest
                self.state.nested_parse(result, 0, alias_paragraph)

            content += category_section

        return [content]
