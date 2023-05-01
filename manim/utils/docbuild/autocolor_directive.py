from __future__ import annotations

from manim import ManimColor

from typing import Any, Iterator, List
from sphinx.application import Sphinx
from sphinx.ext.autodoc import ModuleDocumenter, bool_option
from docutils.statemachine import StringList

def setup(app: Sphinx) -> None:
    app.setup_extension('sphinx.ext.autodoc')  # Require autodoc extension
    app.add_autodocumenter(ManimColorModuleDocumenter)

class ManimColorModuleDocumenter(ModuleDocumenter):
    objtype = 'manimcolormodule'
    directivetype = ModuleDocumenter.objtype
    priority = 10 + ModuleDocumenter.priority
    option_spec = dict(ModuleDocumenter.option_spec)

    def process_doc(self, docstrings: List[List[str]]) -> Iterator[str]:
        return []

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

    def add_content(self,
                    more_content: StringList | None,
                    no_docstring: bool = False,
                    ) -> None:

        super().add_content(more_content, no_docstring)

        source_name = self.get_sourcename()
        self.add_line('', source_name)

        self.add_line('.. list-table::', source_name)
        self.add_line('   :header-rows: 1', source_name)
        self.add_line('', source_name)
        self.add_line('   * - Color name', source_name)
        self.add_line('     - Color', source_name)
        self.add_line('     - RGB hex code', source_name)


        for member_name in self.get_module_members():
            member = getattr(self.module, member_name)
            if isinstance(member, ManimColor):
                self.add_line(f'   * - ``{member_name}``', source_name)
                self.add_line(f'     - .. raw:: html', source_name)
                self.add_line(f'', source_name)
                self.add_line(
                    f'         <div style="background-color:{member.to_hex()};width:50px;height:2ex;"></div>',
                    source_name
                )
                self.add_line('', source_name)
                self.add_line(f'     - {member.to_hex()}', source_name)
        
        self.add_line('', source_name)