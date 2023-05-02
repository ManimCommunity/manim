from __future__ import annotations

import inspect

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx

from manim import ManimColor


def setup(app: Sphinx) -> None:
    app.add_directive("automanimcolormodule", ManimColorModuleDocumenter)


class ManimColorModuleDocumenter(Directive):
    objtype = "automanimcolormodule"
    required_arguments = 1
    has_content = True

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

    def run(
        self,
    ) -> None:
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

        # Number of Colors displayed in one row
        num_color_cols = 2
        table = nodes.table()

        tgroup = nodes.tgroup(cols=num_color_cols * 2)
        table += tgroup
        for _ in range(num_color_cols * 2):
            tgroup += nodes.colspec(colwidth=1)

        # Create header rows for the table
        thead = nodes.thead()
        row = nodes.row()
        for _ in range(num_color_cols):
            col1 = nodes.paragraph(text="Color Name")
            col2 = nodes.paragraph(text="RGB Hex Code")
            row += nodes.entry("", col1)
            row += nodes.entry("", col2)
        thead += row
        tgroup += thead

        color_elements = []
        for member_name, member_obj in inspect.getmembers(module):
            if isinstance(member_obj, ManimColor):
                hex_code = member_obj.to_hex()[1:]
                r, g, b = tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))
                luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255

                # Choose the font color based on the background luminance
                if luminance > 0.5:
                    font_color = "black"
                else:
                    font_color = "white"

                color_elements.append((member_name, member_obj.to_hex(), font_color))

        tbody = nodes.tbody()

        def get_color_element():
            yield from color_elements

        for base_i in range(0, len(color_elements), num_color_cols):
            row = nodes.row()
            for member_name, hex_code, font_color in color_elements[
                base_i : base_i + num_color_cols
            ]:
                col1 = nodes.literal()
                col1 += nodes.inline(text=member_name)
                col2 = nodes.paragraph()
                col2 += nodes.raw(
                    "",
                    f'<div style="width:100px;height:2rem;background-color:{hex_code};text-align:center;line-height:37px;border-radius:8px;"><p style="color:{font_color}; font-weight:bold; font-family: mono;">{hex_code}</p></div>',
                    format="html",
                )
                row += nodes.entry("", col1)
                row += nodes.entry("", col2)
            tbody += row
        tgroup += tbody

        return [table]