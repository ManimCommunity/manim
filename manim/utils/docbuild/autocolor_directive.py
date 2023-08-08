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
        table = nodes.table(align="center")

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
                r, g, b = member_obj.to_rgb()
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

                # Choose the font color based on the background luminance
                if luminance > 0.5:
                    font_color = "black"
                else:
                    font_color = "white"

                color_elements.append((member_name, member_obj.to_hex(), font_color))

        tbody = nodes.tbody()

        for base_i in range(0, len(color_elements), num_color_cols):
            row = nodes.row()
            for member_name, hex_code, font_color in color_elements[
                base_i : base_i + num_color_cols
            ]:
                col1 = nodes.literal(text=member_name)
                col2 = nodes.raw(
                    "",
                    f'<div style="background-color:{hex_code};padding: 0.25rem 0;border-radius:8px;margin: 0.5rem 0.2rem"><code style="color:{font_color};">{hex_code}</code></div>',
                    format="html",
                )
                row += nodes.entry("", col1)
                row += nodes.entry("", col2)
            tbody += row
        tgroup += tbody

        return [table]
