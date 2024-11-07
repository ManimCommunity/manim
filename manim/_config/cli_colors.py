from __future__ import annotations

import configparser
from typing import Any

from cloup import Context, HelpFormatter, HelpTheme, Style

__all__ = ["parse_cli_ctx"]


def parse_cli_ctx(parser: configparser.SectionProxy) -> dict[str, Any]:
    formatter_settings: dict[str, str | int] = {
        "indent_increment": int(parser["indent_increment"]),
        "width": int(parser["width"]),
        "col1_max_width": int(parser["col1_max_width"]),
        "col2_min_width": int(parser["col2_min_width"]),
        "col_spacing": int(parser["col_spacing"]),
        "row_sep": parser["row_sep"] if parser["row_sep"] else None,
    }
    theme_settings = {}
    theme_keys = {
        "command_help",
        "invoked_command",
        "heading",
        "constraint",
        "section_help",
        "col1",
        "col2",
        "epilog",
    }
    for k, v in parser.items():
        if k in theme_keys and v:
            theme_settings.update({k: Style(v)})

    formatter = {}
    theme = parser["theme"] if parser["theme"] else None
    if theme is None:
        formatter = HelpFormatter.settings(
            theme=HelpTheme(**theme_settings),
            **formatter_settings,  # type: ignore[arg-type]
        )
    elif theme.lower() == "dark":
        formatter = HelpFormatter.settings(
            theme=HelpTheme.dark().with_(**theme_settings),
            **formatter_settings,  # type: ignore[arg-type]
        )
    elif theme.lower() == "light":
        formatter = HelpFormatter.settings(
            theme=HelpTheme.light().with_(**theme_settings),
            **formatter_settings,  # type: ignore[arg-type]
        )

    return Context.settings(
        align_option_groups=parser["align_option_groups"].lower() == "true",
        align_sections=parser["align_sections"].lower() == "true",
        show_constraints=True,
        formatter_settings=formatter,
    )
