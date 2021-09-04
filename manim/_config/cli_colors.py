import configparser

from cloup import Context, HelpFormatter, HelpTheme, Style

from manim.constants import HELP_OPTIONS


def parse_cli_ctx(parser: configparser.ConfigParser):
    formatter_settings = {
        "indent_increment": int(parser["indent_increment"]),
        "width": int(parser["width"]),
        "col1_max_width": int(parser["col1_max_width"]),
        "col2_min_width": int(parser["col2_min_width"]),
        "col_spacing": int(parser["col_spacing"]),
        "row_sep": parser["row_sep"] if parser["row_sep"] else None,
    }
    theme_settings = {
        "invoked_command": Style(parser["invoked_command"])
        if parser["invoked_command"]
        else None,
        "command_help": Style(parser["command_help"])
        if parser["command_help"]
        else None,
        "heading": Style(parser["heading"]) if parser["heading"] else None,
        "constraint": Style(parser["constraint"]) if parser["constraint"] else None,
        "section_help": Style(parser["section_help"])
        if parser["section_help"]
        else None,
        "col1": Style(parser["col1"]) if parser["col1"] else None,
        "col2": Style(parser["col2"]) if parser["col2"] else None,
    }
    formatter = {}
    theme = parser["theme"]
    if theme is None:
        formatter = HelpFormatter().settings(
            theme=HelpTheme(**theme_settings), **formatter_settings
        )
    elif theme.lower() == "dark":
        formatter = HelpFormatter().settings(
            theme=HelpTheme.dark().with_(**theme_settings), **formatter_settings
        )
    elif theme.lower() == "light":
        formatter = HelpFormatter().settings(
            theme=HelpTheme.light().with_(**theme_settings), **formatter_settings
        )

    ctx_settings = Context.settings(
        align_option_groups=parser["align_option_groups"].lower() == "true",
        align_sections=parser["align_sections"].lower() == "true",
        show_constraints=True,
        help_option_names=HELP_OPTIONS,
        formatter_settings=formatter,
    )
    return ctx_settings
