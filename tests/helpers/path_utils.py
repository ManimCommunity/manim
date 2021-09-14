from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_svg_resource(filename):
    return str(
        get_project_root() / "tests/test_graphical_units/img_svg_resources" / filename,
    )
