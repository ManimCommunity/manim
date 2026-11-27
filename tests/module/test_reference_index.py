"""Every public module has a page in the reference documentation.

The reference pages are generated from the autosummary lists of
``docs/source/reference_index/*.rst`` and of the docstrings of packages that list
their modules; the page of a package lists its submodules too (see
``docs/source/_templates/autosummary/module.rst``). A module whose name has no part
starting with an underscore is public and must be on one of these pages. Modules that had no page before this test are listed
below; the list is not to grow.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import manim

PACKAGE = Path(manim.__file__).parent
REFERENCE_INDEX = PACKAGE.parent / "docs" / "source" / "reference_index"

# Without a page when this test was added; remove entries as pages are added.
UNLISTED = {
    "manim",
    "manim.animation",
    "manim.mobject",
    "manim.plugins",
    "manim.plugins.plugins_flags",
    "manim.renderer",
    "manim.renderer.cairo",
    "manim.renderer.cairo.rendering",
    "manim.renderer.cairo.target",
    "manim.renderer.opengl",
    "manim.renderer.opengl.shader",
    "manim.renderer.opengl.window",
    "manim.renderer.opengl.window_settings",
    "manim.renderer.protocol",
    "manim.scene",
    "manim.scene.audio_mixer",
    "manim.scene.video_segment_encoder",
    "manim.utils",
    "manim.utils.caching",
    "manim.utils.exceptions",
    "manim.utils.family",
    "manim.utils.family_ops",
    "manim.utils.file_ops",
    "manim.utils.importing",
    "manim.utils.module_ops",
    "manim.utils.parameter_parsing",
    "manim.utils.polylabel",
    "manim.utils.qhull",
    "manim.utils.typst_file_writing",
    "manim.utils.unit",
}

_ENTRY = re.compile(r"^\s+~?([\w.]+)\s*$", re.MULTILINE)


def _public_modules() -> set[str]:
    modules = set()
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts[:2] == ("manim", "cli") or any(p.startswith("_") for p in parts[1:]):
            continue
        modules.add(".".join(parts))
    return modules


def _documented_modules() -> set[str]:
    documented = set()
    for index in REFERENCE_INDEX.glob("*.rst"):
        for entry in _ENTRY.findall(index.read_text(encoding="utf-8")):
            documented.add(entry if entry.startswith("manim") else f"manim.{entry}")
    # Packages that list their modules in their docstring.
    for init in PACKAGE.rglob("__init__.py"):
        docstring = ast.get_docstring(ast.parse(init.read_text(encoding="utf-8")))
        if not docstring or "autosummary" not in docstring:
            continue
        package = ".".join(init.parent.relative_to(PACKAGE.parent).parts)
        for entry in _ENTRY.findall(docstring):
            documented.add(entry if entry.startswith("manim") else f"{package}.{entry}")
    return documented


def _covered(module: str, documented: set[str]) -> bool:
    """Whether the module, or a package it is in, is documented."""
    parts = module.split(".")
    return any(".".join(parts[:i]) in documented for i in range(2, len(parts) + 1))


def test_every_public_module_has_a_reference_page():
    documented = _documented_modules()
    missing = {
        module
        for module in _public_modules() - UNLISTED
        if not _covered(module, documented)
    }
    assert not missing, (
        "Add these modules to a list in docs/source/reference_index/ (or make them "
        f"private with a leading underscore): {sorted(missing)}"
    )


def test_unlisted_modules_are_still_unlisted():
    stale = UNLISTED & (_documented_modules() | (UNLISTED - _public_modules()))
    assert not stale, f"Remove these from UNLISTED: {sorted(stale)}"
