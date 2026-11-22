"""Manim's modules import only from the layers below them.

A table of rules says which modules may not import which: lower layers never import
higher ones. Imports inside functions count too (they still couple the modules); only
imports for type checking (under ``if TYPE_CHECKING:``) are left out. The test lists
each import that breaks a rule, with the file and line.

Rules are added when they become true, never with a temporary exception. The known
exceptions are listed below with their reason, and are not to grow.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import manim

PACKAGE = Path(manim.__file__).parent

# (importer prefix, forbidden prefix, why)
RULES: list[tuple[str, str, str]] = [
    (
        "manim.camera",
        "manim.renderer._",
        "cameras describe views; private renderer modules are renderer internals",
    ),
    ("manim.camera", "manim.renderer.cairo", "cameras belong to no renderer"),
    ("manim.camera", "manim.renderer.opengl", "cameras belong to no renderer"),
    (
        "manim.renderer.cairo",
        "manim.renderer.opengl",
        "renderer packages do not import each other",
    ),
    (
        "manim.renderer.opengl",
        "manim.renderer.cairo",
        "renderer packages do not import each other",
    ),
    *(
        (
            f"manim.renderer.{low}",
            forbidden,
            "snapshot types and the helpers shared by renderers are plain data and"
            " functions: they know no mobject, camera or scene",
        )
        for low in ("snapshot", "protocol", "_lighting", "_mesh", "_paint")
        for forbidden in (
            "manim.mobject",
            "manim.animation",
            "manim.camera",
            "manim.scene",
            "manim.manager",
            "manim._run",
            "manim.renderer.extract",
            "manim.renderer.cairo",
            "manim.renderer.opengl",
        )
    ),
]

# (importer, imported) pairs that break a rule on purpose, with the reason.
EXCEPTIONS: dict[tuple[str, str], str] = {}


def _modules() -> dict[str, Path]:
    modules = {}
    for path in PACKAGE.rglob("*.py"):
        parts = ["manim", *path.relative_to(PACKAGE).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


def _imports(module: str, path: Path) -> list[tuple[str, int]]:
    """The manim modules ``module`` imports, with the line of each import, at module
    level and inside functions; imports for type checking are left out.
    """
    found: list[tuple[str, int]] = []

    def resolve(node: ast.Import | ast.ImportFrom) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        base = node.module or ""
        if node.level:
            package = module.split(".")
            if path.name != "__init__.py":
                package.pop()
            package = package[: len(package) - node.level + 1]
            base = ".".join([*package, *([node.module] if node.module else [])])
        # ``from package import submodule`` imports the submodule.
        return [base, *(f"{base}.{alias.name}" for alias in node.names)]

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.extend(
                    (name, child.lineno)
                    for name in resolve(child)
                    if name == "manim" or name.startswith("manim.")
                )
            else:
                visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def _within(name: str, prefix: str) -> bool:
    """Whether module ``name`` is ``prefix`` or inside it. A prefix ending in ``_``
    or ``.`` matches modules whose name starts with it (``manim.renderer._``).
    """
    if prefix.endswith(("_", ".")):
        return name.startswith(prefix)
    return name == prefix or name.startswith(prefix + ".")


def _violations(
    rules: list[tuple[str, str, str]], exceptions: dict[tuple[str, str], str]
) -> list[str]:
    modules = _modules()
    problems = []
    for module, path in sorted(modules.items()):
        applying = [rule for rule in rules if _within(module, rule[0])]
        if not applying:
            continue
        reported = set()
        for name, line in _imports(module, path):
            for _, forbidden, why in applying:
                if (
                    _within(name, forbidden)
                    and (module, name) not in exceptions
                    and line not in reported
                ):
                    reported.add(line)
                    relative = path.relative_to(PACKAGE.parent)
                    problems.append(
                        f"{relative}:{line}: {module} imports {name} ({why})"
                    )
    return problems


def test_modules_import_only_from_lower_layers():
    problems = _violations(RULES, EXCEPTIONS)
    assert not problems, "Imports across layers:\n" + "\n".join(problems)


def test_exceptions_are_still_needed():
    unused = [
        pair
        for pair in EXCEPTIONS
        if not _violations(RULES, {k: v for k, v in EXCEPTIONS.items() if k != pair})
    ]
    assert not unused, f"Exceptions no import needs any more: {unused}"


@pytest.mark.parametrize(
    ("importer", "imported", "rule"),
    [
        ("manim.camera", "manim.renderer._lighting", RULES[0]),
        ("manim.renderer.opengl.drawing", "manim.renderer.cairo.mesh_faces", RULES[4]),
        ("manim.renderer.snapshot", "manim.mobject.mobject", RULES[5]),
    ],
)
def test_a_rule_finds_imports_that_break_it(importer, imported, rule, monkeypatch):
    """The rules match what they should: an import added to a module is reported."""
    real = _imports

    def with_extra(module: str, path: Path) -> list[tuple[str, int]]:
        found = real(module, path)
        return [*found, (imported, 0)] if module == importer else found

    monkeypatch.setattr(f"{__name__}._imports", with_extra)
    assert any(imported in problem for problem in _violations([rule], {}))
