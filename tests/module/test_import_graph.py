"""The modules of manim import each other without cycles.

A cycle makes the import order matter: whichever module of the cycle is
imported first sees the others only partly initialised. Imports for type
checking and imports inside functions are not part of the graph; they run
after all modules are loaded. An import that would close a cycle at module
level stays inside its function, marked ``# local import to avoid import
cycle``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import manim

PACKAGE = Path(manim.__file__).parent


def _modules() -> dict[str, Path]:
    modules = {}
    for path in PACKAGE.rglob("*.py"):
        parts = ["manim", *path.relative_to(PACKAGE).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


def _imported_names(module: str, path: Path, node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    assert isinstance(node, ast.ImportFrom)
    base = node.module or ""
    if node.level:
        package = module.split(".")
        if path.name != "__init__.py":
            package.pop()
        package = package[: len(package) - node.level + 1]
        base = ".".join([*package, *([node.module] if node.module else [])])
    # ``from package import submodule`` imports the submodule.
    return [f"{base}.{alias.name}" for alias in node.names] + [base]


def _module_level_imports(tree: ast.Module) -> list[ast.stmt]:
    found: list[ast.stmt] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.append(child)
            else:
                visit(child)

    visit(tree)
    return found


def _import_graph() -> dict[str, set[str]]:
    modules = _modules()
    graph: dict[str, set[str]] = {}
    for module, path in modules.items():
        targets = set()
        for node in _module_level_imports(ast.parse(path.read_text(encoding="utf-8"))):
            for name in _imported_names(module, path, node):
                # Importing a module first imports the packages that contain
                # it. Those that contain this module too are being imported
                # already; they count only when imported from directly.
                parts = name.split(".")
                for end in range(1, len(parts) + 1):
                    target = ".".join(parts[:end])
                    if target not in modules or target == module:
                        continue
                    if module.startswith(target + ".") and target != name:
                        continue
                    targets.add(target)
        graph[module] = targets
    return graph


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """The strongly connected components with more than one module (Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[list[str]] = []

    def visit(module: str) -> None:
        index[module] = low[module] = len(index)
        stack.append(module)
        on_stack.add(module)
        for target in sorted(graph[module]):
            if target not in index:
                visit(target)
                low[module] = min(low[module], low[target])
            elif target in on_stack:
                low[module] = min(low[module], index[target])
        if low[module] == index[module]:
            component = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == module:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))

    for module in sorted(graph):
        if module not in index:
            visit(module)
    return cycles


def test_modules_import_each_other_without_cycles():
    cycles = _cycles(_import_graph())
    assert not cycles, "Import cycles between manim modules:\n" + "\n".join(
        ", ".join(cycle) for cycle in cycles
    )
