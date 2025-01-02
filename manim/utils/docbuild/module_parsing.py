"""Read and parse all the Manim modules and extract documentation from them."""

from __future__ import annotations

import ast
import sys
from ast import Attribute, Name, Subscript
from pathlib import Path
from typing import Any

from typing_extensions import TypeAlias

__all__ = ["parse_module_attributes"]


AliasInfo: TypeAlias = dict[str, str]
"""Dictionary with a `definition` key containing the definition of
a :class:`TypeAlias` as a string, and optionally a `doc` key containing
the documentation for that alias, if it exists.
"""

AliasCategoryDict: TypeAlias = dict[str, AliasInfo]
"""Dictionary which holds an `AliasInfo` for every alias name in a same
category.
"""

ModuleLevelAliasDict: TypeAlias = dict[str, AliasCategoryDict]
"""Dictionary containing every :class:`TypeAlias` defined in a module,
classified by category in different `AliasCategoryDict` objects.
"""

ModuleTypeVarDict: TypeAlias = dict[str, str]
"""Dictionary containing every :class:`TypeVar` defined in a module."""


AliasDocsDict: TypeAlias = dict[str, ModuleLevelAliasDict]
"""Dictionary which, for every module in Manim, contains documentation
about their module-level attributes which are explicitly defined as
:class:`TypeAlias`, separating them from the rest of attributes.
"""

DataDict: TypeAlias = dict[str, list[str]]
"""Type for a dictionary which, for every module, contains a list with
the names of all their DOCUMENTED module-level attributes (identified
by Sphinx via the ``data`` role, hence the name) which are NOT
explicitly defined as :class:`TypeAlias`.
"""

TypeVarDict: TypeAlias = dict[str, ModuleTypeVarDict]
"""A dictionary mapping module names to dictionaries of :class:`TypeVar` objects."""

ALIAS_DOCS_DICT: AliasDocsDict = {}
DATA_DICT: DataDict = {}
TYPEVAR_DICT: TypeVarDict = {}

MANIM_ROOT = Path(__file__).resolve().parent.parent.parent

# In the following, we will use ``type(xyz) is xyz_type`` instead of
# isinstance checks to make sure no subclasses of the type pass the
# check
# ruff: noqa: E721


def parse_module_attributes() -> tuple[AliasDocsDict, DataDict, TypeVarDict]:
    """Read all files, generate Abstract Syntax Trees from them, and
    extract useful information about the type aliases defined in the
    files: the category they belong to, their definition and their
    description, separating them from the "regular" module attributes.

    Returns
    -------
    ALIAS_DOCS_DICT : :class:`AliasDocsDict`
        A dictionary containing the information from all the type
        aliases in Manim. See :class:`AliasDocsDict` for more information.

    DATA_DICT : :class:`DataDict`
        A dictionary containing the names of all DOCUMENTED
        module-level attributes which are not a :class:`TypeAlias`.

    TYPEVAR_DICT : :class:`TypeVarDict`
        A dictionary containing the definitions of :class:`TypeVar` objects,
        organized by modules.
    """
    global ALIAS_DOCS_DICT
    global DATA_DICT
    global TYPEVAR_DICT

    if ALIAS_DOCS_DICT or DATA_DICT or TYPEVAR_DICT:
        return ALIAS_DOCS_DICT, DATA_DICT, TYPEVAR_DICT

    for module_path in MANIM_ROOT.rglob("*.py"):
        module_name_t1 = module_path.resolve().relative_to(MANIM_ROOT)
        module_name_t2 = list(module_name_t1.parts)
        module_name_t2[-1] = module_name_t2[-1].removesuffix(".py")
        module_name = ".".join(module_name_t2)

        module_content = module_path.read_text(encoding="utf-8")

        # For storing TypeAliases
        module_dict: ModuleLevelAliasDict = {}
        category_dict: AliasCategoryDict | None = None
        alias_info: AliasInfo | None = None

        # For storing TypeVars
        module_typevars: ModuleTypeVarDict = {}

        # For storing regular module attributes
        data_list: list[str] = []
        data_name: str | None = None

        for node in ast.iter_child_nodes(ast.parse(module_content)):
            # If we encounter a string:
            if (
                type(node) is ast.Expr
                and type(node.value) is ast.Constant
                and type(node.value.value) is str
            ):
                string = node.value.value.strip()
                # It can be the start of a category
                section_str = "[CATEGORY]"
                if string.startswith(section_str):
                    category_name = string[len(section_str) :].strip()
                    module_dict[category_name] = {}
                    category_dict = module_dict[category_name]
                    alias_info = None
                # or a docstring of the alias defined before
                elif alias_info:
                    alias_info["doc"] = string
                # or a docstring of the module attribute defined before
                elif data_name:
                    data_list.append(data_name)
                continue

            # if it's defined under if TYPE_CHECKING
            # go through the body of the if statement
            if (
                # NOTE: This logic does not (and cannot)
                # check if the comparison is against a
                # variable called TYPE_CHECKING
                # It also says that you cannot do the following
                # import typing as foo
                # if foo.TYPE_CHECKING:
                #   BAR: TypeAlias = ...
                type(node) is ast.If
                and (
                    (
                        # if TYPE_CHECKING
                        type(node.test) is ast.Name and node.test.id == "TYPE_CHECKING"
                    )
                    or (
                        # if typing.TYPE_CHECKING
                        type(node.test) is ast.Attribute
                        and type(node.test.value) is ast.Name
                        and node.test.value.id == "typing"
                        and node.test.attr == "TYPE_CHECKING"
                    )
                )
            ):
                inner_nodes: list[Any] = node.body
            else:
                inner_nodes = [node]

            for node in inner_nodes:
                # Check if this node is a TypeAlias (type <name> = <value>)
                # or an AnnAssign annotated as TypeAlias (<target>: TypeAlias = <value>).
                is_type_alias = (
                    sys.version_info >= (3, 12) and type(node) is ast.TypeAlias
                )
                is_annotated_assignment_with_value = (
                    type(node) is ast.AnnAssign
                    and type(node.annotation) is ast.Name
                    and node.annotation.id == "TypeAlias"
                    and type(node.target) is ast.Name
                    and node.value is not None
                )
                if is_type_alias or is_annotated_assignment_with_value:
                    # TODO: ast.TypeAlias does not exist before Python 3.12, and that
                    # could be the reason why MyPy does not recognize these as
                    # attributes of node.
                    alias_name = node.name.id if is_type_alias else node.target.id  # type: ignore[attr-defined]
                    definition_node = node.value  # type: ignore[attr-defined]

                    # If the definition is a Union, replace with vertical bar notation.
                    # Instead of "Union[Type1, Type2]", we'll have "Type1 | Type2".
                    if (
                        type(definition_node) is ast.Subscript
                        and type(definition_node.value) is ast.Name
                        and definition_node.value.id == "Union"
                    ):
                        union_elements = definition_node.slice.elts  # type: ignore[attr-defined]
                        definition = " | ".join(
                            ast.unparse(elem) for elem in union_elements
                        )
                    else:
                        definition = ast.unparse(definition_node)

                    definition = definition.replace("npt.", "")
                    if category_dict is None:
                        module_dict[""] = {}
                        category_dict = module_dict[""]
                    category_dict[alias_name] = {"definition": definition}
                    alias_info = category_dict[alias_name]
                    continue

                # Check if it is a typing.TypeVar (<target> = TypeVar(...)).
                elif (
                    type(node) is ast.Assign
                    and type(node.targets[0]) is ast.Name
                    and type(node.value) is ast.Call
                    and type(node.value.func) is ast.Name
                    and node.value.func.id.endswith("TypeVar")
                ):
                    module_typevars[node.targets[0].id] = ast.unparse(
                        node.value
                    ).replace("_", r"\_")
                    continue

                # If here, the node is not a TypeAlias definition
                alias_info = None

                # It could still be a module attribute definition.
                # Does the assignment have a target of type Name? Then
                # it could be considered a definition of a module attribute.
                if type(node) is ast.AnnAssign:
                    target: Name | Attribute | Subscript | ast.expr | None = node.target
                elif type(node) is ast.Assign and len(node.targets) == 1:
                    target = node.targets[0]
                else:
                    target = None

                if type(target) is ast.Name and not (
                    type(node) is ast.Assign and target.id not in module_typevars
                ):
                    data_name = target.id
                else:
                    data_name = None

        if len(module_dict) > 0:
            ALIAS_DOCS_DICT[module_name] = module_dict
        if len(data_list) > 0:
            DATA_DICT[module_name] = data_list
        if module_typevars:
            TYPEVAR_DICT[module_name] = module_typevars

    return ALIAS_DOCS_DICT, DATA_DICT, TYPEVAR_DICT
