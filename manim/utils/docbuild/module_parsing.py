from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeAlias

__all__ = ["parse_module_attributes"]


MANIM_ROOT = Path(__file__).resolve().parent.parent.parent

AliasDocsDict: TypeAlias = dict[str, dict[str, dict[str, str]]]
DataDict: TypeAlias = dict[str, list[str]]

ALIAS_DOCS_DICT: AliasDocsDict = {}
DATA_DICT: DataDict = {}


def parse_module_attributes() -> tuple[AliasDocsDict, DataDict]:
    """Read all files, generate an Abstract Syntax Tree
    from it, and extract useful information about the type aliases
    defined in the file: the category they belong to, their definition
    and their description.

    Returns
    -------
    MANIM_TYPING_DOCS : dict[str, dict[str, dict[str, str]]
        A dictionary containing the information from all the type
        aliases. Each key is the name of a category of types, and
        its corresponding value is another subdictionary containing
        information about the type aliases under that category:

        -   The keys of this subdictionary are the names of the type
            aliases, and the values are subsubdictionaries containing
            field-value pairs with information about the type alias.
    """
    global ALIAS_DOCS_DICT
    global DATA_DICT

    if ALIAS_DOCS_DICT or DATA_DICT:
        return ALIAS_DOCS_DICT, DATA_DICT

    for module_path in MANIM_ROOT.rglob("*.py"):
        module_name = module_path.resolve().relative_to(MANIM_ROOT)
        module_name = list(module_name.parts)
        module_name[-1] = module_name[-1][:-3]  # remove .py
        module_name = ".".join(module_name)

        module_content = module_path.read_text()

        # For storing TypeAliases
        module_dict: dict[str, dict[str, dict[str, str]]] = {}
        category_dict: dict[str, dict[str, str]] | None = None
        alias_dict: dict[str, str] | None = None

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
                    alias_dict = None
                # or a docstring of the alias defined before
                elif alias_dict:
                    alias_dict["doc"] = string
                # or a docstring of the module attribute defined before
                elif data_name:
                    data_list.append(data_name)
                continue

            # If we encounter an assignment annotated as "TypeAlias":
            if (
                type(node) is ast.AnnAssign
                and type(node.annotation) is ast.Name
                and node.annotation.id == "TypeAlias"
                and type(node.target) is ast.Name
                and node.value is not None
            ):
                alias_name = node.target.id
                def_node = node.value
                # If it's an Union, replace it with vertical bar notation
                if (
                    type(def_node) is ast.Subscript
                    and type(def_node.value) is ast.Name
                    and def_node.value.id == "Union"
                ):
                    definition = " | ".join(
                        ast.unparse(elem) for elem in def_node.slice.elts
                    )
                else:
                    definition = ast.unparse(def_node)
                # for subnode in ast.walk(node.value):
                #     print(ast.dump(subnode, indent=4), end="\n\n")
                definition = definition.replace("npt.", "")
                if category_dict is None:
                    module_dict[""] = {}
                    category_dict = module_dict[""]
                category_dict[alias_name] = {"definition": definition}
                alias_dict = category_dict[alias_name]
                continue

            # The node is not a TypeAlias definition
            alias_dict = None

            # It could still be a module attribute definition
            if type(node) is ast.AnnAssign:
                target = node.target
            elif type(node) is ast.Assign and len(node.targets) == 1:
                target = node.targets[0]
            else:
                target = None

            if type(target) is ast.Name:
                data_name = target.id
            else:
                data_name = None

        if len(module_dict) > 0:
            ALIAS_DOCS_DICT[module_name] = module_dict
        if len(data_list) > 0:
            DATA_DICT[module_name] = data_list

    return ALIAS_DOCS_DICT, DATA_DICT
