from __future__ import annotations

import ast
from pathlib import Path

from manim import typing

__all__ = ["get_typing_docs"]


def get_typing_docs() -> dict[str, dict[str, dict[str, str]]]:
    """Read the manim/typing.py file, generate an Abstract Syntax Tree
    from it, and extract useful information about the type aliases
    defined in the file: the category they belong to, their definition
    and their description.

    Returns
    -------
    typing_docs_dict : dict[str, dict[str, dict[str, str]]
        A dictionary containing the information from all the type
        aliases. Each key is the name of a category of types, and
        its corresponding value is another subdictionary containing
        information about the type aliases under that category:

        -   The keys of this subdictionary are the names of the type
            aliases, and the values are subsubdictionaries containing
            field-value pairs with information about the type alias.
    """

    with open(typing.__file__) as typing_file:
        typing_file_content = typing_file.read()

    typing_docs_dict: dict[str, dict[str, dict[str, str]]] = {}
    category_dict: dict[str, dict[str, str]] | None = None
    alias_dict: dict[str, str] | None = None

    for node in ast.walk(ast.parse(typing_file_content)):
        if (
            type(node) is ast.Expr
            and type(node.value) is ast.Constant
            and type(node.value.value) is str
        ):
            string = node.value.value.strip()
            # It can be the start of a category
            section_str = "[CATEGORY]"
            if string.startswith(section_str):
                category_name = string[len(section_str):].strip()
                typing_docs_dict[category_name] = {}
                category_dict = typing_docs_dict[category_name]
                alias_dict = None
            # or a docstring of the alias defined before
            elif alias_dict:
                alias_dict["doc"] = string

        elif category_dict is None:
            continue

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
            category_dict[alias_name] = {"definition": definition}
            alias_dict = category_dict[alias_name]

        else:
            alias_dict = None

    return typing_docs_dict
