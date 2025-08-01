from __future__ import annotations

import sys

from manim._config import console
from manim.constants import CHOOSE_NUMBER_MESSAGE

ESCAPE_CHAR = "CTRL+Z" if sys.platform == "win32" else "CTRL+D"

NOT_FOUND_IMPORT = "Import statement for Manim was not found. Importing is added."
INPUT_CODE_ENTER = f"Enter the animation code & end with an EOF: {ESCAPE_CHAR}:"


def code_input_prompt() -> str:
    """Little CLI interface in which user can insert code."""
    console.print(INPUT_CODE_ENTER)
    code = sys.stdin.read()
    if len(code.strip()) == 0:
        raise ValueError("Empty input of code")

    if not code.startswith("from manim import"):
        console.print(NOT_FOUND_IMPORT, style="logging.level.warning")
        code = "from manim import *\n" + code
    return code


def prompt_user_with_list(items: list[str]) -> list[int]:
    """Prompt user with choices and return indices of chosen items

    Parameters
    -----------
    items
        list of strings representing items to be chosen
    """
    max_index = len(items) - 1
    for count, name in enumerate(items, 1):
        console.print(f"{count}: {name}", style="logging.level.info")

    user_input = console.input(CHOOSE_NUMBER_MESSAGE)
    result = user_input.strip().rstrip(",").split(",")
    cleaned = [n.strip() for n in result]

    if not all(a.isnumeric() for a in cleaned):
        raise ValueError(f"Invalid non-numeric input(s): {result}")

    indices = [int(int_str) - 1 for int_str in cleaned]
    if all(a <= max_index >= 0 for a in indices):
        return indices
    else:
        raise KeyError("One or more choice is outside of range")
