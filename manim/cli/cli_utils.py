from __future__ import annotations

import re
import sys

from manim._config import console
from manim.constants import CHOOSE_NUMBER_MESSAGE

ESCAPE_CHAR = "CTRL+Z" if sys.platform == "win32" else "CTRL+D"
NOT_FOUND_IMPORT = "Import statement for Manim was not found. Importing is added."

INPUT_CODE_ENTER = f"Enter the animation code & end with an EOF: {ESCAPE_CHAR}:"


def code_input_prompt() -> str:
    console.print(INPUT_CODE_ENTER)
    code = sys.stdin.read()
    if len(code.strip()) == 0:
        raise ValueError("Empty input of code")

    if not code.startswith("from manim import"):
        console.print(NOT_FOUND_IMPORT, style="logging.level.warning")
        code = "from manim import *\n" + code
    return code


def prompt_user_with_choice(choise_list: list[str]) -> list[int]:
    """Prompt user with chooses and return indices of choised items"""
    max_index = len(choise_list)
    for count, name in enumerate(choise_list, 1):
        console.print(f"{count}: {name}", style="logging.level.info")

    user_input = console.input(CHOOSE_NUMBER_MESSAGE)
    # CTRL + Z, CTRL + D, Remove common EOF escape chars
    cleaned = user_input.strip().removesuffix("\x1a").removesuffix("\x04")
    result = re.split(r"\s*,\s*", cleaned)

    if not all(a.isnumeric() for a in result):
        raise ValueError("Invalid non-numeric input: ", user_input)

    indices = [int(i_str.strip()) - 1 for i_str in result]
    if all(a <= max_index >= 0 for a in indices):
        return indices
    else:
        raise KeyError("One or more chooses is outside of range")
