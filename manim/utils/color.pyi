from enum import Enum
from typing import Any, Iterable, List, Union

import numpy as np
from colour import Color

class Colors(Enum):
    white: str
    gray_a: str
    gray_b: str
    gray_c: str
    gray_d: str
    gray_e: str
    black: str
    lighter_gray: Any
    light_gray: Any
    gray: Any
    dark_gray: Any
    darker_gray: Any
    blue_a: str
    blue_b: str
    blue_c: str
    blue_d: str
    blue_e: str
    pure_blue: str
    blue: Any
    dark_blue: Any
    teal_a: str
    teal_b: str
    teal_c: str
    teal_d: str
    teal_e: str
    teal: Any
    green_a: str
    green_b: str
    green_c: str
    green_d: str
    green_e: str
    pure_green: str
    green: Any
    yellow_a: str
    yellow_b: str
    yellow_c: str
    yellow_d: str
    yellow_e: str
    yellow: Any
    gold_a: str
    gold_b: str
    gold_c: str
    gold_d: str
    gold_e: str
    gold: Any
    red_a: str
    red_b: str
    red_c: str
    red_d: str
    red_e: str
    pure_red: str
    red: Any
    maroon_a: str
    maroon_b: str
    maroon_c: str
    maroon_d: str
    maroon_e: str
    maroon: Any
    purple_a: str
    purple_b: str
    purple_c: str
    purple_d: str
    purple_e: str
    purple: Any
    pink: str
    light_pink: str
    orange: str
    light_brown: str
    dark_brown: str
    gray_brown: str

def color_to_rgb(color: Union[Color, str]) -> np.ndarray: ...
def color_to_rgba(color: Union[Color, str], alpha: float = ...) -> np.ndarray: ...
def rgb_to_color(rgb: Iterable[float]) -> Color: ...
def rgba_to_color(rgba: Iterable[float]) -> Color: ...
def rgb_to_hex(rgb: Iterable[float]) -> str: ...
def hex_to_rgb(hex_code: str) -> np.ndarray: ...
def invert_color(color: Color) -> Color: ...
def color_to_int_rgb(color: Color) -> np.ndarray: ...
def color_to_int_rgba(color: Color, opacity: float = ...) -> np.ndarray: ...
def color_gradient(
    reference_colors: Iterable[Color], length_of_output: int
) -> List[Color]: ...
def interpolate_color(color1: Color, color2: Color, alpha: float) -> Color: ...
def average_color(*colors: Color) -> Color: ...
def random_bright_color() -> Color: ...
def random_color() -> Color: ...
def get_shaded_rgb(
    rgb: np.ndarray,
    point: np.ndarray,
    unit_normal_vect: np.ndarray,
    light_source: np.ndarray,
) -> np.ndarray: ...

BLACK: str
BLUE: str
BLUE_A: str
BLUE_B: str
BLUE_C: str
BLUE_D: str
BLUE_E: str
DARKER_GRAY: str
DARKER_GREY: str
DARK_BLUE: str
DARK_BROWN: str
DARK_GRAY: str
DARK_GREY: str
GOLD: str
GOLD_A: str
GOLD_B: str
GOLD_C: str
GOLD_D: str
GOLD_E: str
GRAY: str
GRAY_A: str
GRAY_B: str
GRAY_BROWN: str
GRAY_C: str
GRAY_D: str
GRAY_E: str
GREEN: str
GREEN_A: str
GREEN_B: str
GREEN_C: str
GREEN_D: str
GREEN_E: str
GREY: str
GREY_A: str
GREY_B: str
GREY_BROWN: str
GREY_C: str
GREY_D: str
GREY_E: str
LIGHTER_GRAY: str
LIGHTER_GREY: str
LIGHT_BROWN: str
LIGHT_GRAY: str
LIGHT_GREY: str
LIGHT_PINK: str
MAROON: str
MAROON_A: str
MAROON_B: str
MAROON_C: str
MAROON_D: str
MAROON_E: str
ORANGE: str
PINK: str
PURE_BLUE: str
PURE_GREEN: str
PURE_RED: str
PURPLE: str
PURPLE_A: str
PURPLE_B: str
PURPLE_C: str
PURPLE_D: str
PURPLE_E: str
RED: str
RED_A: str
RED_B: str
RED_C: str
RED_D: str
RED_E: str
TEAL: str
TEAL_A: str
TEAL_B: str
TEAL_C: str
TEAL_D: str
TEAL_E: str
WHITE: str
YELLOW: str
YELLOW_A: str
YELLOW_B: str
YELLOW_C: str
YELLOW_D: str
YELLOW_E: str
