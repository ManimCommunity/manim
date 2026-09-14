from __future__ import annotations

import colorsys

import numpy as np
import numpy.testing as nt

from manim.utils.color import (
    BLACK,
    HSV,
    RED,
    WHITE,
    YELLOW,
    ManimColor,
    ManimColorDType,
)
from manim.utils.color.XKCD import GREEN


def test_init_with_int() -> None:
    color = ManimColor(0x123456, 0.5)
    nt.assert_array_equal(
        color._internal_value,
        np.array([0x12, 0x34, 0x56, 0.5 * 255], dtype=ManimColorDType) / 255,
    )
    color = BLACK
    nt.assert_array_equal(
        color._internal_value, np.array([0, 0, 0, 1.0], dtype=ManimColorDType)
    )
    color = WHITE
    nt.assert_array_equal(
        color._internal_value, np.array([1.0, 1.0, 1.0, 1.0], dtype=ManimColorDType)
    )


def test_init_with_hex() -> None:
    color = ManimColor("0xFF0000")
    nt.assert_array_equal(color._internal_value, np.array([1, 0, 0, 1]))
    color = ManimColor("0xFF000000")
    nt.assert_array_equal(color._internal_value, np.array([1, 0, 0, 0]))

    color = ManimColor("#FF0000")
    nt.assert_array_equal(color._internal_value, np.array([1, 0, 0, 1]))
    color = ManimColor("#FF000000")
    nt.assert_array_equal(color._internal_value, np.array([1, 0, 0, 0]))


def test_init_with_hex_short() -> None:
    color = ManimColor("#F00")
    nt.assert_array_equal(color._internal_value, np.array([1, 0, 0, 1]))
    color = ManimColor("0xF00")
    nt.assert_array_equal(color._internal_value, np.array([1, 0, 0, 1]))

    color = ManimColor("#F000")
    nt.assert_array_equal(color._internal_value, np.array([1, 0, 0, 0]))
    color = ManimColor("0xF000")
    nt.assert_array_equal(color._internal_value, np.array([1, 0, 0, 0]))


def test_init_with_string() -> None:
    color = ManimColor("BLACK")
    nt.assert_array_equal(color._internal_value, BLACK._internal_value)


def test_init_with_tuple_int() -> None:
    color = ManimColor((50, 10, 50))
    nt.assert_array_equal(
        color._internal_value, np.array([50 / 255, 10 / 255, 50 / 255, 1.0])
    )

    color = ManimColor((50, 10, 50, 50))
    nt.assert_array_equal(
        color._internal_value, np.array([50 / 255, 10 / 255, 50 / 255, 50 / 255])
    )


def test_init_with_tuple_float() -> None:
    color = ManimColor((0.5, 0.6, 0.7))
    nt.assert_array_equal(color._internal_value, np.array([0.5, 0.6, 0.7, 1.0]))

    color = ManimColor((0.5, 0.6, 0.7, 0.1))
    nt.assert_array_equal(color._internal_value, np.array([0.5, 0.6, 0.7, 0.1]))


def test_to_integer() -> None:
    color = ManimColor((0x1, 0x2, 0x3, 0x4))
    nt.assert_equal(color.to_integer(), 0x010203)


def test_to_rgb() -> None:
    color = ManimColor((0x1, 0x2, 0x3, 0x4))
    nt.assert_array_equal(color.to_rgb(), (0x1 / 255, 0x2 / 255, 0x3 / 255))
    nt.assert_array_equal(color.to_int_rgb(), (0x1, 0x2, 0x3))
    nt.assert_array_equal(color.to_rgba(), (0x1 / 255, 0x2 / 255, 0x3 / 255, 0x4 / 255))
    nt.assert_array_equal(color.to_int_rgba(), (0x1, 0x2, 0x3, 0x4))
    nt.assert_array_equal(
        color.to_rgba_with_alpha(0.5), (0x1 / 255, 0x2 / 255, 0x3 / 255, 0.5)
    )
    nt.assert_array_equal(
        color.to_int_rgba_with_alpha(0.5), (0x1, 0x2, 0x3, int(0.5 * 255))
    )


def test_to_hex() -> None:
    color = ManimColor((0x1, 0x2, 0x3, 0x4))
    nt.assert_equal(color.to_hex(), "#010203")
    nt.assert_equal(color.to_hex(True), "#01020304")


def test_to_hsv() -> None:
    color = ManimColor((0x1, 0x2, 0x3, 0x4))
    nt.assert_array_equal(
        color.to_hsv(), colorsys.rgb_to_hsv(0x1 / 255, 0x2 / 255, 0x3 / 255)
    )


def test_to_hsl() -> None:
    color = ManimColor((0x1, 0x2, 0x3, 0x4))
    hls = colorsys.rgb_to_hls(0x1 / 255, 0x2 / 255, 0x3 / 255)

    nt.assert_array_equal(color.to_hsl(), np.array([hls[0], hls[2], hls[1]]))


def test_from_hsl() -> None:
    hls = colorsys.rgb_to_hls(0x1 / 255, 0x2 / 255, 0x3 / 255)
    hsl = np.array([hls[0], hls[2], hls[1]])

    color = ManimColor.from_hsl(hsl)
    rgb = np.array([0x1 / 255, 0x2 / 255, 0x3 / 255])

    nt.assert_allclose(color.to_rgb(), rgb)


def test_invert() -> None:
    color = ManimColor((0x1, 0x2, 0x3, 0x4))
    rgba = color._internal_value
    inverted = color.invert()
    nt.assert_array_equal(
        inverted._internal_value, (1 - rgba[0], 1 - rgba[1], 1 - rgba[2], rgba[3])
    )


def test_invert_with_alpha() -> None:
    color = ManimColor((0x1, 0x2, 0x3, 0x4))
    rgba = color._internal_value
    inverted = color.invert(True)
    nt.assert_array_equal(
        inverted._internal_value, (1 - rgba[0], 1 - rgba[1], 1 - rgba[2], 1 - rgba[3])
    )


def test_interpolate() -> None:
    r1 = RED._internal_value
    r2 = YELLOW._internal_value
    nt.assert_array_equal(
        RED.interpolate(YELLOW, 0.5)._internal_value, 0.5 * r1 + 0.5 * r2
    )


def test_opacity() -> None:
    nt.assert_equal(RED.opacity(0.5)._internal_value[3], 0.5)


def test_parse() -> None:
    nt.assert_equal(ManimColor.parse([RED, YELLOW]), [RED, YELLOW])


def test_mc_operators() -> None:
    c1 = RED
    c2 = GREEN
    halfway1 = 0.5 * c1 + 0.5 * c2
    halfway2 = c1.interpolate(c2, 0.5)
    nt.assert_equal(halfway1, halfway2)
    nt.assert_array_equal((WHITE / 2.0)._internal_value, np.array([0.5, 0.5, 0.5, 0.5]))


def test_mc_from_functions() -> None:
    color = ManimColor.from_hex("#ff00a0")
    nt.assert_equal(color.to_hex(), "#FF00A0")

    color = ManimColor.from_rgb((1.0, 1.0, 0.0))
    nt.assert_equal(color.to_hex(), "#FFFF00")

    color = ManimColor.from_rgba((1.0, 1.0, 0.0, 1.0))
    nt.assert_equal(color.to_hex(True), "#FFFF00FF")

    color = ManimColor.from_hsv((1.0, 1.0, 1.0), alpha=0.0)
    nt.assert_equal(color.to_hex(True), "#FF000000")


def test_hsv_init() -> None:
    color = HSV((0.25, 1, 1))
    nt.assert_array_equal(color._internal_value, np.array([0.5, 1.0, 0.0, 1.0]))


def test_into_HSV() -> None:
    nt.assert_equal(RED.into(HSV).into(ManimColor), RED)


def test_contrasting() -> None:
    nt.assert_equal(BLACK.contrasting(), WHITE)
    nt.assert_equal(WHITE.contrasting(), BLACK)
    nt.assert_equal(RED.contrasting(0.1), BLACK)
    nt.assert_equal(RED.contrasting(0.9), WHITE)
    nt.assert_equal(BLACK.contrasting(dark=GREEN, light=RED), RED)
    nt.assert_equal(WHITE.contrasting(dark=GREEN, light=RED), GREEN)


def test_lighter() -> None:
    c = RED.opacity(0.42)
    cl = c.lighter(0.2)
    nt.assert_array_equal(
        cl._internal_value[:3],
        0.8 * c._internal_value[:3] + 0.2 * WHITE._internal_value[:3],
    )
    nt.assert_equal(cl[-1], c[-1])


def test_darker() -> None:
    c = RED.opacity(0.42)
    cd = c.darker(0.2)
    nt.assert_array_equal(
        cd._internal_value[:3],
        0.8 * c._internal_value[:3] + 0.2 * BLACK._internal_value[:3],
    )
    nt.assert_equal(cd[-1], c[-1])
