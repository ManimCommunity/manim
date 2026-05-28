from __future__ import annotations

import numpy as np
import numpy.testing as nt
import pytest

from manim.utils.color import (
    BLACK,
    BLUE,
    GREEN,
    RED,
    WHITE,
    YELLOW,
    ManimColor,
)
from manim.utils.color.core import (
    RandomColorGenerator,
    average_color,
    color_gradient,
    color_to_int_rgb,
    color_to_int_rgba,
    color_to_rgb,
    color_to_rgba,
    hex_to_rgb,
    interpolate_color,
    invert_color,
    random_bright_color,
    random_color,
    rgb_to_color,
    rgb_to_hex,
    rgba_to_color,
)

# ---------------------------------------------------------------------------
# Parsing — one case per linearly independent input branch in ManimColor.__init__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # : PT006
    ("color_input", "expected_rgb"),
    [
        ("#FF0000", (1.0, 0.0, 0.0)),
        ("#F00", (1.0, 0.0, 0.0)),
        ("RED", (0xFC / 255, 0x62 / 255, 0x55 / 255)),
        (0xFF0000, (1.0, 0.0, 0.0)),
        ((255, 0, 0), (1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (RED, (0xFC / 255, 0x62 / 255, 0x55 / 255)),
    ],
    ids=[
        "hex_long",
        "hex_short",
        "name",
        "packed_int",
        "int_tuple",
        "float_tuple",
        "ManimColor",
    ],
)
def test_color_to_rgb_accepts_all_parsable_forms(color_input, expected_rgb) -> None:
    nt.assert_allclose(color_to_rgb(color_input), expected_rgb)


def test_color_to_rgb_returns_a_float64_array_of_length_3() -> None:
    rgb = color_to_rgb("#123456")
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (3,)
    assert rgb.dtype == np.float64


def test_color_to_rgb_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Color TOMATO not found"):
        color_to_rgb("TOMATO")


# ---------------------------------------------------------------------------
# Alpha & int conversions
# ---------------------------------------------------------------------------


def test_color_to_rgba_default_alpha_is_opaque() -> None:
    nt.assert_array_equal(color_to_rgba("#FF0000"), (1.0, 0.0, 0.0, 1.0))


def test_color_to_rgba_uses_alpha_argument() -> None:
    nt.assert_array_equal(color_to_rgba("#FF0000", alpha=0.25), (1.0, 0.0, 0.0, 0.25))


def test_color_to_int_rgb_returns_signed_ints_in_0_255_range() -> None:
    int_rgb = color_to_int_rgb("#FF8040")
    nt.assert_array_equal(int_rgb, (0xFF, 0x80, 0x40))
    assert int_rgb.dtype.kind == "i"


def test_color_to_int_rgba_default_alpha_is_fully_opaque_byte() -> None:
    # Pins the default alpha=1.0 in the signature (without this, mutations
    # to the default value silently survive).
    nt.assert_array_equal(color_to_int_rgba("#FF8040"), (0xFF, 0x80, 0x40, 255))


def test_color_to_int_rgba_appends_alpha_byte() -> None:
    nt.assert_array_equal(
        color_to_int_rgba("#FF8040", alpha=0.5),
        (0xFF, 0x80, 0x40, int(0.5 * 255)),
    )


# ---------------------------------------------------------------------------
# Inverse direction — rgb_to_color / rgba_to_color route through from_rgb /
# from_rgba, a different code path from ManimColor(value).
# ---------------------------------------------------------------------------


def test_rgb_to_color_normalizes_int_input_to_floats() -> None:
    assert rgb_to_color((255, 128, 0)) == ManimColor((1.0, 128 / 255, 0.0))


def test_rgba_to_color_preserves_alpha() -> None:
    assert rgba_to_color((1.0, 0.0, 0.0, 0.25)) == ManimColor((1.0, 0.0, 0.0, 0.25))


# ---------------------------------------------------------------------------
# Hex ↔ RGB
# ---------------------------------------------------------------------------


def test_rgb_to_hex_format_is_uppercase_with_hash() -> None:
    nt.assert_equal(rgb_to_hex((1.0, 0.0, 0xA0 / 255)), "#FF00A0")


@pytest.mark.parametrize(
    "hex_input",
    ["#000000", "#FFFFFF", "#FF0000", "#FF00A0"],
)
def test_hex_rgb_roundtrip_is_lossless_for_8bit_aligned_values(hex_input) -> None:
    nt.assert_equal(rgb_to_hex(hex_to_rgb(hex_input)), hex_input)


def test_rgb_hex_roundtrip_drift_is_under_one_byte_per_channel() -> None:
    rgb = np.array([0.42, 0.18, 0.93])
    nt.assert_allclose(hex_to_rgb(rgb_to_hex(rgb)), rgb, atol=1 / 255)


# ---------------------------------------------------------------------------
# invert_color
# ---------------------------------------------------------------------------


def test_invert_color_flips_white_to_black() -> None:
    # Anchors the actual semantic (1 - x), independent of the involution property.
    assert invert_color(WHITE) == BLACK


@pytest.mark.parametrize("color", [RED, GREEN, BLUE, YELLOW])
def test_invert_color_is_an_involution(color) -> None:
    # ManimColor.__eq__ uses np.allclose, absorbing the machine-epsilon drift.
    assert invert_color(invert_color(color)) == color


def test_invert_color_preserves_alpha_by_default() -> None:
    c = ManimColor("#FF0000", alpha=0.3)
    nt.assert_equal(invert_color(c)._internal_value[3], 0.3)


# ---------------------------------------------------------------------------
# interpolate_color — three samples of a linear function in alpha
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # : PT006
    ("alpha", "expected"),
    [
        (0.0, BLACK),
        (0.5, ManimColor((0.5, 0.5, 0.5))),
        (1.0, WHITE),
    ],
    ids=["start", "midpoint", "end"],
)
def test_interpolate_color_between_black_and_white(alpha, expected) -> None:
    assert interpolate_color(BLACK, WHITE, alpha) == expected


# ---------------------------------------------------------------------------
# average_color
# ---------------------------------------------------------------------------


def test_average_color_of_black_and_white_is_mid_gray() -> None:
    assert average_color(BLACK, WHITE) == ManimColor((0.5, 0.5, 0.5))


def test_average_color_always_returns_alpha_one() -> None:
    # average_color drops input alpha per its docstring contract.
    avg = average_color(
        ManimColor("#FF0000", alpha=0.1), ManimColor("#FF0000", alpha=0.9)
    )
    nt.assert_equal(avg._internal_value[3], 1.0)


# ---------------------------------------------------------------------------
# color_gradient
# ---------------------------------------------------------------------------


def test_color_gradient_zero_length_returns_empty_list() -> None:
    assert color_gradient([RED], 0) == []


def test_color_gradient_empty_reference_with_positive_length_raises() -> None:
    with pytest.raises(ValueError, match="Expected 1 or more reference colors"):
        color_gradient([], 5)


def test_color_gradient_single_reference_is_repeated_n_times() -> None:
    gradient = color_gradient([RED], 5)
    assert len(gradient) == 5
    assert all(color == RED for color in gradient)


def test_color_gradient_interpolates_endpoints_and_respects_length() -> None:
    gradient = color_gradient([BLACK, WHITE], 7)
    assert len(gradient) == 7
    assert gradient[0] == BLACK
    assert gradient[-1] == WHITE


def test_color_gradient_passes_through_each_of_four_reference_colors() -> None:
    # With >= 4 reference colors the internal `num_colors - 2` bookkeeping
    # diverges from `num_colors % 2`; pins the former.
    refs = [BLACK, RED, BLUE, WHITE]
    gradient = color_gradient(refs, 4)
    assert len(gradient) == 4
    assert gradient[0] == BLACK
    assert gradient[-1] == WHITE


# ---------------------------------------------------------------------------
# Random color machinery
# ---------------------------------------------------------------------------


def test_random_color_returns_a_manim_color() -> None:
    assert isinstance(random_color(), ManimColor)


def test_random_bright_color_has_every_channel_at_or_above_half() -> None:
    # By construction: 0.5 * (random_rgb + 1) => each channel >= 0.5.
    assert (random_bright_color().to_rgb() >= 0.5).all()


def test_random_color_generator_is_deterministic_under_a_fixed_seed() -> None:
    a = RandomColorGenerator(seed=42)
    b = RandomColorGenerator(seed=42)
    for _ in range(5):
        assert a.next() == b.next()


def test_random_color_generator_only_samples_from_custom_palette() -> None:
    palette = [RED, GREEN, BLUE]
    gen = RandomColorGenerator(seed=1, sample_colors=palette)
    for _ in range(10):
        assert gen.next() in palette
