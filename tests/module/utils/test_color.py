from __future__ import annotations

import numpy as np

from manim import BLACK, RED, WHITE, ManimColor, Mobject, Scene, VMobject


def test_import_color():
    import manim.utils.color as C

    C.WHITE


def test_background_color():
    S = Scene()
    S.camera.background_color = "#ff0000"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([255, 0, 0, 255])
    )

    S.camera.background_color = "#436f80"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([67, 111, 128, 255])
    )

    S.camera.background_color = "#ffffff"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([255, 255, 255, 255])
    )

    S.camera.background_color = "#bbffbb"
    S.camera.background_opacity = 0.5
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([187, 255, 187, 127])
    )


def test_set_color():
    m = Mobject()
    assert m.color.to_hex() == "#FFFFFF"
    m.set_color(BLACK)
    assert m.color.to_hex() == "#000000"

    m = VMobject()
    assert m.color.to_hex() == "#FFFFFF"
    m.set_color(BLACK)
    assert m.color.to_hex() == "#000000"


def test_color_hash():
    assert hash(WHITE) == hash(ManimColor([1.0, 1.0, 1.0, 1.0]))
    assert hash(WHITE) == hash("#FFFFFFFF")
    assert hash(WHITE) != hash(RED)
