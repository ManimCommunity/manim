from __future__ import annotations

import numpy as np
from colour import Color

from manim import BLACK, BLUE, GREEN, PURE_BLUE, PURE_GREEN, PURE_RED, Scene
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject


def test_import_color(using_opengl_renderer):
    import manim.utils.color as C

    C.WHITE


def test_background_color(using_opengl_renderer):
    S = Scene()
    S.renderer.background_color = "#ff0000"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([255, 0, 0, 255])
    )

    S.renderer.background_color = "#436f80"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([67, 111, 128, 255])
    )

    S.renderer.background_color = "#fff"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([255, 255, 255, 255])
    )


def test_set_color(using_opengl_renderer):
    m = OpenGLMobject()
    assert m.color.hex == "#fff"
    np.alltrue(m.rgbas == np.array((0.0, 0.0, 0.0, 1.0)))

    m.set_color(BLACK)
    assert m.color.hex == "#000"
    np.alltrue(m.rgbas == np.array((1.0, 1.0, 1.0, 1.0)))

    m.set_color(PURE_GREEN, opacity=0.5)
    assert m.color.hex == "#0f0"
    np.alltrue(m.rgbas == np.array((0.0, 1.0, 0.0, 0.5)))

    m = OpenGLVMobject()
    assert m.color.hex == "#fff"
    np.alltrue(m.fill_rgba == np.array((0.0, 0.0, 0.0, 1.0)))
    np.alltrue(m.stroke_rgba == np.array((0.0, 0.0, 0.0, 1.0)))

    m.set_color(BLACK)
    assert m.color.hex == "#000"
    np.alltrue(m.fill_rgba == np.array((1.0, 1.0, 1.0, 1.0)))
    np.alltrue(m.stroke_rgba == np.array((1.0, 1.0, 1.0, 1.0)))

    m.set_color(PURE_GREEN, opacity=0.5)
    assert m.color.hex == "#0f0"
    np.alltrue(m.fill_rgba == np.array((0.0, 1.0, 0.0, 0.5)))
    np.alltrue(m.stroke_rgba == np.array((0.0, 1.0, 0.0, 0.5)))


def test_set_fill_color(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.fill_color.hex == "#fff"
    np.alltrue(m.fill_rgba == np.array((0.0, 1.0, 0.0, 0.5)))

    m.set_fill(BLACK)
    assert m.fill_color.hex == "#000"
    np.alltrue(m.fill_rgba == np.array((1.0, 1.0, 1.0, 1.0)))

    m.set_fill(PURE_GREEN, opacity=0.5)
    assert m.fill_color.hex == "#0f0"
    np.alltrue(m.fill_rgba == np.array((0.0, 1.0, 0.0, 0.5)))


def test_set_stroke_color(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.stroke_color.hex == "#fff"
    np.alltrue(m.stroke_rgba == np.array((0.0, 1.0, 0.0, 0.5)))

    m.set_stroke(BLACK)
    assert m.stroke_color.hex == "#000"
    np.alltrue(m.stroke_rgba == np.array((1.0, 1.0, 1.0, 1.0)))

    m.set_stroke(PURE_GREEN, opacity=0.5)
    assert m.stroke_color.hex == "#0f0"
    np.alltrue(m.stroke_rgba == np.array((0.0, 1.0, 0.0, 0.5)))


def test_set_fill(using_opengl_renderer):
    m = OpenGLMobject()
    assert m.color.hex == "#fff"
    m.set_color(BLACK)
    assert m.color.hex == "#000"

    m = OpenGLVMobject()
    assert m.color.hex == "#fff"
    m.set_color(BLACK)
    assert m.color.hex == "#000"


def test_set_color_handles_lists_of_strs(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.color.hex == "#fff"
    m.set_color([BLACK, BLUE, GREEN])
    assert m.get_colors()[0] == Color(BLACK)
    assert m.get_colors()[1] == Color(BLUE)
    assert m.get_colors()[2] == Color(GREEN)

    assert m.get_fill_colors()[0] == Color(BLACK)
    assert m.get_fill_colors()[1] == Color(BLUE)
    assert m.get_fill_colors()[2] == Color(GREEN)

    assert m.get_stroke_colors()[0] == Color(BLACK)
    assert m.get_stroke_colors()[1] == Color(BLUE)
    assert m.get_stroke_colors()[2] == Color(GREEN)


def test_set_color_handles_lists_of_color_objects(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.color.hex == "#fff"
    m.set_color([Color(PURE_BLUE), Color(PURE_GREEN), Color(PURE_RED)])
    assert m.get_colors()[0].hex == "#00f"
    assert m.get_colors()[1].hex == "#0f0"
    assert m.get_colors()[2].hex == "#f00"

    assert m.get_fill_colors()[0].hex == "#00f"
    assert m.get_fill_colors()[1].hex == "#0f0"
    assert m.get_fill_colors()[2].hex == "#f00"

    assert m.get_stroke_colors()[0].hex == "#00f"
    assert m.get_stroke_colors()[1].hex == "#0f0"
    assert m.get_stroke_colors()[2].hex == "#f00"


def test_set_fill_handles_lists_of_strs(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.fill_color.hex == "#fff"
    m.set_fill([BLACK, BLUE, GREEN])
    assert m.get_fill_colors()[0] == Color(BLACK)
    assert m.get_fill_colors()[1] == Color(BLUE)
    assert m.get_fill_colors()[2] == Color(GREEN)


def test_set_fill_handles_lists_of_color_objects(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.fill_color.hex == "#fff"
    m.set_fill([Color(PURE_BLUE), Color(PURE_GREEN), Color(PURE_RED)])
    assert m.get_fill_colors()[0].hex == "#00f"
    assert m.get_fill_colors()[1].hex == "#0f0"
    assert m.get_fill_colors()[2].hex == "#f00"


def test_set_stroke_handles_lists_of_strs(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.stroke_color.hex == "#fff"
    m.set_stroke([BLACK, BLUE, GREEN])
    assert m.get_stroke_colors()[0] == Color(BLACK)
    assert m.get_stroke_colors()[1] == Color(BLUE)
    assert m.get_stroke_colors()[2] == Color(GREEN)


def test_set_stroke_handles_lists_of_color_objects(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.stroke_color.hex == "#fff"
    m.set_stroke([Color(PURE_BLUE), Color(PURE_GREEN), Color(PURE_RED)])
    assert m.get_stroke_colors()[0].hex == "#00f"
    assert m.get_stroke_colors()[1].hex == "#0f0"
    assert m.get_stroke_colors()[2].hex == "#f00"
