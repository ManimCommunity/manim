from __future__ import annotations

import numpy as np

from manim import BLACK, BLUE, GREEN, PURE_BLUE, PURE_GREEN, PURE_RED, Scene
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject


def test_import_color(using_opengl_renderer):
    import manim.utils.color as C

    C.WHITE


def test_background_color(using_opengl_renderer):
    S = Scene()
    S.renderer.background_color = "#FF0000"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([255, 0, 0, 255])
    )

    S.renderer.background_color = "#436F80"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([67, 111, 128, 255])
    )

    S.renderer.background_color = "#FFFFFF"
    S.renderer.update_frame(S)
    np.testing.assert_array_equal(
        S.renderer.get_frame()[0, 0], np.array([255, 255, 255, 255])
    )


def test_set_color(using_opengl_renderer):
    m = OpenGLMobject()
    assert m.color.to_hex() == "#FFFFFF"
    np.all(m.rgbas == np.array([[0.0, 0.0, 0.0, 1.0]]))

    m.set_color(BLACK)
    assert m.color.to_hex() == "#000000"
    np.all(m.rgbas == np.array([[1.0, 1.0, 1.0, 1.0]]))

    m.set_color(PURE_GREEN, opacity=0.5)
    assert m.color.to_hex() == "#00FF00"
    np.all(m.rgbas == np.array([[0.0, 1.0, 0.0, 0.5]]))

    m = OpenGLVMobject()
    assert m.color.to_hex() == "#FFFFFF"
    np.all(m.fill_rgba == np.array([[0.0, 0.0, 0.0, 1.0]]))
    np.all(m.stroke_rgba == np.array([[0.0, 0.0, 0.0, 1.0]]))

    m.set_color(BLACK)
    assert m.color.to_hex() == "#000000"
    np.all(m.fill_rgba == np.array([[1.0, 1.0, 1.0, 1.0]]))
    np.all(m.stroke_rgba == np.array([[1.0, 1.0, 1.0, 1.0]]))

    m.set_color(PURE_GREEN, opacity=0.5)
    assert m.color.to_hex() == "#00FF00"
    np.all(m.fill_rgba == np.array([[0.0, 1.0, 0.0, 0.5]]))
    np.all(m.stroke_rgba == np.array([[0.0, 1.0, 0.0, 0.5]]))


def test_set_fill_color(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.fill_color.to_hex() == "#FFFFFF"
    np.all(m.fill_rgba == np.array([[0.0, 1.0, 0.0, 0.5]]))

    m.set_fill(BLACK)
    assert m.fill_color.to_hex() == "#000000"
    np.all(m.fill_rgba == np.array([[1.0, 1.0, 1.0, 1.0]]))

    m.set_fill(PURE_GREEN, opacity=0.5)
    assert m.fill_color.to_hex() == "#00FF00"
    np.all(m.fill_rgba == np.array([[0.0, 1.0, 0.0, 0.5]]))


def test_set_stroke_color(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.stroke_color.to_hex() == "#FFFFFF"
    np.all(m.stroke_rgba == np.array([[0.0, 1.0, 0.0, 0.5]]))

    m.set_stroke(BLACK)
    assert m.stroke_color.to_hex() == "#000000"
    np.all(m.stroke_rgba == np.array([[1.0, 1.0, 1.0, 1.0]]))

    m.set_stroke(PURE_GREEN, opacity=0.5)
    assert m.stroke_color.to_hex() == "#00FF00"
    np.all(m.stroke_rgba == np.array([[0.0, 1.0, 0.0, 0.5]]))


def test_set_fill(using_opengl_renderer):
    m = OpenGLMobject()
    assert m.color.to_hex() == "#FFFFFF"
    m.set_color(BLACK)
    assert m.color.to_hex() == "#000000"

    m = OpenGLVMobject()
    assert m.color.to_hex() == "#FFFFFF"
    m.set_color(BLACK)
    assert m.color.to_hex() == "#000000"


def test_set_color_handles_lists_of_strs(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.color.to_hex() == "#FFFFFF"
    m.set_color([BLACK, BLUE, GREEN])
    assert m.get_colors()[0] == BLACK
    assert m.get_colors()[1] == BLUE
    assert m.get_colors()[2] == GREEN

    assert m.get_fill_colors()[0] == BLACK
    assert m.get_fill_colors()[1] == BLUE
    assert m.get_fill_colors()[2] == GREEN

    assert m.get_stroke_colors()[0] == BLACK
    assert m.get_stroke_colors()[1] == BLUE
    assert m.get_stroke_colors()[2] == GREEN


def test_set_color_handles_lists_of_color_objects(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.color.to_hex() == "#FFFFFF"
    m.set_color([PURE_BLUE, PURE_GREEN, PURE_RED])
    assert m.get_colors()[0].to_hex() == "#0000FF"
    assert m.get_colors()[1].to_hex() == "#00FF00"
    assert m.get_colors()[2].to_hex() == "#FF0000"

    assert m.get_fill_colors()[0].to_hex() == "#0000FF"
    assert m.get_fill_colors()[1].to_hex() == "#00FF00"
    assert m.get_fill_colors()[2].to_hex() == "#FF0000"

    assert m.get_stroke_colors()[0].to_hex() == "#0000FF"
    assert m.get_stroke_colors()[1].to_hex() == "#00FF00"
    assert m.get_stroke_colors()[2].to_hex() == "#FF0000"


def test_set_fill_handles_lists_of_strs(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.fill_color.to_hex() == "#FFFFFF"
    m.set_fill([BLACK.to_hex(), BLUE.to_hex(), GREEN.to_hex()])
    assert m.get_fill_colors()[0].to_hex() == BLACK.to_hex()
    assert m.get_fill_colors()[1].to_hex() == BLUE.to_hex()
    assert m.get_fill_colors()[2].to_hex() == GREEN.to_hex()


def test_set_fill_handles_lists_of_color_objects(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.fill_color.to_hex() == "#FFFFFF"
    m.set_fill([PURE_BLUE, PURE_GREEN, PURE_RED])
    assert m.get_fill_colors()[0].to_hex() == "#0000FF"
    assert m.get_fill_colors()[1].to_hex() == "#00FF00"
    assert m.get_fill_colors()[2].to_hex() == "#FF0000"


def test_set_stroke_handles_lists_of_strs(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.stroke_color.to_hex() == "#FFFFFF"
    m.set_stroke([BLACK.to_hex(), BLUE.to_hex(), GREEN.to_hex()])
    assert m.get_stroke_colors()[0].to_hex() == BLACK.to_hex()
    assert m.get_stroke_colors()[1].to_hex() == BLUE.to_hex()
    assert m.get_stroke_colors()[2].to_hex() == GREEN.to_hex()


def test_set_stroke_handles_lists_of_color_objects(using_opengl_renderer):
    m = OpenGLVMobject()
    assert m.stroke_color.to_hex() == "#FFFFFF"
    m.set_stroke([PURE_BLUE, PURE_GREEN, PURE_RED])
    assert m.get_stroke_colors()[0].to_hex() == "#0000FF"
    assert m.get_stroke_colors()[1].to_hex() == "#00FF00"
    assert m.get_stroke_colors()[2].to_hex() == "#FF0000"
