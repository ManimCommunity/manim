import numpy as np

from manim import BLACK, PURE_GREEN, Scene
from manim.mobject.opengl_mobject import OpenGLMobject
from manim.mobject.types.opengl_vectorized_mobject import OpenGLVMobject


def test_import_color(using_opengl_renderer):
    import manim.utils.color as C

    C.WHITE


def test_background_color(using_opengl_renderer):
    S = Scene()
    S.renderer.background_color = "#ff0000"
    S.renderer.update_frame(S)
    assert np.all(S.renderer.get_frame()[0, 0] == np.array([255, 0, 0, 255]))

    S.renderer.background_color = "#436f80"
    S.renderer.update_frame(S)
    assert np.all(S.renderer.get_frame()[0, 0] == np.array([67, 111, 128, 255]))

    S.renderer.background_color = "#fff"
    S.renderer.update_frame(S)
    assert np.all(S.renderer.get_frame()[0, 0] == np.array([255, 255, 255, 255]))


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
