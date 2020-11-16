import pytest
import numpy as np

from manim import Camera, Scene, tempconfig, config


def test_import_color():
    import manim.utils.color as C

    C.WHITE


def test_background_color():
    import manim.utils.color as C

    S = Scene()
    S.camera.background_color = C.RED
    S.renderer.update_frame(S)
    assert np.all(S.renderer.get_frame()[0, 0] == np.array([252, 98, 85, 255]))

    S.camera.background_color = C.BLUE
    S.renderer.update_frame(S)
    assert np.all(S.renderer.get_frame()[0, 0] == np.array([88, 196, 221, 255]))

    S.camera.background_color = C.GREEN
    S.camera.background_opacity = 0.5
    S.renderer.update_frame(S)
    assert np.all(S.renderer.get_frame()[0, 0] == np.array([131, 193, 103, 127]))