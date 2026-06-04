from __future__ import annotations

from manim import Mobject
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.opengl.opengl_mobject import OpenGLMobject


def test_metaclass_registry(config):
    class SomeTestMobject(Mobject, metaclass=ConvertToOpenGL):
        pass

    assert SomeTestMobject in ConvertToOpenGL._converted_classes

    config.renderer = "opengl"
    assert OpenGLMobject in SomeTestMobject.__bases__
    assert Mobject not in SomeTestMobject.__bases__

    config.renderer = "cairo"
    assert Mobject in SomeTestMobject.__bases__
    assert OpenGLMobject not in SomeTestMobject.__bases__
