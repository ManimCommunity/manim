from manim import Mobject, config, tempconfig
from manim.mobject.opengl_compatibility import ConvertToOpenGL
from manim.mobject.opengl_mobject import OpenGLMobject


def test_metaclass_registry():
    class SomeTestMobject(Mobject, metaclass=ConvertToOpenGL):
        pass

    assert SomeTestMobject in ConvertToOpenGL._converted_classes

    with tempconfig({"renderer": "opengl"}):
        assert OpenGLMobject in SomeTestMobject.__bases__
        assert Mobject not in SomeTestMobject.__bases__

        config.renderer = "cairo"
        assert Mobject in SomeTestMobject.__bases__
        assert OpenGLMobject not in SomeTestMobject.__bases__
