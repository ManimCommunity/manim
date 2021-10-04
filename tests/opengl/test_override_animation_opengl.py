import pytest

from manim import Animation, override_animation
from manim.mobject.opengl_mobject import OpenGLMobject
from manim.utils.exceptions import MultiAnimationOverrideException


class AnimationA1(Animation):
    pass


class AnimationA2(Animation):
    pass


class AnimationA3(Animation):
    pass


class AnimationB1(AnimationA1):
    pass


class AnimationC1(AnimationB1):
    pass


class AnimationX(Animation):
    pass


class OpenGLMobjectA(OpenGLMobject):
    @override_animation(AnimationA1)
    def anim_a1(self):
        return AnimationA2(self)

    @override_animation(AnimationX)
    def anim_x(self, *args, **kwargs):
        return args, kwargs


class OpenGLMobjectB(OpenGLMobjectA):
    pass


class OpenGLMobjectC(OpenGLMobjectB):
    @override_animation(AnimationA1)
    def anim_a1(self):
        return AnimationA3(self)


class OpenGLMobjectX(OpenGLMobject):
    @override_animation(AnimationB1)
    def animation(self):
        return "Overridden"


@pytest.mark.xfail(reason="Needs investigating")
def test_opengl_mobject_inheritance():
    mob = OpenGLMobject()
    a = OpenGLMobjectA()
    b = OpenGLMobjectB()
    c = OpenGLMobjectC()

    assert type(AnimationA1(mob)) is AnimationA1
    assert type(AnimationA1(a)) is AnimationA2
    assert type(AnimationA1(b)) is AnimationA2
    assert type(AnimationA1(c)) is AnimationA3


@pytest.mark.xfail(reason="Needs investigating")
def test_arguments():
    a = OpenGLMobjectA()
    args = (1, "two", {"three": 3}, ["f", "o", "u", "r"])
    kwargs = {"test": "manim", "keyword": 42, "arguments": []}
    animA = AnimationX(a, *args, **kwargs)

    assert animA[0] == args
    assert animA[1] == kwargs


@pytest.mark.xfail(reason="Needs investigating")
def test_multi_animation_override_exception():
    with pytest.raises(MultiAnimationOverrideException):

        class OpenGLMobjectB2(OpenGLMobjectA):
            @override_animation(AnimationA1)
            def anim_a1_different_name(self):
                pass


@pytest.mark.xfail(reason="Needs investigating")
def test_animation_inheritance():
    x = OpenGLMobjectX()

    assert type(AnimationA1(x)) is AnimationA1
    assert AnimationB1(x) == "Overridden"
    assert type(AnimationC1(x)) is AnimationC1
