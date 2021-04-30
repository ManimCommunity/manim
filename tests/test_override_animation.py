
from manim.animation.transform import Transform
from manim.constants import RIGHT
from manim import (
    Mobject,
    Animation,
    override_animation,
    FadeInFrom,
    FadeIn,
    FadeInFromLarge,
    FadeInFromPoint,
    FadeOut,
    FadeOutToPoint,
    ORIGIN,
)
from manim.utils.exceptions import MultiAnimationOverrideException

import pytest

class AnimA(Animation):
    pass

class AnimB(AnimA):
    pass

class AnimC(AnimB):
    pass


class MobjA(Mobject):
    @override_animation(FadeIn)
    def fadeIn(self):
        return FadeInFrom(self)

    @override_animation(AnimA)
    def animA(self, *args, **kwargs):
        return args, kwargs


class MobjB(MobjA):
    pass


class MobjC(MobjB):
    @override_animation(FadeIn)
    def fadeIn(self):
        return FadeInFromLarge(self)

class MobjA2(Mobject):
    @override_animation(AnimB)
    def animation(self):
        return "Overridden"


def test_mobj_inheritance():
    override_animation.setup()
    mob = Mobject()
    a = MobjA()
    b = MobjB()
    c = MobjC()

    assert type(FadeIn(mob)) is FadeIn
    assert type(FadeIn(a)) is FadeInFrom
    assert type(FadeIn(b)) is FadeInFrom
    assert type(FadeIn(c)) is FadeInFromLarge

def test_arguments():
    a = MobjA()
    args = (1,"two",{"three":3},["f","o","u","r"])
    kwargs = {
        "test": "manim",
        "keyword": 42,
        "arguments": []
    }
    animA = AnimA(a, *args, **kwargs)

    assert animA[0] == args
    assert animA[1] == kwargs

def test_multi_animation_override_exception():
    class MobjB2(MobjA):
        @override_animation(FadeIn)
        def fadeIn2(self):
            return FadeInFrom(self)
    
    with pytest.raises(MultiAnimationOverrideException):
        override_animation.setup()

def test_animation_inheritance():
    a2 = MobjA2()

    assert type(AnimA(a2)) is AnimA
    assert AnimB(a2) == "Overridden"
    assert type(AnimC(a2)) is AnimC
