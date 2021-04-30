import pytest

from manim import (
    Animation,
    Mobject,
    override_animation,
)
from manim.utils.exceptions import MultiAnimationOverrideException


class AnimA1(Animation):
    pass
class AnimA2(Animation):
    pass
class AnimA3(Animation):
    pass


class AnimB1(AnimA1):
    pass


class AnimC1(AnimB1):
    pass

class AnimX(Animation):
    pass


class MobjA(Mobject):
    @override_animation(AnimA1)
    def anim_a1(self):
        return AnimA2(self)

    @override_animation(AnimX)
    def anim_x(self, *args, **kwargs):
        return args, kwargs


class MobjB(MobjA):
    pass


class MobjC(MobjB):
    @override_animation(AnimA1)
    def anim_a1(self):
        return AnimA3(self)


class MobjX(Mobject):
    @override_animation(AnimB1)
    def animation(self):
        return "Overridden"


def test_mobj_inheritance():
    override_animation.setup()
    mob = Mobject()
    a = MobjA()
    b = MobjB()
    c = MobjC()

    assert type(AnimA1(mob)) is AnimA1
    assert type(AnimA1(a)) is AnimA2
    assert type(AnimA1(b)) is AnimA2
    assert type(AnimA1(c)) is AnimA3


def test_arguments():
    a = MobjA()
    args = (1, "two", {"three": 3}, ["f", "o", "u", "r"])
    kwargs = {"test": "manim", "keyword": 42, "arguments": []}
    animA = AnimX(a, *args, **kwargs)

    assert animA[0] == args
    assert animA[1] == kwargs


def test_multi_animation_override_exception():
    class MobjB2(MobjA):
        @override_animation(AnimA1)
        def anim_a1_different_name(self):
            return None

    with pytest.raises(MultiAnimationOverrideException):
        override_animation.setup()


def test_animation_inheritance():
    x = MobjX()

    assert type(AnimA1(x)) is AnimA1
    assert AnimB1(x) == "Overridden"
    assert type(AnimC1(x)) is AnimC1
