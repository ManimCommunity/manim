from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "composition"


@frames_comparison
def test_animationgroup_is_passing_remover_to_animations(scene):
    animation_group = AnimationGroup(Create(Square()), Write(Circle()), remover=True)

    scene.play(animation_group)
    scene.wait(0.1)


@frames_comparison
def test_animationgroup_is_passing_remover_to_nested_animationgroups(scene):
    animation_group = AnimationGroup(
        AnimationGroup(Create(Square()), Create(RegularPolygon(5))),
        Write(Circle(), remover=True),
        remover=True,
    )

    scene.play(animation_group)
    scene.wait(0.1)
