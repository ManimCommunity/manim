from __future__ import annotations

from manim import Circle, Manager, ReplacementTransform, Scene, Square, VGroup


def test_no_duplicate_references():
    with Manager(Scene) as manager:
        scene = manager.scene
        c = Circle()
        sq = Square()
        scene.add(c, sq)

        scene.play(ReplacementTransform(c, sq))
    assert len(scene.mobjects) == 1
    assert scene.mobjects[0] is sq


def test_duplicate_references_in_group():
    with Manager(Scene) as manager:
        scene = manager.scene
        c = Circle()
        sq = Square()
        vg = VGroup(c, sq)
        scene.add(vg)

        scene.play(ReplacementTransform(c, sq))
    submobs = vg.submobjects
    assert len(submobs) == 1
    assert submobs[0] is sq
