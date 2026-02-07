from __future__ import annotations

from manim import Circle, Manager, ReplacementTransform, Scene, Square, Triangle, VGroup


def test_no_duplicate_references():
    manager = Manager(Scene)
    scene = manager.scene
    c = Circle()
    sq = Square()
    scene.add(c, sq)

    scene.play(ReplacementTransform(c, sq))
    assert len(scene.mobjects) == 1
    assert scene.mobjects[0] is sq


def test_duplicate_references_in_group():
    manager = Manager(Scene)
    scene = manager.scene
    c = Circle()
    sq = Square()
    vg = VGroup(c, sq)
    scene.add(vg)

    scene.play(ReplacementTransform(c, sq))
    submobs = vg.submobjects
    assert len(submobs) == 1
    assert submobs[0] is sq


def test_duplicate_references_in_multiple_groups():
    manager = Manager(Scene)
    scene = manager.scene
    c = Circle()
    sq = Square()
    tr = Triangle()
    vg_1 = VGroup(c, sq)
    vg_2 = VGroup(c, tr)
    scene.add(vg_1, vg_2)

    scene.play(ReplacementTransform(c, sq))
    submobs_1 = vg_1.submobjects
    submobs_2 = vg_2.submobjects
    assert len(submobs_1) == 1
    assert submobs_1[0] is sq
    assert len(submobs_2) == 2
    assert submobs_2[0] is sq
    assert submobs_2[1] is tr
