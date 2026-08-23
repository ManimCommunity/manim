from __future__ import annotations

import numpy as np

from manim import RIGHT, Circle
from manim.mobject.opengl.opengl_mobject import OpenGLMobject


def test_family(using_opengl_renderer):
    """Check that the family is gathered correctly."""
    # Check that an empty OpenGLMobject's family only contains itself
    mob = OpenGLMobject()
    assert mob.get_family() == [mob]

    # Check that all children are in the family
    mob = OpenGLMobject()
    children = [OpenGLMobject() for _ in range(10)]
    mob.add(*children)
    family = mob.get_family()
    assert len(family) == 1 + 10
    assert mob in family
    for c in children:
        assert c in family

    # Nested children should be in the family
    mob = OpenGLMobject()
    grandchildren = {}
    for _ in range(10):
        child = OpenGLMobject()
        grandchildren[child] = [OpenGLMobject() for _ in range(10)]
        child.add(*grandchildren[child])
    mob.add(*list(grandchildren.keys()))
    family = mob.get_family()
    assert len(family) == 1 + 10 + 10 * 10
    assert mob in family
    for c in grandchildren:
        assert c in family
        for gc in grandchildren[c]:
            assert gc in family


def test_overlapping_family(using_opengl_renderer):
    """Check that each member of the family is only gathered once."""
    (
        mob,
        child1,
        child2,
    ) = (
        OpenGLMobject(),
        OpenGLMobject(),
        OpenGLMobject(),
    )
    gchild1, gchild2, gchild_common = OpenGLMobject(), OpenGLMobject(), OpenGLMobject()
    child1.add(gchild1, gchild_common)
    child2.add(gchild2, gchild_common)
    mob.add(child1, child2)
    family = mob.get_family()
    assert mob in family
    assert len(family) == 6
    assert family.count(gchild_common) == 1


def test_shift_family(using_opengl_renderer):
    """Check that each member of the family is shifted along with the parent.

    Importantly, here we add a common grandchild to each of the children.  So
    this test will fail if the grandchild moves twice as much as it should.

    """
    # Note shift() needs the OpenGLMobject to have a non-empty `points` attribute, so
    # we cannot use a plain OpenGLMobject or OpenGLVMobject.  We use Circle instead.
    (
        mob,
        child1,
        child2,
    ) = (
        Circle(),
        Circle(),
        Circle(),
    )
    gchild1, gchild2, gchild_common = Circle(), Circle(), Circle()

    child1.add(gchild1, gchild_common)
    child2.add(gchild2, gchild_common)
    mob.add(child1, child2)
    family = mob.get_family()

    positions_before = {m: m.get_center().copy() for m in family}
    mob.shift(RIGHT)
    positions_after = {m: m.get_center().copy() for m in family}

    for m in family:
        np.testing.assert_allclose(positions_before[m] + RIGHT, positions_after[m])


def test_opengl_mobject_family_updated_on_change(using_opengl_renderer):
    """Test that the family of an OpenGLMobject is updated correctly when submobjects
    are added or removed, and that the family is not updated if no changes are made.
    """
    # This is based on the assumption that obj.family is replaced with a new list when submobjects
    # are added or removed. If this implementation detail changes, this test may need to be updated.
    obj = OpenGLMobject()
    family = obj.get_family()
    assert family == [obj]
    submobs = [OpenGLMobject() for _ in range(10)]

    # Add new submobjects; family should be updated.
    obj.add(*submobs)
    assert family is not obj.get_family()
    family = obj.get_family()
    assert len(family) == 11
    for submob in submobs:
        assert submob in family

    # Remove a submobject; family should be updated.
    obj.remove(submobs[0])
    family = obj.get_family()
    assert len(family) == 10
    assert submobs[0] not in family

    # Remove a submobject that is not in the family; family should not be updated.
    obj.remove(OpenGLMobject())
    assert family is obj.get_family()

    # Add a submobject that is already in the family; family should not be updated.
    obj.add(submobs[1])
    assert family is obj.get_family()

    # Add a mix of new and existing submobjects; family should be updated.
    obj.add(OpenGLMobject(), submobs[2])
    assert family is not obj.get_family()
    family = obj.get_family()
    assert len(family) == 11

    # Remove a mix of existing and non-existing submobjects; family should be updated.
    obj.remove(submobs[3], OpenGLMobject())
    assert family is not obj.get_family()


def test_opengl_mobject_add_updates_parents(using_opengl_renderer):
    """Test that the parents of an OpenGLMobject are updated correctly when they are added to
    another OpenGLMobject.
    """
    parent = OpenGLMobject(name="parent")
    children = [OpenGLMobject(name=f"child_{i}") for i in range(3)]
    child = children[0]

    # Initially, the child has no parent.
    assert child.parents == []

    # Add the child to the parent; the child's parent should be updated.
    parent.add(child)
    assert child.parents == [parent]

    # Add the child to another parent; the child's parent should be updated.
    new_parent = OpenGLMobject()
    new_parent.add(child)
    for p in parent, new_parent:
        assert p in child.parents

    # Remove the child from the new parent; the child's parent should be updated.
    new_parent.remove(child)
    assert new_parent not in child.parents

    # Add a child multiple times to the same parent; the child's parent should not be duplicated.
    parent.remove(*parent.submobjects)
    parent.add(child)
    parent.add(child)
    assert child.parents == [parent]

    # Remove a mix of existing and non-existing children; the child's parent should be updated correctly.
    child2, child3 = OpenGLMobject(), OpenGLMobject()
    parent.remove(child2, child, child3)
    for c in [child, child2, child3]:
        assert parent not in c.parents
