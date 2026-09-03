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
    child, child2, child3 = [OpenGLMobject(name=f"child_{i}") for i in range(3)]

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

    # Add a child multiple times to the same parent; the child's parent should not
    # be duplicated.
    parent.remove(*parent.submobjects)
    parent.add(child2)
    parent.add(child2)
    assert child2.parents == [parent]


def test_opengl_mobject_remove_updates_parents(using_opengl_renderer):
    """Test that the parents of an OpenGLMobject are updated correctly when they are removed from
    another OpenGLMobject.
    """
    parent = OpenGLMobject(name="parent")
    child, child2, child3 = [OpenGLMobject(name=f"child_{i}") for i in range(3)]

    parent.add(child)

    # Remove the child from the parent; the child's parent should be updated.
    parent.remove(child)
    assert child.parents == []

    # Remove a child with multiple parents
    new_parent = OpenGLMobject()
    parent.add(child)
    new_parent.add(child)
    parent.remove(child)
    assert child.parents == [new_parent]
    assert parent not in child.parents

    # Remove a child that is not in the parent; the child's parent should not be updated.
    parent.remove(child2)
    assert child2.parents == []

    # Remove the child from the new parent; the child's parent should be updated.
    new_parent.remove(child)
    assert new_parent not in child.parents

    # Remove a mix of existing and non-existing children; the existing child's parent should be
    # updated correctly.
    parent.add(child)
    assert child.parents == [parent]
    parent.remove(child2, child, child3)
    for c in [child, child2, child3]:
        assert parent not in c.parents


def test_opengl_mobject_replace_submobject_updates_parents(using_opengl_renderer):
    """Test that replace_submobject() updates the parents of both the removed
    and inserted submobjects correctly.
    """
    parent = OpenGLMobject()
    old_submob = OpenGLMobject()
    new_submob = OpenGLMobject()
    parent.add(old_submob)

    parent.replace_submobject(0, new_submob)

    assert parent not in old_submob.parents
    assert new_submob.parents == [parent]

    # Inserting the same mobject again should not affect the parent list
    parent.add(OpenGLMobject())
    parent.replace_submobject(1, new_submob)
    assert new_submob.parents == [parent]


def test_opengl_mobject_insert_updates_parents(using_opengl_renderer):
    """Test that insert() updates the parents of the inserted submobjects correctly."""
    parent = OpenGLMobject()
    submob1 = OpenGLMobject()
    submob2 = OpenGLMobject()

    parent.insert(0, submob1)
    assert submob1.parents == [parent]

    parent.insert(1, submob2)
    assert submob1.parents == [parent]
    assert submob2.parents == [parent]

    # Inserting an existing submobject should not affect the parent list
    parent.insert(0, submob1)
    assert submob1.parents == [parent]


def test_opengl_mobject_add_to_back_updates_parents(using_opengl_renderer):
    """Test that add_to_back() updates the parents of the added submobjects correctly."""
    parent = OpenGLMobject()
    submob1 = OpenGLMobject()
    submob2 = OpenGLMobject()

    parent.add_to_back(submob1)
    assert submob1.parents == [parent]

    parent.add_to_back(submob2)
    assert submob1.parents == [parent]
    assert submob2.parents == [parent]

    # Adding an existing submobject should not affect the parent list
    parent.add_to_back(submob1)
    assert submob1.parents == [parent]
