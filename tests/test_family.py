from manim import *

def test_family():
    """Check that the family is gathered correclty."""
    # Check that an empty mobject's family only contains itself
    mob = Mobject()
    assert mob.get_family() == [mob]

    # Check that all children are in the family
    mob = Mobject()
    children = [Mobject() for _ in range(10)]
    mob.add(*children)
    family = mob.get_family()
    assert len(family) == 1 + 10
    assert mob in family
    for c in children:
        assert c in family

    # Nested children should be in the family
    mob = Mobject()
    grandchildren = {}
    for _ in range(10):
        child = Mobject()
        grandchildren[child] = [Mobject() for _ in range(10)]
        child.add(*grandchildren[child])
    mob.add(*list(grandchildren.keys()))
    family = mob.get_family()
    assert len(family) == 1 + 10 + 10 * 10
    assert mob in family
    for c in grandchildren:
        assert c in family
        for gc in grandchildren[c]:
            assert gc in family


def test_overlapping_family():
    """Check that each member of the family is only gathered once."""
    mob, child1, child2, = Mobject(), Mobject(), Mobject(),
    gchild1, gchild2, gchild_common = Mobject(), Mobject(), Mobject()
    child1.add(gchild1, gchild_common)
    child2.add(gchild2, gchild_common)
    mob.add(child1, child2)
    family = mob.get_family()
    assert mob in family
    assert len(family) == 6
    assert family.count(gchild_common) == 1
