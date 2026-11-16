from __future__ import annotations

from manim import UP, Circle, Dot, FadeIn, Square
from manim.animation.updaters.mobject_update_utils import turn_animation_into_updater


def test_turn_animation_into_updater_zero_run_time():
    """Test that turn_animation_into_updater handles zero run_time correctly."""
    # Create a simple mobject and animation
    mobject = Circle()
    animation = FadeIn(mobject, run_time=0)

    # Track updater calls
    update_calls = []
    original_updaters = mobject.updaters.copy()

    # Call turn_animation_into_updater
    result = turn_animation_into_updater(animation)

    # Verify mobject is returned
    assert result is mobject

    # Get the updater that was added
    assert len(mobject.updaters) == len(original_updaters) + 1
    updater = mobject.updaters[-1]

    # Simulate calling the updater
    updater(mobject, dt=0.1)

    # The updater should have finished and removed itself
    assert len(mobject.updaters) == len(original_updaters)
    assert updater not in mobject.updaters

    # Animation should be in finished state
    assert animation.total_time >= 0


def test_turn_animation_into_updater_positive_run_time_persists():
    """Test that updater persists with positive run_time."""
    mobject = Circle()
    animation = FadeIn(mobject, run_time=1.0)

    original_updaters = mobject.updaters.copy()

    # Call turn_animation_into_updater
    result = turn_animation_into_updater(animation)

    # Get the updater that was added
    updater = mobject.updaters[-1]

    # Simulate calling the updater (partial progress)
    updater(mobject, dt=0.1)

    # The updater should still be present (not finished)
    assert len(mobject.updaters) == len(original_updaters) + 1
    assert updater in mobject.updaters


def test_always():
    d = Dot()
    circ = Circle()
    d.always.next_to(circ, UP)
    assert len(d.updaters) == 1
    # we should be able to chain updaters
    d2 = Dot()
    d.always.next_to(d2, UP).next_to(circ, UP)
    assert len(d.updaters) == 3


def test_non_time_based_updater_modifies_mobject():
    """Test that non-time-based updaters apply their changes every update step"""
    square = Square(side_length=1, fill_opacity=0)
    square.add_updater(lambda m: m.set(width=m.width + 1))
    square.update(0)  # dt is ignored for non-time-based updaters
    assert square.width == 2.0

    # Multiple updaters should apply
    square.add_updater(lambda m: m.shift(UP))
    square.update(100)
    assert square.width == 3.0
    assert (square.get_center() == UP).all()

    # When multiple updaters conflict, they should be applied in the order they were added
    # Add a new updater that doubles the current displacement
    square.add_updater(lambda m: m.shift(m.get_center()))
    square.update(0)
    assert square.width == 4.0
    # Second shift must be based on the displacement after the first shift
    assert (square.get_center() == 4 * UP).all()


def test_time_based_updater_modifies_mobject():
    """Test that time-based updaters apply their changes correctly based on dt"""
    square = Square(side_length=1, fill_opacity=0)
    square.add_updater(lambda m, dt: m.set(width=m.width + dt))
    square.update(0.5)  # Simulate waiting 0.5 seconds
    assert square.width == 1.5

    # Multiple updaters should apply
    square.add_updater(lambda m, dt: m.set_fill(opacity=dt))
    square.update(0.5)
    assert square.width == 2.0
    assert square.get_fill_opacity() == 0.5

    # When multiple updaters affect the same attribute, they should be applied in the
    # order they were added
    square.add_updater(lambda m, dt: m.set_fill(opacity=dt / 2))
    square.update(0.5)
    assert square.width == 2.5
    assert square.get_fill_opacity() == 0.25  # last updater overrides previous

    # Time-based updaters should handle dt=0 correctly
    square.update(0)
    assert square.width == 2.5  # width is linear in dt here, so no change
    assert square.get_fill_opacity() == 0  # dt sets opacity directly


def test_updater_remove():
    """Test that updaters can be removed correctly"""
    square = Square(side_length=1, fill_opacity=0)

    def updater1(m):
        return

    def updater2(m, dt):
        return

    square.add_updater(updater1)
    square.add_updater(updater2)

    square.remove_updater(updater1)
    assert updater1 not in square.updaters
    assert updater2 in square.updaters

    square.remove_updater(updater2)
    assert updater2 not in square.updaters

    # Removing an updater that is not present should not raise an error
    square.remove_updater(updater1)

    # Multiple instances of the same updater should be removed with one call
    square.add_updater(updater1)
    square.add_updater(updater1)
    assert len(square.updaters) == 2
    square.remove_updater(updater1)
    assert len(square.updaters) == 0


def test_updater_multiples():
    """Test that the same updater can be added multiple times and removed correctly"""
    square = Square(side_length=1, fill_opacity=0)

    def updater(m):
        m.shift(UP)

    assert len(square.updaters) == 0
    square.add_updater(updater)
    assert len(square.updaters) == 1
    square.update(0)
    assert (square.get_center() == UP).all()

    # Add the same updater again
    square.add_updater(updater)
    assert len(square.updaters) == 2
    square.update(0)
    assert (square.get_center() == 3 * UP).all()  # shifted twice
