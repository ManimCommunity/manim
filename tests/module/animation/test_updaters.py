from __future__ import annotations

from manim import UP, Circle, Dot, FadeIn
from manim.animation.updaters.mobject_update_utils import turn_animation_into_updater


def test_turn_animation_into_updater_zero_run_time():
    """Test that turn_animation_into_updater handles zero run_time correctly."""
    # Create a simple mobject and animation
    mobject = Circle()
    animation = FadeIn(mobject, run_time=0.0)

    # Track updater calls
    update_calls = []
    original_updaters = mobject.get_updaters().copy()

    # Call turn_animation_into_updater
    result = turn_animation_into_updater(animation)

    # Verify mobject is returned
    assert result is mobject

    # Get the updater that was added
    current_updaters = mobject.get_updaters()
    assert len(current_updaters) == len(original_updaters) + 1
    updater = current_updaters[-1]

    # Simulate calling the updater
    updater(mobject, dt=0.1)

    # The updater should have finished and removed itself
    current_updaters = mobject.get_updaters()
    assert len(current_updaters) == len(original_updaters)
    assert updater not in current_updaters


def test_turn_animation_into_updater_positive_run_time_persists():
    """Test that updater persists with positive run_time."""
    mobject = Circle()
    animation = FadeIn(mobject, run_time=1.0)

    original_updaters = mobject.get_updaters().copy()

    # Call turn_animation_into_updater
    result = turn_animation_into_updater(animation)

    # Get the updater that was added
    current_updaters = mobject.get_updaters()
    assert len(current_updaters) == len(original_updaters) + 1
    updater = current_updaters[-1]

    # Simulate calling the updater (partial progress)
    updater(mobject, dt=0.1)

    # The updater should still be present (not finished)
    current_updaters = mobject.get_updaters()
    assert len(current_updaters) == len(original_updaters) + 1
    assert updater in current_updaters


def test_always():
    d = Dot()
    circ = Circle()
    d.always.next_to(circ, UP)
    assert len(d.updaters) == 1
    # we should be able to chain updaters
    d2 = Dot()
    d.always.next_to(d2, UP).next_to(circ, UP)
    assert len(d.updaters) == 3
