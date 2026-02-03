from __future__ import annotations

from manim import Circle, FadeIn
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
