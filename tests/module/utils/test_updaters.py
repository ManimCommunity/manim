from __future__ import annotations

import pytest

from manim.constants import RIGHT
from manim.mobject.geometry.polygram import Square
from manim.mobject.graph import Graph
from manim.utils.updaters import MobjectUpdaterWrapper


def test_UpdaterWrapper() -> None:
    square = Square().move_to(RIGHT)

    # Non-time-based updater: 1 parameter with no default value
    wrapper = MobjectUpdaterWrapper(lambda mob: mob.next_to(square))
    assert not wrapper.is_time_based

    # Time-based updater: 2 parameters with no default value
    wrapper = MobjectUpdaterWrapper(lambda mob, dt: mob.rotate(dt))
    assert wrapper.is_time_based

    # It's not necessary for the 2nd parameter to be called dt
    wrapper = MobjectUpdaterWrapper(lambda mob, delta_time: mob.rotate(delta_time))
    assert wrapper.is_time_based

    # An updater can even have more than 2 parameters, as long as they have
    # default values
    wrapper = MobjectUpdaterWrapper(lambda mob, dt, rate=2: mob.rotate(rate * dt))
    assert wrapper.is_time_based

    # An updater cannot have no parameters
    with pytest.raises(ValueError):
        wrapper = MobjectUpdaterWrapper(lambda: square.move_to(RIGHT))

    # An updater cannot have more than 2 parameters without a default value
    with pytest.raises(ValueError):
        wrapper = MobjectUpdaterWrapper(lambda mob, rate, third: mob.rotate(rate))

    # Only parameters with no default value are considered when determining
    # whether the updater is time-based or not. If an updater has 2 parameters,
    # but the 2nd one has a default value, it's considered non-time-based.
    wrapper = MobjectUpdaterWrapper(lambda mob, other=square: mob.next_to(other))
    assert not wrapper.is_time_based

    # When using an instance method, the first argument is ignored if it's
    # called 'self'. This is an attempt to exclude static methods from this
    # rule.
    graph = Graph([1, 2], [(1, 2)])
    wrapper = MobjectUpdaterWrapper(graph.update_edges)  # signature: (self, graph)
    assert not wrapper.is_time_based  # since only the 'graph' param is counted

    # This doesn't happen when calling the method from the class rather than
    # from an instance.
    wrapper = MobjectUpdaterWrapper(Graph.update_edges)  # signature: (self, graph)
    # 'self' is the 1st, 'graph' is the 2nd, (incorrectly) considered as time
    assert wrapper.is_time_based

    # In general, if the function is not an instance method, the 1st parameter
    # is almost always included, even if it's called "self".
    wrapper = MobjectUpdaterWrapper(lambda self: self.move_to(square))
    assert not wrapper.is_time_based
    wrapper = MobjectUpdaterWrapper(lambda self, dt: self.rotate(dt))
    assert wrapper.is_time_based

    # The only exception is if it's called "cls". Don't call it "cls" if it's
    # not a class method.
    wrapper = MobjectUpdaterWrapper(lambda cls, dt: cls.rotate(dt))
    assert (
        not wrapper.is_time_based
    )  # Only 1 parameter, dt, is considered, and it's used as a Mobject, not float
    with pytest.raises(ValueError):
        # Since cls is excluded, there are no other parameters
        wrapper = MobjectUpdaterWrapper(lambda cls: cls.next_to(square))
