from __future__ import annotations

import random

import numpy as np
import pytest

from manim import BLUE, RIGHT, StreamLines
from manim.utils.hashing import get_hash_from_play_call


@pytest.fixture
def preserve_random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002 -- test the legacy global generator
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)  # noqa: NPY002 -- restore the caller's global state


def make_stream_lines():
    return StreamLines(
        lambda point: RIGHT,
        color=BLUE,
        x_range=[-1, 1, 1],
        y_range=[-1, 1, 1],
        virtual_time=0.2,
        dt=0.05,
    )


@pytest.mark.usefixtures("preserve_random_state")
@pytest.mark.parametrize("warm_up", [False, True])
def test_flow_phases_are_independent_of_global_random_state(warm_up):
    phases = []
    for seed in (1, 42):
        random.seed(seed)
        lines = make_stream_lines()
        lines.start_animation(warm_up=warm_up)
        phases.append([line.time for line in lines.stream_lines])

    np.testing.assert_array_equal(phases[0], phases[1])
    assert len(set(phases[0])) > 1
    if warm_up:
        assert all(-0.2 <= phase <= 0 for phase in phases[0])
    else:
        assert all(0 <= phase <= 0.2 for phase in phases[0])


@pytest.mark.usefixtures("preserve_random_state")
def test_creation_order_is_independent_of_global_random_state():
    orders = []
    for seed in (1, 42):
        random.seed(seed)
        lines = make_stream_lines()
        indices = {id(line): index for index, line in enumerate(lines.stream_lines)}
        animation = lines.create()
        orders.append([indices[id(child.mobject)] for child in animation.animations])

    assert orders[0] == orders[1]
    assert sorted(orders[0]) == list(range(len(orders[0])))


@pytest.mark.usefixtures("preserve_random_state")
def test_stream_lines_do_not_advance_global_random_generators():
    random.seed(42)
    np.random.seed(42)  # noqa: NPY002 -- deliberately exercise the global generator
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002 -- inspect the global generator

    lines = make_stream_lines()
    lines.create()
    lines.start_animation()

    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()  # noqa: NPY002 -- inspect the global generator
    assert current_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]


@pytest.mark.usefixtures("preserve_random_state")
@pytest.mark.parametrize("operation", ["create", "flow"])
def test_identical_stream_line_animations_have_identical_cache_keys(operation):
    keys = []
    for seed in (1, 42):
        random.seed(seed)
        lines = make_stream_lines()
        if operation == "create":
            animations = [lines.create()]
        else:
            lines.start_animation(warm_up=False)
            animations = []
        keys.append(
            get_hash_from_play_call(
                None,
                None,
                animations,
                [lines],
                backend="cairo",
                encoder_fingerprint="test-encoder",
                renderer_state={},
            )
        )

    assert keys[0] == keys[1]
