"""Utility functions for continuous animation of mobjects."""

__all__ = [
    "assert_is_mobject_method",
    "always",
    "f_always",
    "always_redraw",
    "always_shift",
    "always_rotate",
    "cycle_animation",
]


import inspect

import numpy as np

from ..constants import DEGREES, RIGHT
from ..mobject.mobject import Mobject


def assert_is_mobject_method(method):
    assert inspect.ismethod(method)
    mobject = method.__self__
    assert isinstance(mobject, Mobject)


def always(method, *args, **kwargs):
    assert_is_mobject_method(method)
    mobject = method.__self__
    func = method.__func__
    mobject.add_updater(lambda m: func(m, *args, **kwargs))
    return mobject


def f_always(method, *arg_generators, **kwargs):
    """
    More functional version of always, where instead
    of taking in args, it takes in functions which output
    the relevant arguments.
    """
    assert_is_mobject_method(method)
    mobject = method.__self__
    func = method.__func__

    def updater(mob):
        args = [arg_generator() for arg_generator in arg_generators]
        func(mob, *args, **kwargs)

    mobject.add_updater(updater)
    return mobject


def always_redraw(func):
    mob = func()
    mob.add_updater(lambda m: mob.become(func()))
    return mob


def always_shift(mobject, direction=RIGHT, rate=0.1):
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    mobject.add_updater(lambda m, dt: m.shift(dt * rate * normalize(direction)))
    return mobject


def always_rotate(mobject, rate=20 * DEGREES, **kwargs):
    mobject.add_updater(lambda m, dt: m.rotate(dt * rate, **kwargs))
    return mobject


def cycle_animation(animation, **kwargs):
    return turn_animation_into_updater(animation, cycle=True, **kwargs)
