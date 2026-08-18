import time
from collections.abc import Callable
from typing import Any

import numpy as np

from manim.mobject.abstract.positionable import Positionable
from manim.mobject.mobject import Mobject
from manim.typing import Point3D, Vector3D

_RNG = np.random.default_rng(seed=1)


def _random_number(low: float = -10, high: float = 10) -> float:
    return _RNG.uniform(low=low, high=high)


def _random_point(low: float = -10, high: float = 10) -> Point3D:
    return _RNG.uniform(low=low, high=high, size=3)


def _random_points(low: float = -10, high: float = 10, size: int = 1) -> np.ndarray:
    return _RNG.uniform(low=low, high=high, size=(size, 3))


def _random_vector(low: float = -3, high: float = 3) -> Vector3D:
    return _RNG.uniform(low=low, high=high, size=3)


def _random_choice(a: list[Any]) -> Any:
    return _RNG.choice(a=a)


# def _create_another(
#    mob: Mobject | Positionable, points: Point3D_Array
# ) -> Mobject | Positionable:
#    another = type(mob)()
#    another.points = points
#    return another


def validate_function(
    name: str,
    function: Callable[[Mobject | Positionable, dict[Any, Any]], Any],
    validate: Callable[[Any, Any], None],
    create_kwargs: Callable[[], dict[Any, Any]],
    *,
    point_counts: list[int] = list(range(1, 101)),
    loop_count: int = 100,
) -> None:
    time_old, time_new = 0, 0

    for point_count in point_counts:
        for _ in range(loop_count):
            points = _random_points(size=point_count)

            mob_old = Mobject()
            mob_old.points = points.copy()
            mob_new = Positionable()
            mob_new.points = points.copy()

            kwargs = create_kwargs()

            start = time.perf_counter_ns()
            result_old = function(mob_old, kwargs)
            time_old += time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            result_new = function(mob_new, kwargs)
            time_new += time.perf_counter_ns() - start

            validate(result_old, result_new)

    print(name.ljust(25), f"{time_old / time_new:1.2f}x".ljust(6))


def validate_setter(
    name: str,
    function: Callable[[Mobject | Positionable, dict[Any, Any]], Any],
    create_kwargs: Callable[[], Any] = lambda: {},
):
    def validate(result_1: Any, result_2: Any) -> None:
        assert isinstance(result_1, Positionable | Mobject)
        assert isinstance(result_2, Positionable | Mobject)
        assert np.allclose(result_1.points, result_2.points)

    validate_function(
        name=name,
        function=function,
        validate=validate,
        create_kwargs=create_kwargs,
    )


def validate_getter(
    name: str,
    function: Callable[[Mobject | Positionable, dict[Any, Any]], Any],
    create_kwargs: Callable[[], Any] = lambda: {},
):
    def validate(result_1: Any, result_2: Any) -> None:
        assert np.allclose(result_1, result_2)

    validate_function(
        name=name,
        function=function,
        validate=validate,
        create_kwargs=create_kwargs,
    )


def main() -> None:
    validate_setter(
        "align_on_border",
        lambda mob, kwargs: mob.align_on_border(**kwargs),
        lambda: {"direction": _random_vector(), "buff": _random_number()},
    )
    validate_setter(
        "align_to",
        lambda mob, kwargs: mob.align_to(**kwargs),
        lambda: {"mobject_or_point": _random_point(), "direction": _random_vector()},
    )
    validate_setter("center", lambda mob, _: mob.center())
    validate_setter(
        "flip",
        lambda mob, kwargs: mob.flip(**kwargs),
        lambda: {
            "axis": _random_vector(),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_getter("get_bottom", lambda mob, _: mob.get_bottom())
    validate_getter("get_center", lambda mob, _: mob.get_center())
    validate_getter("get_center_of_mass", lambda mob, _: mob.get_center_of_mass())
    validate_getter(
        "get_coord",
        lambda mob, kwargs: mob.get_coord(**kwargs),
        lambda: {"dim": _random_choice([0, 1, 2]), "direction": _random_vector()},
    )
    validate_getter(
        "get_corner",
        lambda mob, kwargs: mob.get_corner(**kwargs),
        lambda: {"direction": _random_vector()},
    )
    validate_getter(
        "get_critical_point",
        lambda mob, kwargs: mob.get_critical_point(**kwargs),
        lambda: {"direction": _random_vector()},
    )
    validate_getter(
        "get_edge_center",
        lambda mob, kwargs: mob.get_edge_center(**kwargs),
        lambda: {"direction": _random_vector()},
    )
    validate_getter("get_end", lambda mob, _: mob.get_end())
    validate_getter("get_left", lambda mob, _: mob.get_left())
    validate_getter("get_nadir", lambda mob, _: mob.get_nadir())
    validate_getter("get_right", lambda mob, _: mob.get_right())
    validate_getter("get_start", lambda mob, _: mob.get_start())
    validate_getter("get_start_and_end", lambda mob, _: mob.get_start_and_end())
    validate_getter("get_top", lambda mob, _: mob.get_top())
    validate_getter("get_x", lambda mob, _: mob.get_x())
    validate_getter("get_y", lambda mob, _: mob.get_y())
    validate_getter("get_z", lambda mob, _: mob.get_z())
    validate_getter("get_zenith", lambda mob, _: mob.get_zenith())
    validate_getter(
        "length_over_dim",
        lambda mob, kwargs: mob.length_over_dim(**kwargs),
        lambda: {"dim": _random_choice([0, 1, 2])},
    )
    validate_setter(
        "move_to",
        lambda mob, kwargs: mob.move_to(**kwargs),
        lambda: {
            "point_or_mobject": _random_point(),
            "aligned_edge": _random_vector(),
            "coor_mask": _random_vector(),
        },
    )
    validate_setter(
        "next_to",
        lambda mob, kwargs: mob.next_to(**kwargs),
        lambda: {
            "mobject_or_point": _random_point(),
            "direction": _random_vector(),
            "buff": _random_number(),
            "aligned_edge": _random_vector(),
            "coor_mask": _random_vector(),
        },
    )
    validate_setter(
        "pose_at_angle",
        lambda mob, kwargs: mob.pose_at_angle(**kwargs),
        lambda: {"about_point": _random_point(), "about_edge": _random_vector()},
    )
    validate_setter(
        "rotate",
        lambda mob, kwargs: mob.rotate(**kwargs),
        lambda: {
            "angle": _random_number(),
            "axis": _random_vector(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "scale",
        lambda mob, kwargs: mob.scale(**kwargs),
        lambda: {
            "scale_factor": _random_number(),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "scale_to_fit_depth",
        lambda mob, kwargs: mob.scale_to_fit_depth(**kwargs),
        lambda: {
            "depth": _random_number(),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "scale_to_fit_height",
        lambda mob, kwargs: mob.scale_to_fit_height(**kwargs),
        lambda: {
            "height": _random_number(),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "scale_to_fit_width",
        lambda mob, kwargs: mob.scale_to_fit_width(**kwargs),
        lambda: {
            "width": _random_number(),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "set_coord",
        lambda mob, kwargs: mob.set_coord(**kwargs),
        lambda: {
            "value": _random_number(),
            "dim": _random_choice([0, 1, 2]),
            "direction": _random_vector(),
        },
    )
    validate_setter(
        "set_x",
        lambda mob, kwargs: mob.set_x(**kwargs),
        lambda: {"x": _random_number(), "direction": _random_vector()},
    )
    validate_setter(
        "set_y",
        lambda mob, kwargs: mob.set_y(**kwargs),
        lambda: {"y": _random_number(), "direction": _random_vector()},
    )
    validate_setter(
        "set_z",
        lambda mob, kwargs: mob.set_z(**kwargs),
        lambda: {"z": _random_number(), "direction": _random_vector()},
    )
    validate_setter(
        "shift",
        lambda mob, kwargs: mob.shift(kwargs["value"]),
        lambda: {"value": _random_vector()},
    )
    validate_setter(
        "stretch",
        lambda mob, kwargs: mob.stretch(**kwargs),
        lambda: {
            "factor": _random_number(),
            "dim": _random_choice([0, 1, 2]),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "stretch_to_fit_depth",
        lambda mob, kwargs: mob.stretch_to_fit_depth(**kwargs),
        lambda: {
            "depth": _random_number(),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "stretch_to_fit_height",
        lambda mob, kwargs: mob.stretch_to_fit_height(**kwargs),
        lambda: {
            "height": _random_number(),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "stretch_to_fit_width",
        lambda mob, kwargs: mob.stretch_to_fit_width(**kwargs),
        lambda: {
            "width": _random_number(),
            "about_point": _random_point(),
            "about_edge": _random_vector(),
        },
    )
    validate_setter(
        "to_corner",
        lambda mob, kwargs: mob.to_corner(**kwargs),
        lambda: {"corner": _random_vector(), "buff": _random_number()},
    )
    validate_setter(
        "to_edge",
        lambda mob, kwargs: mob.to_edge(**kwargs),
        lambda: {"edge": _random_vector(), "buff": _random_number()},
    )


if __name__ == "__main__":
    main()
