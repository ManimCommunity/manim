import time
from collections.abc import Callable
from logging import getLogger
from typing import Any

import numpy as np

from manim.mobject.abstract.positionable import Positionable
from manim.mobject.mobject import Mobject
from manim.typing import Point3D, Point3D_Array, Vector3D

_RNG = np.random.default_rng()
POINT_COUNTS = list(range(1, 101))
LOOPS_PER_POINT_COUNT: int = 10
UNTESTED = [
    name
    for name, attr in Positionable.__dict__.items()
    if not (name.startswith("__") or attr is getattr(Positionable.__base__, name, None))
]


def optional(value: Any, a: float = 0.9) -> Any | None:
    return value if _RNG.uniform() < a else None


def random_number(low: float = -10, high: float = 10) -> float:
    return _RNG.uniform(low=low, high=high)


def random_point(low: float = -10, high: float = 10) -> Point3D:
    return _RNG.uniform(low=low, high=high, size=3)


def random_points(low: float = -10, high: float = 10, size: int = 1) -> np.ndarray:
    return _RNG.uniform(low=low, high=high, size=(size, 3))


def random_vector(low: float = -3, high: float = 3) -> Vector3D:
    dtype = random_choice([int, float])
    return _RNG.uniform(low=low, high=high, size=3).astype(dtype=dtype)


def random_choice(a: list[Any]) -> Any:
    return _RNG.choice(a=a)


def create_another(
    mob: Mobject | Positionable,
    points: Point3D_Array,
) -> Mobject | Positionable:
    another = type(mob)()
    another.points = points
    return another


def validate_function(
    name: str,
    function: Callable[[Mobject | Positionable, dict[Any, Any]], Any],
    validate: Callable[[Any, Any], None],
    create_kwargs: Callable[[], dict[Any, Any]],
) -> None:
    global POINT_COUNTS, LOOPS_PER_POINT_COUNT
    time_old, time_new = 0, 0

    for point_count in POINT_COUNTS:
        for _ in range(LOOPS_PER_POINT_COUNT):
            points = random_points(size=point_count)

            mob_old = Mobject()
            mob_old.points = points.copy()
            mob_new = Positionable()
            mob_new.points = points.copy()

            kwargs = create_kwargs()
            kwargs = {key: value for key, value in kwargs.items() if value is not None}

            start = time.perf_counter_ns()
            result_old = function(mob_old, kwargs)
            time_old += time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            result_new = function(mob_new, kwargs)
            time_new += time.perf_counter_ns() - start

            try:
                validate(result_old, result_new)
            except AssertionError as e:
                raise ValueError(
                    f"\nPoint Count: {point_count}\nKwargs: {kwargs}\nPoints: {points}\nOld Result: {result_old}\nNew Result: {result_new}"
                ) from e  # noqa: B904

    print(f"\t{name.ljust(25)}\t{time_old / time_new:1.2f}x".ljust(6))
    UNTESTED.remove(name)


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
    getLogger("manim").addFilter(lambda x: "deprecated" not in x.getMessage())

    validate_setter(
        "align_on_border",
        lambda mob, kwargs: mob.align_on_border(**kwargs),
        lambda: {
            "direction": random_vector(),
            "buff": optional(random_number()),
        },
    )
    validate_setter(
        "align_to",
        lambda mob, kwargs: mob.align_to(**kwargs),
        lambda: {
            "mobject_or_point": random_point(),
            "direction": optional(random_vector()),
        },
    )
    # TODO: apply_complex_function
    # TODO: apply_function
    # TODO: apply_function_to_position
    # TODO: apply_matrix
    # TODO: apply_points_function_about_point
    validate_setter("center", lambda mob, _: mob.center())
    validate_getter("depth", lambda mob, _: mob.depth)
    validate_setter(
        "depth",
        lambda mob, kwargs: setattr(mob, "depth", kwargs["value"]),
        lambda: {"value": random_number()},
    )
    validate_setter(
        "flip",
        lambda mob, kwargs: mob.flip(**kwargs),
        lambda: {
            "axis": optional(random_vector()),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_getter("get_bottom", lambda mob, _: mob.get_bottom())
    validate_getter("get_center", lambda mob, _: mob.get_center())
    validate_getter("get_center_of_mass", lambda mob, _: mob.get_center_of_mass())
    validate_getter(
        "get_coord",
        lambda mob, kwargs: mob.get_coord(**kwargs),
        lambda: {
            "dim": random_choice([0, 1, 2]),
            "direction": optional(random_vector()),
        },
    )
    validate_getter(
        "get_corner",
        lambda mob, kwargs: mob.get_corner(**kwargs),
        lambda: {"direction": random_vector()},
    )
    validate_getter(
        "get_critical_point",
        lambda mob, kwargs: mob.get_critical_point(**kwargs),
        lambda: {"direction": random_vector()},
    )
    validate_getter(
        "get_edge_center",
        lambda mob, kwargs: mob.get_edge_center(**kwargs),
        lambda: {"direction": random_vector()},
    )
    validate_getter("get_left", lambda mob, _: mob.get_left())
    validate_getter("get_nadir", lambda mob, _: mob.get_nadir())
    validate_getter("get_right", lambda mob, _: mob.get_right())
    validate_getter("get_top", lambda mob, _: mob.get_top())
    validate_getter("get_x", lambda mob, _: mob.get_x())
    validate_getter("get_y", lambda mob, _: mob.get_y())
    validate_getter("get_z", lambda mob, _: mob.get_z())
    validate_getter("get_zenith", lambda mob, _: mob.get_zenith())
    validate_getter(
        "length_over_dim",
        lambda mob, kwargs: mob.length_over_dim(**kwargs),
        lambda: {"dim": random_choice([0, 1, 2])},
    )
    validate_setter(
        "move_to",
        lambda mob, kwargs: mob.move_to(**kwargs),
        lambda: {
            "point_or_mobject": random_point(),
            "aligned_edge": optional(random_vector()),
            "coor_mask": optional(random_vector()),
        },
    )
    validate_setter(
        "next_to",
        lambda mob, kwargs: mob.next_to(**kwargs),
        lambda: {
            "mobject_or_point": random_point(),
            "direction": optional(random_vector()),
            "buff": optional(random_number()),
            "aligned_edge": optional(random_vector()),
            "coor_mask": optional(random_vector()),
        },
    )
    validate_setter(
        "pose_at_angle",
        lambda mob, kwargs: mob.pose_at_angle(**kwargs),
        lambda: {
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "rotate",
        lambda mob, kwargs: mob.rotate(**kwargs),
        lambda: {
            "angle": random_number(),
            "axis": optional(random_vector()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "scale",
        lambda mob, kwargs: mob.scale(**kwargs),
        lambda: {
            "scale_factor": random_number(),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "scale_to_fit_depth",
        lambda mob, kwargs: mob.scale_to_fit_depth(**kwargs),
        lambda: {
            "depth": random_number(),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "scale_to_fit_height",
        lambda mob, kwargs: mob.scale_to_fit_height(**kwargs),
        lambda: {
            "height": random_number(),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "scale_to_fit_width",
        lambda mob, kwargs: mob.scale_to_fit_width(**kwargs),
        lambda: {
            "width": random_number(),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "set_coord",
        lambda mob, kwargs: mob.set_coord(**kwargs),
        lambda: {
            "value": random_number(),
            "dim": random_choice([0, 1, 2]),
            "direction": optional(random_vector()),
        },
    )
    validate_setter(
        "set_x",
        lambda mob, kwargs: mob.set_x(**kwargs),
        lambda: {
            "x": random_number(),
            "direction": optional(random_vector()),
        },
    )
    validate_setter(
        "set_y",
        lambda mob, kwargs: mob.set_y(**kwargs),
        lambda: {
            "y": random_number(),
            "direction": optional(random_vector()),
        },
    )
    validate_setter(
        "set_z",
        lambda mob, kwargs: mob.set_z(**kwargs),
        lambda: {
            "z": random_number(),
            "direction": optional(random_vector()),
        },
    )
    validate_setter(
        "shift",
        lambda mob, kwargs: mob.shift(kwargs["value"]),
        lambda: {
            "value": random_vector(),
        },
    )
    validate_setter(
        "stretch",
        lambda mob, kwargs: mob.stretch(**kwargs),
        lambda: {
            "factor": random_number(),
            "dim": random_choice([0, 1, 2]),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "stretch_to_fit_depth",
        lambda mob, kwargs: mob.stretch_to_fit_depth(**kwargs),
        lambda: {
            "depth": random_number(),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "stretch_to_fit_height",
        lambda mob, kwargs: mob.stretch_to_fit_height(**kwargs),
        lambda: {
            "height": random_number(),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "stretch_to_fit_width",
        lambda mob, kwargs: mob.stretch_to_fit_width(**kwargs),
        lambda: {
            "width": random_number(),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "to_corner",
        lambda mob, kwargs: mob.to_corner(**kwargs),
        lambda: {
            "corner": random_vector(),
            "buff": random_number(),
        },
    )
    validate_setter(
        "to_edge",
        lambda mob, kwargs: mob.to_edge(**kwargs),
        lambda: {
            "edge": optional(random_vector()),
            "buff": optional(random_number()),
        },
    )

    print("Untested")
    for name in UNTESTED:
        print(f"\t{name}")


if __name__ == "__main__":
    main()
