"Tests whether the methods of Positionable behave the exact same as the Mobject methods."

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from manim.mobject.abstract.positionable import Positionable
from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.mobject.types.vectorized_mobject import VMobject
from manim.typing import Point3D, Point3D_Array, Vector3D

_RNG = np.random.default_rng()
POINT_COUNTS = list(range(1, 100 + 1, 1))
LOOPS_PER_POINT_COUNT: int = 100
UNTESTED = [
    name
    for name, attr in Positionable.__dict__.items()
    if not (name.startswith("__") or attr is getattr(Positionable.__base__, name, None))
]


def main() -> None:
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
    validate_getter(
        "get_boundary_point",
        lambda mob, kwargs: mob.get_boundary_point(**kwargs),
        lambda: {"direction": random_vector()},
    )
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
    validate_getter(
        "get_extremum_along_dim",
        lambda mob, kwargs: mob.get_extremum_along_dim(**kwargs),
        lambda: {
            "dim": random_choice([0, 1, 2]),
            "key": random_choice([0, 1, 2]),
        },
    )
    validate_getter("get_left", lambda mob, _: mob.get_left())
    validate_getter("get_nadir", lambda mob, _: mob.get_nadir())
    validate_getter("get_right", lambda mob, _: mob.get_right())
    validate_getter("get_top", lambda mob, _: mob.get_top())
    validate_getter("get_x", lambda mob, _: mob.get_x())
    validate_getter("get_y", lambda mob, _: mob.get_y())
    validate_getter("get_z", lambda mob, _: mob.get_z())
    validate_getter("get_zenith", lambda mob, _: mob.get_zenith())
    validate_getter("height", lambda mob, _: mob.height)
    validate_setter(
        "height",
        lambda mob, kwargs: setattr(mob, "height", kwargs["value"]),
        lambda: {
            "value": random_number(),
        },
    )
    validate_getter("is_off_screen", lambda mob, _: mob.is_off_screen())
    validate_getter(
        "length_over_dim",
        lambda mob, kwargs: mob.length_over_dim(**kwargs),
        lambda: {"dim": random_choice([0, 1, 2])},
    )
    validate_setter(
        "match_coord",
        lambda mob, kwargs: mob.match_coord(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
            "dim": random_choice([0, 1, 2]),
            "direction": optional(random_vector()),
        },
    )
    validate_setter(
        "match_depth",
        lambda mob, kwargs: mob.match_depth(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
            "stretch": optional(random_choice([True, False])),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "match_dim_size",
        lambda mob, kwargs: mob.match_dim_size(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
            "dim": random_choice([0, 1, 2]),
            "stretch": optional(random_choice([True, False])),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "match_height",
        lambda mob, kwargs: mob.match_height(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
            "stretch": optional(random_choice([True, False])),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "match_points",
        lambda mob, kwargs: mob.match_points(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
        },
    )
    validate_setter(
        "match_width",
        lambda mob, kwargs: mob.match_width(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
            "stretch": optional(random_choice([True, False])),
            "about_point": optional(random_point()),
            "about_edge": optional(random_vector()),
        },
    )
    validate_setter(
        "match_x",
        lambda mob, kwargs: mob.match_x(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
            "direction": optional(random_vector()),
        },
    )
    validate_setter(
        "match_y",
        lambda mob, kwargs: mob.match_y(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
            "direction": optional(random_vector()),
        },
    )
    validate_setter(
        "match_z",
        lambda mob, kwargs: mob.match_z(
            create_another(mob, kwargs.pop("points")), **kwargs
        ),
        lambda: {
            "points": random_points(size=int(random_number(1, 100))),
            "direction": optional(random_vector()),
        },
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
    # TODO: reduce_across_dimension
    validate_setter(
        "rescale_to_fit",
        lambda mob, kwargs: mob.rescale_to_fit(**kwargs),
        lambda: {
            "length": random_number(),
            "dim": random_choice([0, 1, 2]),
            "stretch": optional(random_choice([True, False])),
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
        "rotate_about_origin",
        lambda mob, kwargs: mob.rotate_about_origin(**kwargs),
        lambda: {
            "angle": random_number(),
            "axis": optional(random_vector()),
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
        "shift_onto_screen",
        lambda mob, kwargs: mob.shift_onto_screen(**kwargs),
        lambda: {
            "buff": random_number(),
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
        "stretch_about_point",
        lambda mob, kwargs: mob.stretch_about_point(**kwargs),
        lambda: {
            "factor": random_number(),
            "dim": random_choice([0, 1, 2]),
            "point": random_point(),
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

    validate_getter("width", lambda mob, _: mob.width)
    validate_setter(
        "width",
        lambda mob, kwargs: setattr(mob, "width", kwargs["value"]),
        lambda: {
            "value": random_number(),
        },
    )

    print("Untested")
    for name in UNTESTED:
        if hasattr(Mobject, name):
            print(f"\t{name}")


def validate_function(
    name: str,
    function: Callable[[Mobject | Positionable, dict[Any, Any]], Any],
    validate: Callable[[Mobject, Positionable, Any, Any], None],
    random_parameters: Callable[[], dict[Any, Any]],
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

            kwargs = random_parameters()
            kwargs = {key: value for key, value in kwargs.items() if value is not None}

            start = time.perf_counter_ns()
            result_old = function(mob_old, kwargs.copy())
            time_old += time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            result_new = function(mob_new, kwargs.copy())
            time_new += time.perf_counter_ns() - start

            try:
                validate(mob_old, mob_new, result_old, result_new)
            except AssertionError as e:
                raise ValueError(
                    f"""
                    Point Count: {point_count}
                    Kwargs: {kwargs}
                    Points: {points}
                    Old Result: {result_old}
                    New Result: {result_new}
                    Old Points: {mob_old.points}
                    New Points: {mob_new.points}
                    """.replace("                    ", "")
                ) from e

    print(
        f"\t{name.ljust(25)}\t{time_old / time_new:1.2f}x\t{time_old / 1e9:.2f}s\t{time_new / 1e9:.2f}s"
    )
    if name in UNTESTED:
        UNTESTED.remove(name)


def validate_setter(
    name: str,
    function: Callable[[Mobject | Positionable, dict[Any, Any]], Any],
    create_kwargs: Callable[[], Any] = lambda: {},
):
    def validate(
        mob_old: Mobject,
        mob_new: Positionable,
        result_old: Any,
        result_new: Any,
    ) -> None:
        assert (result_old is None) == (result_new is None)
        assert np.allclose(mob_old.points, mob_new.points)

    validate_function(
        name=name,
        function=function,
        validate=validate,
        random_parameters=create_kwargs,
    )


def validate_getter(
    name: str,
    function: Callable[[Mobject | Positionable, dict[Any, Any]], Any],
    create_kwargs: Callable[[], Any] = lambda: {},
):
    def validate(
        mob_old: Mobject,
        mob_new: Positionable,
        result_old: Any,
        result_new: Any,
    ) -> None:
        assert np.allclose(result_old, result_new)

    validate_function(
        name=name,
        function=function,
        validate=validate,
        random_parameters=create_kwargs,
    )


def optional(value: Any, a: float = 0.9) -> Any | None:
    return value if _RNG.uniform() < a else None


def random_number(low: float = -10, high: float = 10) -> float:
    return _RNG.uniform(low=low, high=high)


def random_point(low: float = -25, high: float = 25) -> Point3D:
    return _RNG.uniform(low=low, high=high, size=3)


def random_points(low: float = -25, high: float = 25, size: int = 1) -> np.ndarray:
    return _RNG.uniform(low=low, high=high, size=(size, 3))


def random_vector(low: float = -3, high: float = 3) -> Vector3D:
    v = _RNG.uniform(low=low, high=high, size=3)
    if random_number(0, 1) <= 0.5:
        v = np.round(v)
    return v


def random_choice(a: list[Any]) -> Any:
    return _RNG.choice(a=a)


def create_another(
    mob: Mobject | Positionable,
    points: Point3D_Array,
) -> Mobject | Positionable:
    another = type(mob)()
    another.points = points
    return another


def dump_attributes() -> None:
    seen: set[str] = set()

    for cls in [Mobject, VMobject, OpenGLMobject, OpenGLVMobject]:
        assert isinstance(cls, type)
        print(cls.__name__)
        for name, attr in sorted(cls.__dict__.items()):
            if (
                name in seen
                or name.startswith("__")
                or attr is getattr(cls.__base__, name, None)
            ):
                continue
            print(
                f"\t{'-+'[getattr(Positionable, name, None) is not getattr(Positionable.__base__, name, None)]} {name}"
            )
        seen |= cls.__dict__.keys()

    print(Positionable.__name__)
    for name, attr in Positionable.__dict__.items():
        if name.startswith("__") or attr is getattr(Positionable.__base__, name, None):
            continue
        print(f"\t* {name}", "(new)" if name not in seen else "")


if __name__ == "__main__":
    main()
