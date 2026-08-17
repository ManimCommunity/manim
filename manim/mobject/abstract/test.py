from collections.abc import Callable
import contextlib
import time
from typing import Any

import numpy as np

from manim.mobject.abstract.positionable import Positionable
from manim.mobject.mobject import Mobject
from manim.typing import Point3D, Point3D_Array, Vector3D


_RNG = np.random.default_rng(seed=1)


def random_number(low: float = -10, high: float = 10) -> float:
    return _RNG.uniform(low=low, high=high)


def random_point(low: float = -10, high: float = 10) -> Point3D:
    return _RNG.uniform(low=low, high=high, size=3)


def random_points(low: float = -10, high: float = 10, size: int = 1) -> np.ndarray:
    return _RNG.uniform(low=low, high=high, size=(size, 3))


def random_vector(low: float = -3, high: float = 3) -> Vector3D:
    return _RNG.uniform(low=low, high=high, size=3)


def create_another[T: Mobject | Positionable](mob: T, points: Point3D_Array) -> T:
    another = mob.__class__()
    another.points = points
    return another


def create_mobs(point_count: int) -> tuple[Mobject, Positionable]:
    points = random_points(size=point_count)
    mob_1 = Mobject()
    mob_2 = Positionable()
    mob_1.points = points.copy()
    mob_2.points = points.copy()
    return mob_1, mob_2


def validate_function(
    name: str,
    function: Callable[[Mobject | Positionable, dict], Any],
    validate: Callable[[Any, Any], None],
    create_kwargs: Callable[[], dict],
    point_counts: list[int] = list(range(1, 101)),
    loop_count: int = 100,
) -> None:
    point_counts = point_counts.copy()
    _RNG.shuffle(point_counts)

    time_0, time_1 = 0, 0

    for point_count in point_counts:
        for _ in range(loop_count):
            mob_1, mob_2 = create_mobs(point_count=point_count)
            kwargs = create_kwargs()

            result_0: np.typing.ArrayLike | None = None
            with contextlib.suppress(Exception):
                start = time.perf_counter_ns()
                result_0 = function(mob_1, kwargs)
                time_0 += time.perf_counter_ns() - start

            if result_0 is not None:
                start = time.perf_counter_ns()
                result_1 = function(mob_2, kwargs)
                time_1 += time.perf_counter_ns() - start

                validate(result_0, result_1)

    print(name, f"{time_0 / time_1:2.4f}")


def validate_setter(
    name: str,
    function: Callable[[Mobject | Positionable, dict], Any],
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
    function: Callable[[Mobject | Positionable, dict], Any],
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
    validate_setter("align_on_border", lambda mob, kwargs: mob.align_on_border(**kwargs), lambda: {"direction": random_vector(), "buff": random_number()})
    validate_setter("align_to", lambda mob, kwargs: mob.align_to(**kwargs), lambda: {"mobject_or_point": random_point(), "direction": random_vector()})
    # validate_setter("apply_complex_function", lambda mob: mob.apply_complex_function(...))
    # validate_setter("apply_function", lambda mob: mob.apply_function(...))
    # validate_setter("apply_function_to_position", lambda mob: mob.apply_function_to_position(...))
    # validate_setter("apply_matrix", lambda mob: mob.apply_matrix(...))
    # validate_setter("apply_points_function_about_point", lambda mob: mob.apply_points_function_about_point(...))
    validate_setter("center", lambda mob, _: mob.center())
    validate_getter("depth", lambda mob, _: mob.depth)
    # validate_setter("depth", lambda mob, v=random_number(): setattr(mob, "depth", v))
    validate_setter("flip", lambda mob, kwargs: mob.flip(**kwargs), lambda: {"axis": random_vector(), "about_point": random_point(), "about_edge": random_vector()})
    validate_getter("get_bottom", lambda mob, _: mob.get_bottom())
    validate_getter("get_boundary_point", lambda mob, kwargs: mob.get_boundary_point(**kwargs), lambda: {"direction": random_vector()})
    validate_getter("get_center", lambda mob, _: mob.get_center())
    validate_getter("get_center_of_mass", lambda mob, _: mob.get_center_of_mass())
    for dim in [0, 1, 2]:
        validate_getter("get_coord", lambda mob, kwargs: mob.get_coord(**kwargs), lambda: {"dim": dim, "direction": random_vector()})
    validate_getter("get_corner", lambda mob, kwargs: mob.get_corner(**kwargs), lambda: {"direction": random_vector()})
    validate_getter("get_critical_point", lambda mob, kwargs: mob.get_critical_point(**kwargs), lambda: {"direction": random_vector()})
    validate_getter("get_edge_center", lambda mob, kwargs: mob.get_edge_center(**kwargs), lambda: {"direction": random_vector()})
    validate_getter("get_end", lambda mob, _: mob.get_end())
    for dim in [0, 1, 2]:
        for key in [0, 1, 2]:
            validate_getter("get_extremum_along_dim", lambda mob, kwargs: mob.get_extremum_along_dim(**kwargs), lambda: {"dim": dim, "key": key})
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
    validate_getter("height", lambda mob, _: mob.height)
    # validate_setter("height", lambda mob, h=random_number(): setattr(mob, "height", h))
    for dim in [0, 1, 2]:
        validate_getter("length_over_dim", lambda mob, _: mob.length_over_dim(dim=dim))

    for dim in [0, 1, 2]:
        validate_setter("match_coord", lambda mob, kwargs: mob.match_coord(mobject=create_another(mob=mob, **kwargs), dim=dim), lambda: {"points": random_points(size=int(random_number(1, 100)))})
    # validate_setter("match_depth", lambda mob: mob.match_depth())
    # validate_setter("match_dim_size", lambda mob: mob.match_dim_size())
    # validate_setter("match_height", lambda mob: mob.match_height())
    validate_setter("match_points", lambda mob, kwargs: mob.match_points(mobject=create_another(mob=mob, **kwargs)), lambda: {"points": random_points(size=int(random_number(1, 100)))})
    # validate_setter("match_width", lambda mob: mob.match_width())
    validate_setter("match_x", lambda mob, kwargs: mob.match_x(mobject=create_another(mob=mob, **kwargs)), lambda: {"points": random_points(size=int(random_number(1, 100)))})
    validate_setter("match_y", lambda mob, kwargs: mob.match_y(mobject=create_another(mob=mob, **kwargs)), lambda: {"points": random_points(size=int(random_number(1, 100)))})
    validate_setter("match_z", lambda mob, kwargs: mob.match_z(mobject=create_another(mob=mob, **kwargs)), lambda: {"points": random_points(size=int(random_number(1, 100)))})
    validate_setter("move_to", lambda mob, kwargs: mob.move_to(**kwargs), lambda: {"point_or_mobject": random_point(), "aligned_edge": random_vector(), "coor_mask": random_vector()})
    validate_setter(
        "next_to",
        lambda mob, kwargs: mob.next_to(**kwargs),
        lambda: {"mobject_or_point": random_point(), "direction": random_vector(), "buff": random_number(), "aligned_edge": random_vector(), "coor_mask": random_vector()},
    )
    # validate_setter("pose_at_angle", lambda mob: mob.pose_at_angle())
    # validate_setter("put_start_and_end_on", lambda mob: mob.put_start_and_end_on())
    # validate_getter("reduce_across_dimension", lambda mob: mob.reduce_across_dimension())
    # validate_setter("rescale_to_fit", lambda mob: mob.rescale_to_fit())
    validate_setter("rotate", lambda mob, kwargs: mob.rotate(**kwargs), lambda: {"angle": random_number(), "axis": random_vector(), "about_edge": random_vector()})
    # validate_setter("rotate_about_origin", lambda mob, a=random_number(), ax=random_vector(),: mob.rotate_about_origin(angle=a, axis=ax))
    # validate_setter("scale", lambda mob: mob.scale())
    # validate_setter("scale_to_fit_depth", lambda mob: mob.scale_to_fit_depth())
    # validate_setter("scale_to_fit_height", lambda mob: mob.scale_to_fit_height())
    # validate_setter("scale_to_fit_width", lambda mob: mob.scale_to_fit_width())
    for dim in [0, 1, 2]:
        validate_setter("set_coord", lambda mob, kwargs: mob.set_coord(**kwargs), lambda: {"value": random_number(), "dim": dim, "direction": random_vector()})
    validate_setter("set_x", lambda mob, kwargs: mob.set_x(**kwargs), lambda: {"x": random_number(), "direction": random_vector()})
    validate_setter("set_y", lambda mob, kwargs: mob.set_y(**kwargs), lambda: {"y": random_number(), "direction": random_vector()})
    validate_setter("set_z", lambda mob, kwargs: mob.set_z(**kwargs), lambda: {"z": random_number(), "direction": random_vector()})
    validate_setter("shift", lambda mob, kwargs: mob.shift(kwargs["value"]), lambda: {"value": random_vector()})
    # validate_setter("shift_onto_screen", lambda mob, v=random_vector(): mob.shift_onto_screen())
    for dim in [0, 1, 2]:
        validate_setter("stretch", lambda mob, kwargs: mob.stretch(**kwargs), lambda: {"factor": random_number(), "dim": dim, "about_point": random_point(), "about_edge": random_vector()})
    # validate_setter("stretch_about_point", lambda mob: mob.stretch_about_point())
    # validate_setter("stretch_to_fit_depth", lambda mob: mob.stretch_to_fit_depth())
    # validate_setter("stretch_to_fit_height", lambda mob: mob.stretch_to_fit_height())
    # validate_setter("stretch_to_fit_width", lambda mob: mob.stretch_to_fit_width())
    validate_setter("to_corner", lambda mob, kwargs: mob.to_corner(**kwargs), lambda: {"corner": random_vector(), "buff": random_number()})
    validate_setter("to_edge", lambda mob, kwargs: mob.to_edge(**kwargs), lambda: {"edge": random_vector(), "buff": random_number()})
    validate_getter("width", lambda mob, _: mob.width)
    # validate_setter("width", lambda mob, w=random_number(): setattr(mob, "width", w))


if __name__ == "__main__":
    main()
