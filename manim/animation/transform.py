"""Animations transforming one mobject into another."""

__all__ = [
    "Transform",
    "ReplacementTransform",
    "TransformFromCopy",
    "ClockwiseTransform",
    "CounterclockwiseTransform",
    "MoveToTarget",
    "ApplyMethod",
    "ApplyPointwiseFunction",
    "ApplyPointwiseFunctionToCenter",
    "FadeToColor",
    "ScaleInPlace",
    "ShrinkToCenter",
    "Restore",
    "ApplyFunction",
    "ApplyMatrix",
    "ApplyComplexFunction",
    "CyclicReplace",
    "Swap",
    "TransformAnimations",
]

import inspect
import types
import typing

import numpy as np

from ..animation.animation import Animation
from ..constants import DEFAULT_POINTWISE_FUNCTION_RUN_TIME, DEGREES, OUT
from ..mobject.mobject import Group, Mobject
from ..mobject.opengl_mobject import OpenGLMobject
from ..utils.paths import path_along_arc, straight_path
from ..utils.rate_functions import smooth, squish_rate_func

if typing.TYPE_CHECKING:
    from ..scene.scene import Scene


class Transform(Animation):
    def __init__(
        self,
        mobject: Mobject,
        target_mobject: typing.Optional[Mobject] = None,
        path_func: typing.Optional[typing.Callable] = None,
        path_arc: float = 0,
        path_arc_axis: np.ndarray = OUT,
        replace_mobject_with_target_in_scene: bool = False,
        **kwargs,
    ) -> None:
        self.path_arc = path_arc
        self.path_func = path_func
        self.path_arc_axis = path_arc_axis
        self.replace_mobject_with_target_in_scene = replace_mobject_with_target_in_scene
        self.target_mobject = target_mobject
        super().__init__(mobject, **kwargs)
        self._init_path_func()

    def _init_path_func(self) -> None:
        if self.path_func is not None:
            return
        elif self.path_arc == 0:
            self.path_func = straight_path
        else:
            self.path_func = path_along_arc(
                self.path_arc,
                self.path_arc_axis,
            )

    def begin(self) -> None:
        # Use a copy of target_mobject for the align_data
        # call so that the actual target_mobject stays
        # preserved.
        self.target_mobject = self.create_target()
        self.check_target_mobject_validity()
        self.target_copy = self.target_mobject.copy()
        # Note, this potentially changes the structure
        # of both mobject and target_mobject
        self.mobject.align_data(self.target_copy)
        super().begin()

    def create_target(self) -> typing.Union[Mobject, None]:
        # Has no meaningful effect here, but may be useful
        # in subclasses
        return self.target_mobject

    def check_target_mobject_validity(self) -> None:
        if self.target_mobject is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.create_target not properly implemented"
            )

    def clean_up_from_scene(self, scene: "Scene") -> None:
        super().clean_up_from_scene(scene)
        if self.replace_mobject_with_target_in_scene:
            scene.remove(self.mobject)
            scene.add(self.target_mobject)

    def update_config(self, **kwargs: typing.Dict[str, typing.Any]) -> None:
        Animation.update_config(self, **kwargs)
        if "path_arc" in kwargs:
            self.path_func = path_along_arc(
                kwargs["path_arc"], kwargs.get("path_arc_axis", OUT)
            )

    def get_all_mobjects(self) -> typing.List[Mobject]:
        return [
            self.mobject,
            self.starting_mobject,
            self.target_mobject,
            self.target_copy,
        ]

    def get_all_families_zipped(self) -> typing.Iterable[tuple]:  # more precise typing?
        return zip(
            *[
                mob.family_members_with_points()
                for mob in [
                    self.mobject,
                    self.starting_mobject,
                    self.target_copy,
                ]
            ]
        )

    def interpolate_submobject(
        self,
        submobject: Mobject,
        starting_submobject: Mobject,
        target_copy: Mobject,
        alpha: float,
    ) -> "Transform":  # doesn't match the parent class?
        submobject.interpolate(starting_submobject, target_copy, alpha, self.path_func)
        return self


class ReplacementTransform(Transform):
    def __init__(self, mobject: Mobject, target_mobject: Mobject, **kwargs) -> None:
        super().__init__(
            mobject, target_mobject, replace_mobject_with_target_in_scene=True, **kwargs
        )


class TransformFromCopy(Transform):
    """
    Performs a reversed Transform
    """

    def __init__(self, mobject: Mobject, target_mobject: Mobject, **kwargs) -> None:
        super().__init__(target_mobject, mobject, **kwargs)

    def interpolate(self, alpha: float) -> None:
        super().interpolate(1 - alpha)


class ClockwiseTransform(Transform):
    def __init__(
        self,
        mobject: Mobject,
        target_mobject: Mobject,
        path_arc: float = -np.pi,
        **kwargs,
    ) -> None:
        super().__init__(mobject, target_mobject, path_arc=path_arc, **kwargs)


class CounterclockwiseTransform(Transform):
    def __init__(
        self,
        mobject: Mobject,
        target_mobject: Mobject,
        path_arc: float = np.pi,
        **kwargs,
    ) -> None:
        super().__init__(mobject, target_mobject, path_arc=path_arc, **kwargs)


class MoveToTarget(Transform):
    def __init__(self, mobject: Mobject, **kwargs) -> None:
        self.check_validity_of_input(mobject)
        super().__init__(mobject, mobject.target, **kwargs)

    def check_validity_of_input(self, mobject: Mobject) -> None:
        if not hasattr(mobject, "target"):
            raise ValueError(
                "MoveToTarget called on mobject" "without attribute 'target'"
            )


class _MethodAnimation(MoveToTarget):
    def __init__(self, mobject, methods):
        self.methods = methods
        super().__init__(mobject)


class ApplyMethod(Transform):
    def __init__(
        self, method: types.MethodType, *args, **kwargs
    ) -> None:  # method typing? for args?
        """
        Method is a method of Mobject, ``args`` are arguments for
        that method.  Key word arguments should be passed in
        as the last arg, as a dict, since ``kwargs`` is for
        configuration of the transform itself

        Relies on the fact that mobject methods return the mobject
        """
        self.check_validity_of_input(method)
        self.method = method
        self.method_args = args
        super().__init__(method.__self__, **kwargs)

    def check_validity_of_input(self, method: types.MethodType) -> None:
        if not inspect.ismethod(method):
            raise ValueError(
                "Whoops, looks like you accidentally invoked "
                "the method you want to animate"
            )
        assert isinstance(method.__self__, (Mobject, OpenGLMobject))

    def create_target(self) -> Mobject:
        method = self.method
        # Make sure it's a list so that args.pop() works
        args = list(self.method_args)

        if len(args) > 0 and isinstance(args[-1], dict):
            method_kwargs = args.pop()
        else:
            method_kwargs = {}
        target = method.__self__.copy()
        method.__func__(target, *args, **method_kwargs)
        return target


class ApplyPointwiseFunction(ApplyMethod):
    """Animation that applies a pointwise function to a mobject.

    Examples
    --------

    .. manim:: WarpSquare
        :quality: low

        class WarpSquare(Scene):
            def construct(self):
                square = Square()
                self.play(
                    ApplyPointwiseFunction(
                        lambda point: complex_to_R3(np.exp(R3_to_complex(point))), square
                    )
                )
                self.wait()

    """

    def __init__(
        self,
        function: types.MethodType,
        mobject: Mobject,
        run_time: float = DEFAULT_POINTWISE_FUNCTION_RUN_TIME,
        **kwargs,
    ) -> None:
        super().__init__(mobject.apply_function, function, run_time=run_time, **kwargs)


class ApplyPointwiseFunctionToCenter(ApplyPointwiseFunction):
    def __init__(self, function: types.MethodType, mobject: Mobject, **kwargs) -> None:
        self.function = function
        super().__init__(mobject.move_to, **kwargs)

    def begin(self) -> None:
        self.method_args = [self.function(self.mobject.get_center())]
        super().begin()


class FadeToColor(ApplyMethod):
    def __init__(self, mobject: Mobject, color: str, **kwargs) -> None:
        super().__init__(mobject.set_color, color, **kwargs)


class ScaleInPlace(ApplyMethod):
    def __init__(self, mobject: Mobject, scale_factor: float, **kwargs) -> None:
        super().__init__(mobject.scale, scale_factor, **kwargs)


class ShrinkToCenter(ScaleInPlace):
    def __init__(self, mobject: Mobject, **kwargs) -> None:
        super().__init__(mobject, 0, **kwargs)


class Restore(ApplyMethod):
    def __init__(self, mobject: Mobject, **kwargs) -> None:
        super().__init__(mobject.restore, **kwargs)


class ApplyFunction(Transform):
    def __init__(self, function: types.MethodType, mobject: Mobject, **kwargs) -> None:
        self.function = function
        super().__init__(mobject, **kwargs)

    def create_target(self) -> typing.Any:
        target = self.function(self.mobject.copy())
        if not isinstance(target, Mobject):
            raise TypeError(
                "Functions passed to ApplyFunction must return object of type Mobject"
            )
        return target


class ApplyMatrix(ApplyPointwiseFunction):
    def __init__(self, matrix: np.ndarray, mobject: Mobject, **kwargs) -> None:
        matrix = self.initialize_matrix(matrix)

        def func(p):
            return np.dot(p, matrix.T)

        super().__init__(func, mobject, **kwargs)

    def initialize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.array(matrix)
        if matrix.shape == (2, 2):
            new_matrix = np.identity(3)
            new_matrix[:2, :2] = matrix
            matrix = new_matrix
        elif matrix.shape != (3, 3):
            raise ValueError("Matrix has bad dimensions")
        return matrix


class ApplyComplexFunction(ApplyMethod):
    def __init__(self, function: types.MethodType, mobject: Mobject, **kwargs) -> None:
        self.function = function
        method = mobject.apply_complex_function
        super().__init__(method, function, **kwargs)

    def _init_path_func(self) -> None:
        func1 = self.function(complex(1))
        self.path_arc = np.log(func1).imag
        super()._init_path_func()


###


class CyclicReplace(Transform):
    def __init__(
        self, *mobjects: Mobject, path_arc: float = 90 * DEGREES, **kwargs
    ) -> None:
        self.group = Group(*mobjects)
        super().__init__(self.group, path_arc=path_arc, **kwargs)

    def create_target(self) -> Group:
        target = self.group.copy()
        cycled_targets = [target[-1], *target[:-1]]
        for m1, m2 in zip(cycled_targets, self.group):
            m1.move_to(m2)
        return target


class Swap(CyclicReplace):
    pass  # Renaming, more understandable for two entries


# TODO, this may be deprecated...worth reimplementing?
class TransformAnimations(Transform):
    def __init__(
        self,
        start_anim: Animation,
        end_anim: Animation,
        rate_func: typing.Callable = squish_rate_func(smooth),
        **kwargs,
    ) -> None:
        self.start_anim = start_anim
        self.end_anim = end_anim
        if "run_time" in kwargs:
            self.run_time = kwargs.pop("run_time")
        else:
            self.run_time = max(start_anim.run_time, end_anim.run_time)
        for anim in start_anim, end_anim:
            anim.set_run_time(self.run_time)

        if (
            start_anim.starting_mobject.get_num_points()
            != end_anim.starting_mobject.get_num_points()
        ):
            start_anim.starting_mobject.align_data(end_anim.starting_mobject)
            for anim in start_anim, end_anim:
                if hasattr(anim, "target_mobject"):
                    anim.starting_mobject.align_data(anim.target_mobject)

        super().__init__(
            start_anim.mobject, end_anim.mobject, rate_func=rate_func, **kwargs
        )
        # Rewire starting and ending mobjects
        start_anim.mobject = self.starting_mobject
        end_anim.mobject = self.target_mobject

    def interpolate(self, alpha: float) -> None:
        self.start_anim.interpolate(alpha)
        self.end_anim.interpolate(alpha)
        Transform.interpolate(self, alpha)
