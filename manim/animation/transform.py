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
    "FadeTransform",
    "FadeTransformPieces",
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
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

import numpy as np

from .. import config
from ..animation.animation import Animation
from ..constants import DEFAULT_POINTWISE_FUNCTION_RUN_TIME, DEGREES, OUT
from ..mobject.mobject import Group, Mobject
from ..mobject.opengl_mobject import OpenGLGroup, OpenGLMobject
from ..utils.paths import path_along_arc
from ..utils.rate_functions import smooth, squish_rate_func

if TYPE_CHECKING:
    from ..scene.scene import Scene


class Transform(Animation):
    def __init__(
        self,
        mobject: Optional[Mobject],
        target_mobject: Optional[Mobject] = None,
        path_func: Optional[Callable] = None,
        path_arc: float = 0,
        path_arc_axis: np.ndarray = OUT,
        path_arc_centers: np.ndarray = None,
        replace_mobject_with_target_in_scene: bool = False,
        **kwargs,
    ) -> None:
        self.path_arc_axis: np.ndarray = path_arc_axis
        self.path_arc_centers: np.ndarray = path_arc_centers
        self.path_arc: float = path_arc
        self.path_func: Optional[Callable] = path_func
        self.replace_mobject_with_target_in_scene: bool = (
            replace_mobject_with_target_in_scene
        )
        self.target_mobject: Mobject = (
            target_mobject if target_mobject is not None else Mobject()
        )
        super().__init__(mobject, **kwargs)

    @property
    def path_arc(self) -> float:
        return self._path_arc

    @path_arc.setter
    def path_arc(self, path_arc: float) -> None:
        self._path_arc = path_arc
        self._path_func = path_along_arc(
            arc_angle=self._path_arc,
            axis=self.path_arc_axis,
            arc_centers=self.path_arc_centers,
        )

    @property
    def path_func(
        self,
    ) -> Callable[
        [Iterable[np.ndarray], Iterable[np.ndarray], float],
        Iterable[np.ndarray],
    ]:
        return self._path_func

    @path_func.setter
    def path_func(
        self,
        path_func: Callable[
            [Iterable[np.ndarray], Iterable[np.ndarray], float],
            Iterable[np.ndarray],
        ],
    ) -> None:
        if path_func is not None:
            self._path_func = path_func

    def begin(self) -> None:
        # Use a copy of target_mobject for the align_data
        # call so that the actual target_mobject stays
        # preserved.
        self.target_mobject = self.create_target()
        self.target_copy = self.target_mobject.copy()
        # Note, this potentially changes the structure
        # of both mobject and target_mobject
        if config["renderer"] == "opengl":
            self.mobject.align_data_and_family(self.target_copy)
        else:
            self.mobject.align_data(self.target_copy)
        super().begin()

    def create_target(self) -> Mobject:
        # Has no meaningful effect here, but may be useful
        # in subclasses
        return self.target_mobject

    def clean_up_from_scene(self, scene: "Scene") -> None:
        super().clean_up_from_scene(scene)
        if self.replace_mobject_with_target_in_scene:
            scene.remove(self.mobject)
            scene.add(self.target_mobject)

    def get_all_mobjects(self) -> Sequence[Mobject]:
        return [
            self.mobject,
            self.starting_mobject,
            self.target_mobject,
            self.target_copy,
        ]

    def get_all_families_zipped(self) -> Iterable[tuple]:  # more precise typing?
        mobs = [
            self.mobject,
            self.starting_mobject,
            self.target_copy,
        ]
        if config["renderer"] == "opengl":
            return zip(*(mob.get_family() for mob in mobs))
        return zip(*(mob.family_members_with_points() for mob in mobs))

    def interpolate_submobject(
        self,
        submobject: Mobject,
        starting_submobject: Mobject,
        target_copy: Mobject,
        alpha: float,
    ) -> "Transform":
        submobject.interpolate(starting_submobject, target_copy, alpha, self.path_func)
        return self


class ReplacementTransform(Transform):
    """Replaces and morphs a mobject into a target mobject.

    Parameters
    ----------
    mobject
        The starting :class:`~.Mobject`.
    target_mobject
        The target :class:`~.Mobject`.
    kwargs
        Further keyword arguments that are passed to :class:`Transform`.

    Examples
    --------

    .. manim:: ReplacementTransformOrTransform
        :quality: low

        class ReplacementTransformOrTransform(Scene):
            def construct(self):
                # set up the numbers
                r_transform = VGroup(*[Integer(i) for i in range(1,4)])
                text_1 = Text("ReplacementTransform", color=RED)
                r_transform.add(text_1)

                transform = VGroup(*[Integer(i) for i in range(4,7)])
                text_2 = Text("Transform", color=BLUE)
                transform.add(text_2)

                ints = VGroup(r_transform, transform)
                texts = VGroup(text_1, text_2).scale(0.75)
                r_transform.arrange(direction=UP, buff=1)
                transform.arrange(direction=UP, buff=1)

                ints.arrange(buff=2)
                self.add(ints, texts)

                # The mobs replace each other and none are left behind
                self.play(ReplacementTransform(r_transform[0], r_transform[1]))
                self.play(ReplacementTransform(r_transform[1], r_transform[2]))

                # The mobs linger after the Transform()
                self.play(Transform(transform[0], transform[1]))
                self.play(Transform(transform[1], transform[2]))
                self.wait()

    """

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
                "MoveToTarget called on mobject" "without attribute 'target'",
            )


class _MethodAnimation(MoveToTarget):
    def __init__(self, mobject, methods):
        self.methods = methods
        super().__init__(mobject)


class ApplyMethod(Transform):
    def __init__(
        self, method: Callable, *args, **kwargs
    ) -> None:  # method typing (we want to specify Mobject method)? for args?
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

    def check_validity_of_input(self, method: Callable) -> None:
        if not inspect.ismethod(method):
            raise ValueError(
                "Whoops, looks like you accidentally invoked "
                "the method you want to animate",
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

    def create_target(self) -> Any:
        target = self.function(self.mobject.copy())
        if not isinstance(target, (Mobject, OpenGLMobject)):
            raise TypeError(
                "Functions passed to ApplyFunction must return object of type Mobject",
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
        rate_func: Callable = squish_rate_func(smooth),
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
            start_anim.starting_mobject is not None
            and end_anim.starting_mobject is not None
            and start_anim.starting_mobject.get_num_points()
            != end_anim.starting_mobject.get_num_points()
        ):
            start_anim.starting_mobject.align_data(end_anim.starting_mobject)
            for anim in start_anim, end_anim:
                if isinstance(anim, Transform) and anim.starting_mobject is not None:
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
        super().interpolate(alpha)


class FadeTransform(Transform):
    """Fades one mobject into another.

    Parameters
    ----------
    mobject
        The starting :class:`~.Mobject`.
    target_mobject
        The target :class:`~.Mobject`.
    stretch
        Controls whether the target :class:`~.Mobject` is stretched during
        the animation. Default: ``True``.
    dim_to_match
        If the target mobject is not stretched automatically, this allows
        to adjust the initial scale of the target :class:`~.Mobject` while
        it is shifted in. Setting this to 0, 1, and 2, respectively,
        matches the length of the target with the length of the starting
        :class:`~.Mobject` in x, y, and z direction, respectively.
    kwargs
        Further keyword arguments are passed to the parent class.

    Examples
    --------

    .. manim:: DifferentFadeTransforms

        class DifferentFadeTransforms(Scene):
            def construct(self):
                starts = [Rectangle(width=4, height=1) for _ in range(3)]
                VGroup(*starts).arrange(DOWN, buff=1).shift(3*LEFT)
                targets = [Circle(fill_opacity=1).scale(0.25) for _ in range(3)]
                VGroup(*targets).arrange(DOWN, buff=1).shift(3*RIGHT)

                self.play(*[FadeIn(s) for s in starts])
                self.play(
                    FadeTransform(starts[0], targets[0], stretch=True),
                    FadeTransform(starts[1], targets[1], stretch=False, dim_to_match=0),
                    FadeTransform(starts[2], targets[2], stretch=False, dim_to_match=1)
                )

                self.play(*[FadeOut(mobj) for mobj in self.mobjects])

    """

    def __init__(self, mobject, target_mobject, stretch=True, dim_to_match=1, **kwargs):
        self.to_add_on_completion = target_mobject
        self.stretch = stretch
        self.dim_to_match = dim_to_match
        mobject.save_state()
        if config["renderer"] == "opengl":
            group = OpenGLGroup(mobject, target_mobject.copy())
        else:
            group = Group(mobject, target_mobject.copy())
        super().__init__(group, **kwargs)

    def begin(self):
        """Initial setup for the animation.

        The mobject to which this animation is bound is a group consisting of
        both the starting and the ending mobject. At the start, the ending
        mobject replaces the starting mobject (and is completely faded). In the
        end, it is set to be the other way around.
        """
        self.ending_mobject = self.mobject.copy()
        Animation.begin(self)
        # Both 'start' and 'end' consists of the source and target mobjects.
        # At the start, the target should be faded replacing the source,
        # and at the end it should be the other way around.
        start, end = self.starting_mobject, self.ending_mobject
        for m0, m1 in ((start[1], start[0]), (end[0], end[1])):
            self.ghost_to(m0, m1)

    def ghost_to(self, source, target):
        """Replaces the source by the target and sets the opacity to 0."""
        source.replace(target, stretch=self.stretch, dim_to_match=self.dim_to_match)
        source.set_opacity(0)

    def get_all_mobjects(self) -> Sequence[Mobject]:
        return [
            self.mobject,
            self.starting_mobject,
            self.ending_mobject,
        ]

    def get_all_families_zipped(self):
        return Animation.get_all_families_zipped(self)

    def clean_up_from_scene(self, scene):
        Animation.clean_up_from_scene(self, scene)
        scene.remove(self.mobject)
        self.mobject[0].restore()
        scene.add(self.to_add_on_completion)


class FadeTransformPieces(FadeTransform):
    """Fades submobjects of one mobject into submobjects of another one.

    See also
    --------
    :class:`~.FadeTransform`

    Examples
    --------
    .. manim:: FadeTransformSubmobjects

        class FadeTransformSubmobjects(Scene):
            def construct(self):
                src = VGroup(Square(), Circle().shift(LEFT + UP))
                src.shift(3*LEFT + 2*UP)
                src_copy = src.copy().shift(4*DOWN)

                target = VGroup(Circle(), Triangle().shift(RIGHT + DOWN))
                target.shift(3*RIGHT + 2*UP)
                target_copy = target.copy().shift(4*DOWN)

                self.play(FadeIn(src), FadeIn(src_copy))
                self.play(
                    FadeTransform(src, target),
                    FadeTransformPieces(src_copy, target_copy)
                )
                self.play(*[FadeOut(mobj) for mobj in self.mobjects])

    """

    def begin(self):
        self.mobject[0].align_submobjects(self.mobject[1])
        super().begin()

    def ghost_to(self, source, target):
        """Replaces the source submobjects by the target submobjects and sets
        the opacity to 0.
        """
        for sm0, sm1 in zip(source.get_family(), target.get_family()):
            super().ghost_to(sm0, sm1)
