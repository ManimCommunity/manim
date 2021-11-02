"""Animations that try to transform Mobjects while keeping track of identical parts."""

__all__ = ["TransformMatchingShapes", "TransformMatchingTex"]

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from .._config import config
from ..mobject.mobject import Group, Mobject
from ..mobject.opengl_mobject import OpenGLGroup, OpenGLMobject
from ..mobject.types.opengl_vectorized_mobject import OpenGLVGroup, OpenGLVMobject
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from .composition import AnimationGroup
from .fading import FadeIn, FadeOut
from .transform import FadeTransformPieces, Transform

if TYPE_CHECKING:
    from ..scene.scene import Scene


class TransformMatchingAbstractBase(AnimationGroup):
    """Abstract base class for transformations that keep track of matching parts.

    Subclasses have to implement the two static methods
    :meth:`~.TransformMatchingAbstractBase.get_mobject_parts` and
    :meth:`~.TransformMatchingAbstractBase.get_mobject_key`.

    Basically, this transformation first maps all submobjects returned
    by the ``get_mobject_parts`` method to certain keys by applying the
    ``get_mobject_key`` method. Then, submobjects with matching keys
    are transformed into each other.

    Parameters
    ----------
    mobject
        The starting :class:`~.Mobject`.
    target_mobject
        The target :class:`~.Mobject`.
    transform_mismatches
        Controls whether submobjects without a matching key are transformed
        into each other by using :class:`~.Transform`. Default: ``False``.
    fade_transform_mismatches
        Controls whether submobjects without a matching key are transformed
        into each other by using :class:`~.FadeTransform`. Default: ``False``.
    key_map
        Optional. A dictionary mapping keys belonging to some of the starting mobject's
        submobjects (i.e., the return values of the ``get_mobject_key`` method)
        to some keys belonging to the target mobject's submobjects that should
        be transformed although the keys don't match.
    kwargs
        All further keyword arguments are passed to the submobject transformations.


    Note
    ----
    If neither ``transform_mismatches`` nor ``fade_transform_mismatches``
    are set to ``True``, submobjects without matching keys in the starting
    mobject are faded out in the direction of the unmatched submobjects in
    the target mobject, and unmatched submobjects in the target mobject
    are faded in from the direction of the unmatched submobjects in the
    start mobject.

    """

    def __init__(
        self,
        mobject: "Mobject",
        target_mobject: "Mobject",
        transform_mismatches: bool = False,
        fade_transform_mismatches: bool = False,
        key_map: Optional[dict] = None,
        **kwargs
    ):

        if isinstance(mobject, OpenGLVMobject):
            group_type = OpenGLVGroup
        elif isinstance(mobject, OpenGLMobject):
            group_type = OpenGLGroup
        elif isinstance(mobject, VMobject):
            group_type = VGroup
        else:
            group_type = Group

        source_map = self.get_shape_map(mobject)
        target_map = self.get_shape_map(target_mobject)

        if key_map is None:
            key_map = {}

        # Create two mobjects whose submobjects all match each other
        # according to whatever keys are used for source_map and
        # target_map
        transform_source = group_type()
        transform_target = group_type()
        kwargs["final_alpha_value"] = 0
        for key in set(source_map).intersection(target_map):
            transform_source.add(source_map[key])
            transform_target.add(target_map[key])
        anims = [Transform(transform_source, transform_target, **kwargs)]
        # User can manually specify when one part should transform
        # into another despite not matching by using key_map
        key_mapped_source = group_type()
        key_mapped_target = group_type()
        for key1, key2 in key_map.items():
            if key1 in source_map and key2 in target_map:
                key_mapped_source.add(source_map[key1])
                key_mapped_target.add(target_map[key2])
                source_map.pop(key1, None)
                target_map.pop(key2, None)
        if len(key_mapped_source) > 0:
            anims.append(
                FadeTransformPieces(key_mapped_source, key_mapped_target, **kwargs),
            )

        fade_source = group_type()
        fade_target = group_type()
        for key in set(source_map).difference(target_map):
            fade_source.add(source_map[key])
        for key in set(target_map).difference(source_map):
            fade_target.add(target_map[key])

        if transform_mismatches:
            if "replace_mobject_with_target_in_scene" not in kwargs:
                kwargs["replace_mobject_with_target_in_scene"] = True
            anims.append(Transform(fade_source, fade_target, **kwargs))
        elif fade_transform_mismatches:
            anims.append(FadeTransformPieces(fade_source, fade_target, **kwargs))
        else:
            anims.append(FadeOut(fade_source, target_position=fade_target, **kwargs))
            anims.append(
                FadeIn(fade_target.copy(), target_position=fade_target, **kwargs),
            )

        super().__init__(*anims)

        self.to_remove = mobject
        self.to_add = target_mobject

    def get_shape_map(self, mobject: "Mobject") -> dict:
        shape_map = {}
        for sm in self.get_mobject_parts(mobject):
            key = self.get_mobject_key(sm)
            if key not in shape_map:
                if config["renderer"] == "opengl":
                    shape_map[key] = OpenGLVGroup()
                else:
                    shape_map[key] = VGroup()
            shape_map[key].add(sm)
        return shape_map

    def clean_up_from_scene(self, scene: "Scene") -> None:
        for anim in self.animations:
            anim.interpolate(0)
        scene.remove(self.mobject)
        scene.remove(self.to_remove)
        scene.add(self.to_add)

    @staticmethod
    def get_mobject_parts(mobject: "Mobject"):
        raise NotImplementedError("To be implemented in subclass.")

    @staticmethod
    def get_mobject_key(mobject: "Mobject"):
        raise NotImplementedError("To be implemented in subclass.")


class TransformMatchingShapes(TransformMatchingAbstractBase):
    """An animation trying to transform groups by matching the shape
    of their submobjects.

    Two submobjects match if the hash of their point coordinates after
    normalization (i.e., after translation to the origin, fixing the submobject
    height at 1 unit, and rounding the coordinates to three decimal places)
    matches.

    See also
    --------
    :class:`~.TransformMatchingAbstractBase`

    Examples
    --------

    .. manim:: Anagram

        class Anagram(Scene):
            def construct(self):
                src = Text("the morse code")
                tar = Text("here come dots")
                self.play(Write(src))
                self.wait(0.5)
                self.play(TransformMatchingShapes(src, tar, path_arc=PI/2))
                self.wait(0.5)

    """

    def __init__(
        self,
        mobject: "Mobject",
        target_mobject: "Mobject",
        transform_mismatches: bool = False,
        fade_transform_mismatches: bool = False,
        key_map: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(
            mobject,
            target_mobject,
            transform_mismatches=transform_mismatches,
            fade_transform_mismatches=fade_transform_mismatches,
            key_map=key_map,
            **kwargs
        )

    @staticmethod
    def get_mobject_parts(mobject: "Mobject") -> List["Mobject"]:
        return mobject.family_members_with_points()

    @staticmethod
    def get_mobject_key(mobject: "Mobject") -> int:
        mobject.save_state()
        mobject.center()
        mobject.set_height(1)
        result = hash(np.round(mobject.points, 3).tobytes())
        mobject.restore()
        return result


class TransformMatchingTex(TransformMatchingAbstractBase):
    """A transformation trying to transform rendered LaTeX strings.

    Two submobjects match if their ``tex_string`` matches.

    See also
    --------
    :class:`~.TransformMatchingAbstractBase`

    Examples
    --------

    .. manim:: MatchingEquationParts

        class MatchingEquationParts(Scene):
            def construct(self):
                eq1 = MathTex("{{a^2}} + {{b^2}} = {{c^2}}")
                eq2 = MathTex("{{a^2}} = {{c^2}} - {{b^2}}")
                self.add(eq1)
                self.wait(0.5)
                self.play(TransformMatchingTex(eq1, eq2))
                self.wait(0.5)

    """

    def __init__(
        self,
        mobject: "Mobject",
        target_mobject: "Mobject",
        transform_mismatches: bool = False,
        fade_transform_mismatches: bool = False,
        key_map: Optional[dict] = None,
        **kwargs
    ):
        assert hasattr(mobject, "tex_string")
        assert hasattr(target_mobject, "tex_string")
        super().__init__(
            mobject,
            target_mobject,
            transform_mismatches=transform_mismatches,
            fade_transform_mismatches=fade_transform_mismatches,
            key_map=key_map,
            **kwargs
        )

    @staticmethod
    def get_mobject_parts(mobject: "Mobject") -> List["Mobject"]:
        return mobject.submobjects

    @staticmethod
    def get_mobject_key(mobject: "Mobject") -> str:
        return mobject.tex_string
