"""Boolean operations for two-dimensional mobjects."""

from __future__ import annotations

import typing

import numpy as np
from pathops import Path as SkiaPath
from pathops import PathVerb, difference, intersection, union, xor

from manim import config
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject

from ...constants import RendererType

__all__ = ["Union", "Intersection", "Difference", "Exclusion"]


class _BooleanOps(VMobject, metaclass=ConvertToOpenGL):
    """This class contains some helper functions which
    helps to convert to and from skia objects and manim
    objects (:class:`~.VMobject`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _convert_2d_to_3d_array(
        self,
        points: typing.Iterable,
        z_dim: float = 0.0,
    ) -> list[np.ndarray]:
        """Converts an iterable with coordinates in 2d to 3d by adding
        :attr:`z_dim` as the z coordinate.

        Parameters
        ----------
        points:
            An iterable which has the coordinates.
        z_dim:
            The default value of z coordinate.

        Returns
        -------
        typing.List[np.ndarray]
            A list of array converted to 3d.

        Example
        -------
        >>> a = _BooleanOps()
        >>> p = [(1, 2), (3, 4)]
        >>> a._convert_2d_to_3d_array(p)
        [array([1., 2., 0.]), array([3., 4., 0.])]
        """
        points = list(points)
        for i, point in enumerate(points):
            if len(point) == 2:
                points[i] = np.array(list(point) + [z_dim])
        return points

    def _convert_vmobject_to_skia_path(self, vmobject: VMobject) -> SkiaPath:
        """Converts a :class:`~.VMobject` to SkiaPath. This method only works for
        cairo renderer because it treats the points as Cubic beizer curves.

        Parameters
        ----------
        vmobject:
            The :class:`~.VMobject` to convert from.

        Returns
        -------
        SkiaPath:
            The converted path.
        """
        path = SkiaPath()

        if not np.all(np.isfinite(vmobject.points)):
            points = np.zeros((1, 3))  # point invalid?
        else:
            points = vmobject.points

        if len(points) == 0:  # what? No points so return empty path
            return path

        # In OpenGL it's quadratic beizer curves while on Cairo it's cubic...
        if config.renderer == RendererType.OPENGL:
            subpaths = vmobject.get_subpaths_from_points(points)
            for subpath in subpaths:
                quads = vmobject.get_bezier_tuples_from_points(subpath)
                start = subpath[0]
                path.moveTo(*start[:2])
                for p0, p1, p2 in quads:
                    path.quadTo(*p1[:2], *p2[:2])
                if vmobject.consider_points_equals(subpath[0], subpath[-1]):
                    path.close()
        elif config.renderer == RendererType.CAIRO:
            subpaths = vmobject.gen_subpaths_from_points_2d(points)
            for subpath in subpaths:
                quads = vmobject.gen_cubic_bezier_tuples_from_points(subpath)
                start = subpath[0]
                path.moveTo(*start[:2])
                for p0, p1, p2, p3 in quads:
                    path.cubicTo(*p1[:2], *p2[:2], *p3[:2])

                if vmobject.consider_points_equals_2d(subpath[0], subpath[-1]):
                    path.close()

        return path

    def _convert_skia_path_to_vmobject(self, path: SkiaPath) -> VMobject:
        """Converts SkiaPath back to VMobject.
        Parameters
        ----------
        path:
            The SkiaPath to convert.

        Returns
        -------
        VMobject:
            The converted VMobject.
        """
        vmobject = self
        current_path_start = np.array([0, 0, 0])

        for path_verb, points in path:
            if path_verb == PathVerb.MOVE:
                parts = self._convert_2d_to_3d_array(points)
                for part in parts:
                    current_path_start = part
                    vmobject.start_new_path(part)
                    # vmobject.move_to(*part)
            elif path_verb == PathVerb.CUBIC:
                n1, n2, n3 = self._convert_2d_to_3d_array(points)
                vmobject.add_cubic_bezier_curve_to(n1, n2, n3)
            elif path_verb == PathVerb.LINE:
                parts = self._convert_2d_to_3d_array(points)
                vmobject.add_line_to(parts[0])
            elif path_verb == PathVerb.CLOSE:
                vmobject.add_line_to(current_path_start)
            elif path_verb == PathVerb.QUAD:
                n1, n2 = self._convert_2d_to_3d_array(points)
                vmobject.add_quadratic_bezier_curve_to(n1, n2)
            else:
                raise Exception("Unsupported: %s" % path_verb)
        return vmobject


class Union(_BooleanOps):
    """Union of two or more :class:`~.VMobject` s. This returns the common region of
    the :class:`~VMobject` s.

    Parameters
    ----------
    vmobjects
        The :class:`~.VMobject` s to find the union of.

    Raises
    ------
    ValueError
        If less than 2 :class:`~.VMobject` s are passed.

    Example
    -------
    .. manim:: UnionExample
        :save_last_frame:

        class UnionExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Union(sq, cr, color=GREEN, fill_opacity=1)
                un.move_to([1.5, 0.3, 0])
                self.add(sq, cr, un)

    """

    def __init__(self, *vmobjects: VMobject, **kwargs) -> None:
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Union.")
        super().__init__(**kwargs)
        paths = []
        for vmobject in vmobjects:
            paths.append(self._convert_vmobject_to_skia_path(vmobject))
        outpen = SkiaPath()
        union(paths, outpen.getPen())
        self._convert_skia_path_to_vmobject(outpen)


class Difference(_BooleanOps):
    """Subtracts one :class:`~.VMobject` from another one.

    Parameters
    ----------
    subject
        The 1st :class:`~.VMobject`.
    clip
        The 2nd :class:`~.VMobject`

    Example
    -------
    .. manim:: DifferenceExample
        :save_last_frame:

        class DifferenceExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Difference(sq, cr, color=GREEN, fill_opacity=1)
                un.move_to([1.5, 0, 0])
                self.add(sq, cr, un)

    """

    def __init__(self, subject, clip, **kwargs) -> None:
        super().__init__(**kwargs)
        outpen = SkiaPath()
        difference(
            [self._convert_vmobject_to_skia_path(subject)],
            [self._convert_vmobject_to_skia_path(clip)],
            outpen.getPen(),
        )
        self._convert_skia_path_to_vmobject(outpen)


class Intersection(_BooleanOps):
    """Find the intersection of two :class:`~.VMobject` s.
    This keeps the parts covered by both :class:`~.VMobject` s.

    Parameters
    ----------
    vmobjects
        The :class:`~.VMobject` to find the intersection.

    Raises
    ------
    ValueError
        If less the 2 :class:`~.VMobject` are passed.

    Example
    -------
    .. manim:: IntersectionExample
        :save_last_frame:

        class IntersectionExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Intersection(sq, cr, color=GREEN, fill_opacity=1)
                un.move_to([1.5, 0, 0])
                self.add(sq, cr, un)

    """

    def __init__(self, *vmobjects, **kwargs) -> None:
        if len(vmobjects) < 2:
            raise ValueError("At least 2 mobjects needed for Intersection.")

        super().__init__(**kwargs)
        outpen = SkiaPath()
        intersection(
            [self._convert_vmobject_to_skia_path(vmobjects[0])],
            [self._convert_vmobject_to_skia_path(vmobjects[1])],
            outpen.getPen(),
        )
        new_outpen = outpen
        for _i in range(2, len(vmobjects)):
            new_outpen = SkiaPath()
            intersection(
                [outpen],
                [self._convert_vmobject_to_skia_path(vmobjects[_i])],
                new_outpen.getPen(),
            )
            outpen = new_outpen

        self._convert_skia_path_to_vmobject(outpen)


class Exclusion(_BooleanOps):
    """Find the XOR between two :class:`~.VMobject`.
    This creates a new :class:`~.VMobject` consisting of the region
    covered by exactly one of them.

    Parameters
    ----------
    subject
        The 1st :class:`~.VMobject`.
    clip
        The 2nd :class:`~.VMobject`

    Example
    -------
    .. manim:: IntersectionExample
        :save_last_frame:

        class IntersectionExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Exclusion(sq, cr, color=GREEN, fill_opacity=1)
                un.move_to([1.5, 0.4, 0])
                self.add(sq, cr, un)

    """

    def __init__(self, subject, clip, **kwargs) -> None:
        super().__init__(**kwargs)
        outpen = SkiaPath()
        xor(
            [self._convert_vmobject_to_skia_path(subject)],
            [self._convert_vmobject_to_skia_path(clip)],
            outpen.getPen(),
        )
        self._convert_skia_path_to_vmobject(outpen)
