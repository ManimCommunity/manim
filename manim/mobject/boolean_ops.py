import typing

import numpy as np
from pathops import Path as SkiaPath
from pathops import difference, intersection, union, xor

from .. import config
from .types.vectorized_mobject import VMobject

__all__ = ["Union", "Intersection", "Difference", "Exclusion"]


class _BooleanOps(VMobject):
    """This class contains some helper functions which
    helps to convert to and from skia objects to manim
    object(VMobjects).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _convert_2d_to_3d_array(
        self,
        points: typing.Iterable,
        z_dim: float = 0.0,
    ) -> typing.List[np.ndarray]:
        """Converts an iterable with coordinates in 2d to 3d by adding
        :attr:`z_dim` as the z coordinate.

        Parameters
        ==========
        points:
            An iterable which has the coordinates.
        z_dim:
            The default value of z coordinate.

        Example
        =======
        >>> a = _BooleanOps()
        >>> p = [(1, 2), (3, 4)]
        >>> a._convert_2d_to_3d_array(p)
        [array([1., 2., 0.]), array([3., 4., 0.])]

        Returns
        =======
        typing.List[np.ndarray]
            A list of array converted to 3d.
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
        ==========
        vmobject:
            The :class:`~.VMobject` to convert from.

        Returns
        =======
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
        ==========
        path:
            The SkiaPath to convert.

        Returns
        =======
        VMobject:
            The converted VMobject.
        """
        vmobject = self
        segments = path.segments
        current_path_start = np.array([0, 0, 0])
        for segment in segments:
            if segment[0] == "moveTo":
                parts = self._convert_2d_to_3d_array(segment[1])
                for part in parts:
                    a = part
                    current_path_start = a
                    vmobject.start_new_path(a)
                    # vmobject.move_to(*part)
            elif segment[0] == "curveTo":
                parts = segment[1]
                n1, n2, n3 = self._convert_2d_to_3d_array(parts)
                vmobject.add_cubic_bezier_curve_to(n1, n2, n3)
            elif segment[0] == "lineTo":
                part = self._convert_2d_to_3d_array(segment[1])
                vmobject.add_line_to(part[0])
            elif segment[0] == "closePath":
                if config.renderer == "opengl":
                    vmobject.close_path()
                else:
                    vmobject.add_line_to(current_path_start)
            elif segment[0] == "qCurveTo":
                parts = segment[1]
                n1, n2 = self._convert_2d_to_3d_array(parts)
                vmobject.add_quadratic_bezier_curve_to(n1, n2)
            elif segment[0] == "endPath":  # usually will not be executed
                pass
            else:
                raise Exception("Unsupported: %s" % segment[0])
        return vmobject


class Union(_BooleanOps):
    """Union of 2 or more :class:`~.VMobject`. This finds the commom outline of
    the two :class:`VMobject`.

    Parameters
    ==========
    vmobjects
        The :class:`~.VMobject` to find the union.

    Raises
    ======
    ValueError
        If less the 2 :class:`~.VMobject` are passed.

    Example
    =======

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
            raise ValueError("Atleast 2 mobjects needed for Union.")
        super().__init__(**kwargs)
        paths = []
        for vmobject in vmobjects:
            paths.append(self._convert_vmobject_to_skia_path(vmobject))
        outpen = SkiaPath()
        union(paths, outpen.getPen())
        self._convert_skia_path_to_vmobject(outpen)


class Difference(_BooleanOps):
    """Subracts one :class:`~.VMobject` from another one.
    Parameters
    ==========
    subject
        The 1st :class:`~.VMobject`.
    clip
        The 2nd :class:`~.VMobject`

    Example
    =======
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
    """Find intersection between two :class:`~.VMobject`.
    This keeps the parts covered by both :class:`VMobject`.

    Parameters
    ==========
    subject
        The 1st :class:`~.VMobject`.
    clip
        The 2nd :class:`~.VMobject`.

    Example
    =======
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

    def __init__(self, subject, clip, **kwargs) -> None:
        super().__init__(**kwargs)
        outpen = SkiaPath()
        intersection(
            [self._convert_vmobject_to_skia_path(subject)],
            [self._convert_vmobject_to_skia_path(clip)],
            outpen.getPen(),
        )
        self._convert_skia_path_to_vmobject(outpen)


class Exclusion(_BooleanOps):
    """Find the XOR between two :class:`VMobject`.
    This creates a new :class:`~.VMobject` which keeps the region
    which is covered by both of them.

    Parameters
    ==========
    subject
        The 1st :class:`~.VMobject`.
    clip
        The 2nd :class:`~.VMobject`

    Example
    =======

    .. manim:: IntersectionExample
        :save_last_frame:

        class IntersectionExample(Scene):
            def construct(self):
                sq = Square(color=RED, fill_opacity=1)
                sq.move_to([-2, 0, 0])
                cr = Circle(color=BLUE, fill_opacity=1)
                cr.move_to([-1.3, 0.7, 0])
                un = Xor(sq, cr, color=GREEN, fill_opacity=1)
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
