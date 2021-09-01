from .types.vectorized_mobject import VMobject
from pathops import Path as SkiaPath, union, intersection, difference
import numpy as np
import typing
from .. import config

__all__ = ["Union", "Intersection"]


class _BooleanOps(VMobject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _convert_2d_to_3d_array(self, points: typing.Iterable, z_dim: float=0.) -> typing.List[np.ndarray]:
        points = list(points)
        for i, point in enumerate(points):
            if len(point) == 2:
                points[i] = np.array(list(point) + [z_dim])
        return points

    def _convert_vmobject_to_skia_path(self, vmobject: VMobject) -> SkiaPath:
        path = SkiaPath()
        
        if not np.all(np.isfinite(vmobject.points)):
            points = np.zeros((1,3)) # point invalid?
        else:
            points = vmobject.points

        if len(points) == 0: # what? No points so return empty path
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
        vmobject = self
        segments = path.segments
        current_path_start = np.array([0 ,0, 0])
        for segment in segments:
            if segment[0] == 'moveTo':
                parts = self._convert_2d_to_3d_array(segment[1])
                for part in parts:
                    a = part
                    current_path_start = a
                    vmobject.start_new_path(a)
                    #vmobject.move_to(*part)
            elif segment[0] == 'curveTo':
                parts = segment[1]
                n1, n2, n3 = self._convert_2d_to_3d_array(parts)
                vmobject.add_cubic_bezier_curve_to(n1, n2, n3)
            elif segment[0] == 'lineTo':
                part = self._convert_2d_to_3d_array(segment[1])
                vmobject.add_line_to(part[0])
            elif segment[0] == 'closePath':
                if config.renderer == 'opengl':
                    vmobject.close_path()
                else:
                    vmobject.add_line_to(current_path_start)
            else:
                raise Exception("Unsupported: %s"%segment[0])
        return vmobject


class Union(_BooleanOps):
    def __init__(self, *vmobjects: VMobject, **kwargs) -> None:
        super().__init__(self, **kwargs)
        paths = []
        for vmobject in vmobjects:
            paths.append(self._convert_vmobject_to_skia_path(vmobject))
        outpen = SkiaPath()
        union(paths, outpen.getPen())
        self._convert_skia_path_to_vmobject(outpen)

class Difference(_BooleanOps):
    def __init__(self, subject, clip, **kwargs) -> None:
        super().__init__(self, **kwargs)
        #paths = []
        #for vmobject in vmobjects:
        #    paths.append(self._convert_vmobject_to_skia_path(vmobject))
        outpen = SkiaPath()
        difference([self._convert_vmobject_to_skia_path(subject)],[self._convert_vmobject_to_skia_path(clip)], outpen.getPen())
        self._convert_skia_path_to_vmobject(outpen)


class Intersection(_BooleanOps):
    def __init__(self, *vmobjects, **kwargs) -> None:
        super().__init__(self, **kwargs)
        outpen = SkiaPath()
#        intersection([])


