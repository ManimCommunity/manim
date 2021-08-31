from pathops import Path, union
from manim import *
import numpy as np
from manim.utils.simple_functions import fdiv
from rich import print

pw = config.pixel_width
ph = config.pixel_height
fw = config.frame_width
fh = config.frame_height
fc = [0, 0, 0]

matrix = {
        'scaleX':fdiv(pw, fw),
        'skewX':0,
        'translateX':0,
        'skewY':-fdiv(ph ,fw),
        'scaleY': (pw / 2) - fc[0] * fdiv(pw, fw),
        'translateY':(ph / 2) + fc[1] * fdiv(ph , fh)
}


def convert_vmobject_to_skia_path(vmobject: VMobject) -> Path:
    path = Path()
    
    path.transform(**matrix)
    
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

def convert_skia_path_to_vmobject(path: Path) -> VMobject:
    vmobject = VMobject()
    segments = path.segments
    for segment in segments:
        if segment[0] == 'moveTo':
            parts = segment[1]
            for part in parts:
                vmobject.move_to(*part)
        elif segment[0] == 'curveTo':
            parts = segment[1]
            print(parts)
            for part in parts:
                pass
#                vmobject.curv TODO: From here

m1 = Square()
m2 = Circle().move_to([.2,.2,0])
a = convert_vmobject_to_skia_path(m1)
b = convert_vmobject_to_skia_path(m2)

#print(list(a.segments))
#print(list(b.segments))


c= Path()
union([a,b],c.getPen())

convert_skia_path_to_vmobject(c)

