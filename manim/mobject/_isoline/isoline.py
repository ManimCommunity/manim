from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

from .cell import Cell, build_tree
from .point import Func, Point, ValuedPoint, binary_search_zero


def plot_implicit(
    fn: Func,
    pmin: Point,
    pmax: Point,
    min_depth: int = 5,
    max_quads: int = 10000,
):
    """Get the curve representing fn([x,y])=0 on pmin[0] ≤ x ≤ pmax[0] ∩ pmin[1] ≤ y ≤ pmax[1]
    Returns as a list of curves, where each curve is a list of points"""
    quadtree = build_tree(2, fn, pmin, pmax, min_depth, max_quads)
    triangles = Triangulator(quadtree, fn).triangulate()
    return CurveTracer(triangles, fn).trace()


@dataclass
class Triangle:
    vertices: list[ValuedPoint]
    """ The order of triangle "next" is such that, when walking along the isoline in the direction of next,
    you keep positive function values on your right and negative function values on your left."""
    next: Triangle | None = None
    prev: Triangle | None = None
    visited: bool = False

    def set_next(self, other: Triangle):
        self.next = other
        other.prev = self


def four_triangles(
    a: ValuedPoint,
    b: ValuedPoint,
    c: ValuedPoint,
    d: ValuedPoint,
    center: ValuedPoint,
):
    """a,b,c,d should be clockwise oriented, with center on the inside of that quad"""
    return (
        Triangle([a, b, center]),
        Triangle([b, c, center]),
        Triangle([c, d, center]),
        Triangle([d, a, center]),
    )


class Triangulator:
    """While triangulating, also compute the isolines.

    Divides each quad into 8 triangles from the quad's center. This simplifies
    adjacencies between triangles for the general case of multiresolution quadtrees.

    Based on Manson, Josiah, and Scott Schaefer. "Isosurfaces
    over simplicial partitions of multiresolution grids." Computer Graphics Forum.
    Vol. 29. No. 2. Oxford, UK: Blackwell Publishing Ltd, 2010.
    (https://people.engr.tamu.edu/schaefer/research/iso_simplicial.pdf), but this
    does not currently implement placing dual vertices based on the gradient.
    """

    def __init__(self, root: Cell, fn: Func):
        self.triangles: list[Triangle] = []
        self.hanging_next = {}
        self.root = root
        self.fn = fn

    def triangulate(self):
        self.triangulate_inside(self.root)
        return self.triangles

    def triangulate_inside(self, quad: Cell):
        if quad.children:
            for child in quad.children:
                self.triangulate_inside(child)
            self.triangulate_crossing_row(quad.children[0], quad.children[1])
            self.triangulate_crossing_row(quad.children[2], quad.children[3])
            self.triangulate_crossing_col(quad.children[0], quad.children[2])
            self.triangulate_crossing_col(quad.children[1], quad.children[3])

    def triangulate_crossing_row(self, a: Cell, b: Cell):
        """Quad b should be to the right (greater x values) than quad a"""
        if a.children and b.children:
            self.triangulate_crossing_row(a.children[1], b.children[0])
            self.triangulate_crossing_row(a.children[3], b.children[2])
        elif a.children:
            self.triangulate_crossing_row(a.children[1], b)
            self.triangulate_crossing_row(a.children[3], b)
        elif b.children:
            self.triangulate_crossing_row(a, b.children[0])
            self.triangulate_crossing_row(a, b.children[2])
        else:
            # a and b are minimal 2-cells
            face_dual_a = self.get_face_dual(a)
            face_dual_b = self.get_face_dual(b)
            # Add the four triangles from the centers of a and b to the shared edge between them
            if a.depth < b.depth:
                # b is smaller
                edge_dual = self.get_edge_dual(b.vertices[2], b.vertices[0])
                triangles = four_triangles(
                    b.vertices[2],
                    face_dual_b,
                    b.vertices[0],
                    face_dual_a,
                    edge_dual,
                )
            else:
                edge_dual = self.get_edge_dual(a.vertices[3], a.vertices[1])
                triangles = four_triangles(
                    a.vertices[3],
                    face_dual_b,
                    a.vertices[1],
                    face_dual_a,
                    edge_dual,
                )
            self.add_four_triangles(triangles)

    def triangulate_crossing_col(self, a: Cell, b: Cell):
        """Mostly a copy-paste of triangulate_crossing_row. For n-dimensions, want to pass a
        dir index into a shared triangulate_crossing_dir function instead"""
        if a.children and b.children:
            self.triangulate_crossing_col(a.children[2], b.children[0])
            self.triangulate_crossing_col(a.children[3], b.children[1])
        elif a.children:
            self.triangulate_crossing_col(a.children[2], b)
            self.triangulate_crossing_col(a.children[3], b)
        elif b.children:
            self.triangulate_crossing_col(a, b.children[0])
            self.triangulate_crossing_col(a, b.children[1])
        else:
            # a and b are minimal 2-cells
            face_dual_a = self.get_face_dual(a)
            face_dual_b = self.get_face_dual(b)
            # Add the four triangles from the centers of a and b to the shared edge between them
            if a.depth < b.depth:
                # b is smaller
                edge_dual = self.get_edge_dual(b.vertices[0], b.vertices[1])
                triangles = four_triangles(
                    b.vertices[0],
                    face_dual_b,
                    b.vertices[1],
                    face_dual_a,
                    edge_dual,
                )
            else:
                edge_dual = self.get_edge_dual(a.vertices[2], a.vertices[3])
                triangles = four_triangles(
                    a.vertices[2],
                    face_dual_b,
                    a.vertices[3],
                    face_dual_a,
                    edge_dual,
                )
            self.add_four_triangles(triangles)

    def add_four_triangles(
        self,
        triangles: tuple[Triangle, Triangle, Triangle, Triangle],
    ):
        for i in range(4):
            self.next_sandwich_triangles(
                triangles[i],
                triangles[(i + 1) % 4],
                triangles[(i + 2) % 4],
            )
        self.triangles.extend(triangles)

    def next_sandwich_triangles(self, a: Triangle, b: Triangle, c: Triangle):
        """Find the "next" triangle for the triangle b. See Triangle for a description of the curve orientation.

        We assume the triangles are oriented such that they share common vertices center←[2]≡b[2]≡c[2]
        and x←a[1]≡b[0], y←b[1]≡c[0]"""

        center = b.vertices[2]
        x = b.vertices[0]
        y = b.vertices[1]

        # Simple connections: inside the same four triangles
        # (Group 0 with negatives)
        if center.val > 0 >= y.val:
            b.set_next(c)
        # (Group 0 with negatives)
        if x.val > 0 >= center.val:
            b.set_next(a)

        # More difficult connections: complete a hanging connection
        # or wait for another triangle to complete this
        # We index using (double) the midpoint of the hanging edge
        id = (x.pos + y.pos).data.tobytes()

        # (Group 0 with negatives)
        if y.val > 0 >= x.val:
            if id in self.hanging_next:
                b.set_next(self.hanging_next[id])
            else:
                self.hanging_next[id] = b
        elif y.val <= 0 < x.val:
            if id in self.hanging_next:
                self.hanging_next[id].set_next(b)
            else:
                self.hanging_next[id] = b

    def get_edge_dual(self, p1: ValuedPoint, p2: ValuedPoint):
        """Returns the dual point on an edge p1--p2"""
        if (p1.val > 0) != (p1.val > 0):
            # The edge crosses the isoline, so take the midpoint
            return ValuedPoint.midpoint(p1, p2, self.fn)
        dt = 0.01
        # We intersect the planes with normals <∇f(p1), -1> and <∇f(p2), -1>
        # move slightly from p1 to p2. df = ∆f, so ∆f/∆t = 100*df1 near p1
        df1 = self.fn(p1.pos * (1 - dt) + p2.pos * dt)
        # move slightly from p2 to p1. df = ∆f, so ∆f/∆t = -100*df2 near p2
        df2 = self.fn(p1.pos * dt + p2.pos * (1 - dt))
        # (Group 0 with negatives)
        if (df1 > 0) == (df2 > 0):
            # The function either increases → ← or ← →, so a lerp would shoot out of bounds
            # Take the midpoint
            return ValuedPoint.midpoint(p1, p2, self.fn)
        else:
            # Increases → 0 → or ← 0 ←
            v1 = ValuedPoint(p1.pos, df1)
            v2 = ValuedPoint(p2.pos, df2)
            return ValuedPoint.intersectZero(v1, v2, self.fn)

    def get_face_dual(self, quad: Cell):
        # TODO: proper face dual
        return ValuedPoint.midpoint(quad.vertices[0], quad.vertices[-1], self.fn)


class CurveTracer:
    active_curve: list[Point]

    def __init__(self, triangles: list[Triangle], fn: Func):
        self.triangles = triangles
        self.fn = fn

    def trace(self) -> list[list[Point]]:
        curves: list[list[Point]] = []
        for triangle in self.triangles:
            if not triangle.visited and triangle.next is not None:
                self.active_curve = []
                self.march_triangle(triangle)
                # triangle.next is not None, so there should be at least one segment
                curves.append(self.active_curve)
        return [[v.pos for v in curve] for curve in curves]

    def march_triangle(self, triangle: Triangle):
        start_triangle = triangle
        closed_loop = False
        # Iterate backwards to the start of a connected curve
        while triangle.prev is not None:
            triangle = triangle.prev
            if triangle is start_triangle:
                closed_loop = True
                break
        while triangle is not None and not triangle.visited:
            for i in range(3):
                self.march_edge(triangle.vertices[i], triangle.vertices[(i + 1) % 3])
            triangle.visited = True
            triangle = triangle.next
        if closed_loop:
            # close back the loop
            self.active_curve.append(self.active_curve[0])

    def march_edge(self, p1: ValuedPoint, p2: ValuedPoint):
        # (Group 0 with negatives)
        if p1.val > 0 >= p2.val:
            intersection, is_zero = binary_search_zero(p1, p2, self.fn)
            if is_zero:
                self.active_curve.append(intersection)
