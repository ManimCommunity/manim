# to support Quad type inside Quad
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import numpy as np

Point = np.ndarray

# TODO: actual x, y tolerance
TOL = 0.002


@dataclass
class ValuedPoint:
    """A position associated with the corresponding function value"""

    pos: Point
    val: float = None

    def calc(self, fn: Func):
        self.val = fn(self.pos)
        return self

    def __repr__(self):
        return f"({self.pos[0]},{self.pos[1]}; {self.val})"

    @staticmethod
    def midpoint(p1: ValuedPoint, p2: ValuedPoint, fn: Func):
        mid = (p1.pos + p2.pos) / 2
        return ValuedPoint(mid, fn(mid))

    @staticmethod
    def intersectZero(p1: ValuedPoint, p2: ValuedPoint, fn: Func):
        """Find the point on line p1--p2 with value 0"""
        denom = p1.val - p2.val
        k1 = -p2.val / denom
        k2 = p1.val / denom
        pt = k1 * p1.pos + k2 * p2.pos
        return ValuedPoint(pt, fn(pt))


Func = Callable[[Point], float]


def plot_implicit(
    fn: Func,
    pmin: Point,
    pmax: Point,
    min_depth: int = 5,
    max_quads: int = 10000,
):
    """Returns the curve representing fn(x,y)=0 on pmin[0] ≤ x ≤ pmax[0] ∩ pmin[1] ≤ y ≤ pmax[1]"""
    quadtree = build_quad_tree(fn, pmin, pmax, min_depth, max_quads)
    triangles = Triangulator(quadtree, fn).triangulate()
    return CurveTracer(triangles, fn).trace()


def vertices_from_extremes(pmin: Point, pmax: Point, fn: Func):
    """Requires pmin.x ≤ pmax.x, pmin.y ≤ pmax.y"""
    w = pmax - pmin
    return [
        ValuedPoint(np.array([pmin[d] + (i >> d & 1) * w[d] for d in range(2)])).calc(
            fn,
        )
        for i in range(4)
    ]


@dataclass
class Quad:
    # In 2 dimensions, vertices = [bottom-left, bottom-right, top-left, top-right] points
    vertices: list[ValuedPoint]
    depth: int
    # Children go in same order: bottom-left, bottom-right, top-left, top-right
    children: list[Quad]

    def compute_children(self, fn: Func):
        assert self.children == []
        for vertex in self.vertices:
            pmin = (self.vertices[0].pos + vertex.pos) / 2
            pmax = (self.vertices[-1].pos + vertex.pos) / 2
            vertices = vertices_from_extremes(pmin, pmax, fn)
            new_quad = Quad(vertices, self.depth + 1, [])
            self.children.append(new_quad)


def should_descend_deep_quad(quad: Quad):
    if np.max(quad.vertices[-1].pos - quad.vertices[0].pos) < TOL:
        return False
    elif all(np.isnan(v.val) for v in quad.vertices):
        # in a region where the function is undefined
        return False
    elif any(np.isnan(v.val) for v in quad.vertices):
        # straddling defined and undefined
        return True
    else:
        # simple approach: only descend if we cross the isoline
        # TODO: This could very much be improved, e.g. by incorporating gradient or second-derivative
        # tests, etc., to cancel descending in approximately linear regions
        return any(
            np.sign(v.val) != np.sign(quad.vertices[0].val) for v in quad.vertices[1:]
        )


def build_quad_tree(
    fn: Func,
    pmin: Point,
    pmax: Point,
    min_depth: int,
    max_quads: int,
) -> Quad:
    # min_depth takes precedence over max_quads
    max_quads = max(4 ** min_depth, max_quads)
    vertices = vertices_from_extremes(pmin, pmax, fn)
    current_quad = root = Quad(vertices, 0, [])
    quad_queue = deque([root])
    leaf_count = 1

    while len(quad_queue) > 0 and leaf_count < max_quads:
        current_quad = quad_queue.popleft()
        if current_quad.depth < min_depth or should_descend_deep_quad(current_quad):
            current_quad.compute_children(fn)
            quad_queue.extend(current_quad.children)
            # add 4 for the new quads, subtract 1 for the old quad not being a leaf anymore
            leaf_count += 3
    return root


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

    triangles: list[Triangle] = []
    hanging_next = {}

    def __init__(self, root: Quad, fn: Func):
        self.root = root
        self.fn = fn

    def triangulate(self):
        self.triangulate_inside(self.root)
        return self.triangles

    def triangulate_inside(self, quad: Quad):
        if quad.children:
            for child in quad.children:
                self.triangulate_inside(child)
            self.triangulate_crossing_row(quad.children[0], quad.children[1])
            self.triangulate_crossing_row(quad.children[2], quad.children[3])
            self.triangulate_crossing_col(quad.children[0], quad.children[2])
            self.triangulate_crossing_col(quad.children[1], quad.children[3])

    def triangulate_crossing_row(self, a: Quad, b: Quad):
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

    def triangulate_crossing_col(self, a: Quad, b: Quad):
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

    def get_face_dual(self, quad: Quad):
        # TODO: proper face dual
        return ValuedPoint.midpoint(quad.vertices[0], quad.vertices[-1], self.fn)


def binary_search_zero(p1: ValuedPoint, p2: ValuedPoint, fn: Func):
    """Returns a pair `(point, is_zero: bool)`

    Use is_zero to make sure it's not an asymptote like at x=0 on f(x,y) = 1/(xy) - 1"""
    if np.max(np.abs(p2.pos - p1.pos)) < TOL:
        # Binary search stop condition: too small to matter
        pt = ValuedPoint.intersectZero(p1, p2, fn)
        is_zero = pt.val == 0 or (
            np.sign(pt.val - p1.val) == np.sign(p2.val - pt.val)
            and np.abs(pt.val < TOL)
        )
        return pt, is_zero
    else:
        # binary search
        mid = ValuedPoint.midpoint(p1, p2, fn)
        if mid.val == 0:
            return mid, True
        # (Group 0 with negatives)
        elif (mid.val > 0) == (p1.val > 0):
            return binary_search_zero(mid, p2, fn)
        else:
            return binary_search_zero(p1, mid, fn)


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
