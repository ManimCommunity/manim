# to support Cell type inside Cell
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generator, List

import numpy as np

from .point import TOL, Func, Point, ValuedPoint


def vertices_from_extremes(dim: int, pmin: Point, pmax: Point, fn: Func):
    """Requires pmin.x ≤ pmax.x, pmin.y ≤ pmax.y"""
    w = pmax - pmin
    return [
        ValuedPoint(np.array([pmin[d] + (i >> d & 1) * w[d] for d in range(dim)])).calc(
            fn,
        )
        for i in range(1 << dim)
    ]


@dataclass
class MinimalCell:
    dim: int
    # In 2 dimensions, vertices = [bottom-left, bottom-right, top-left, top-right] points
    vertices: list[ValuedPoint]

    def get_subcell(self, axis: int, dir: int):
        """Given an n-cell, this returns an (n-1)-cell (with half the vertices)"""
        m = 1 << axis
        return MinimalCell(
            self.dim - 1,
            [v for i, v in enumerate(self.vertices) if (i & m > 0) == dir],
        )

    def get_dual(self, fn: Func):
        return ValuedPoint.midpoint(self.vertices[0], self.vertices[-1], fn)


@dataclass
class Cell(MinimalCell):
    depth: int
    # Children go in same order: bottom-left, bottom-right, top-left, top-right
    children: list[Cell]
    parent: Cell
    child_direction: int

    def compute_children(self, fn: Func):
        assert self.children == []
        for i, vertex in enumerate(self.vertices):
            pmin = (self.vertices[0].pos + vertex.pos) / 2
            pmax = (self.vertices[-1].pos + vertex.pos) / 2
            vertices = vertices_from_extremes(self.dim, pmin, pmax, fn)
            new_quad = Cell(self.dim, vertices, self.depth + 1, [], self, i)
            self.children.append(new_quad)

    def get_leaves_in_direction(self, axis: int, dir: int) -> Generator[Cell]:
        """
        Axis = 0,1,2,etc for x,y,z,etc.
        Dir = 0 for -x, 1 for +x.
        """
        if self.children:
            m = 1 << axis
            for i in range(1 << self.dim):
                if (i & m > 0) == dir:
                    yield from self.children[i].get_leaves_in_direction(axis, dir)
        else:
            yield self

    def walk_in_direction(self, axis: int, dir: int) -> Cell:
        """
        Same arguments as get_leaves_in_direction.

        Returns the quad (with depth <= self.depth) that shares a (dim-1)-cell
        with self, where that (dim-1)-cell is the side of self defined by
        axis and dir.
        """
        m = 1 << axis
        if (self.child_direction & m > 0) == dir:
            # on the right side of the parent cell and moving right (or analogous)
            # so need to go up through the parent's parent
            if self.parent is None:
                return None
            parent_walked = self.parent.walk_in_direction(axis, dir)
            if parent_walked and parent_walked.children:
                # end at same depth
                return parent_walked.children[self.child_direction ^ m]
            else:
                # end at lesser depth
                return parent_walked
        else:
            if self.parent is None:
                return None
            return self.parent.children[self.child_direction ^ m]

    def walk_leaves_in_direction(self, axis: int, dir: int):
        walked = self.walk_in_direction(axis, dir)
        if walked is not None:
            yield from walked.get_leaves_in_direction(axis, dir)
        else:
            yield None


def should_descend_deep_cell(cell: Cell):
    if np.max(cell.vertices[-1].pos - cell.vertices[0].pos) < TOL:
        return False
    elif all(np.isnan(v.val) for v in cell.vertices):
        # in a region where the function is undefined
        return False
    elif any(np.isnan(v.val) for v in cell.vertices):
        # straddling defined and undefined
        return True
    else:
        # simple approach: only descend if we cross the isoline
        # TODO: This could very much be improved, e.g. by incorporating gradient or second-derivative
        # tests, etc., to cancel descending in approximately linear regions
        return any(
            np.sign(v.val) != np.sign(cell.vertices[0].val) for v in cell.vertices[1:]
        )


def build_tree(
    dim: int,
    fn: Func,
    pmin: Point,
    pmax: Point,
    min_depth: int,
    max_cells: int,
) -> Cell:
    branching_factor = 1 << dim
    # min_depth takes precedence over max_quads
    max_cells = max(branching_factor ** min_depth, max_cells)
    vertices = vertices_from_extremes(dim, pmin, pmax, fn)
    # root's childDirection is 0, even though none is reasonable
    current_quad = root = Cell(dim, vertices, 0, [], None, 0)
    quad_queue = deque([root])
    leaf_count = 1

    while len(quad_queue) > 0 and leaf_count < max_cells:
        current_quad = quad_queue.popleft()
        if current_quad.depth < min_depth or should_descend_deep_cell(current_quad):
            current_quad.compute_children(fn)
            quad_queue.extend(current_quad.children)
            # add 4 for the new quads, subtract 1 for the old quad not being a leaf anymore
            leaf_count += branching_factor - 1
    return root
