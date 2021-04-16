from copy import copy
from typing import Hashable, List, Tuple, Union

import numpy as np

from .geometry import Polygon
from .graph import Graph
from .three_dimensions import Dot3D
from .types.vectorized_mobject import VMobject, VGroup

__all__ = ["Polyhedra", "Tetrahedron", "Octahedron"]

class Polyhedra(VGroup):
    """An abstract polyhedra class.
    """
    def __init__(
        self,
        vertices: List[np.ndarray],
        faces: List[List[Hashable]],
        faces_config = {},
        graph_config = {}
    ):
        VGroup.__init__(self)
        self.vertices = vertices
        self.vertex_indices = list(range(len(self.vertices)))
        self.layout = dict(enumerate(self.vertices))
        self.faces = faces
        self.face_coords = [[self.layout[j] for j in i] for i in faces]
        self.edges = self.get_edges(self.faces)
        self.add(self.create_faces(self.face_coords))
        self.add(Graph(self.vertex_indices, self.edges, layout=self.layout, vertex_type=Dot3D))

    def get_edges(self, faces):
        """Creates list of cyclic pairwise tuples."""
        edges = []
        for face in faces:
            edges += zip(face, face[1:] + face[:1])
        return edges

    def create_faces(self, face_coords):
        face_group = VGroup()
        for face in face_coords:
            face_group.add(Polygon(*face, fill_opacity=0.5, shade_in_3d=True))
        return face_group

class Tetrahedron(Polyhedra):
    """A tetrahedron."""
    def __init__(
        self,
        side_length=1
    ):
        unit = side_length*np.sqrt(2)/4
        Polyhedra.__init__(
            self,
            vertices = [
                np.array([unit, unit, unit]),
                np.array([unit, -unit, -unit]),
                np.array([-unit, unit, -unit]),
                np.array([-unit, -unit, unit])
            ],
            faces = [
                [0, 1, 2],
                [3, 0, 2],
                [0, 1, 3],
                [3, 1, 2]
            ]
        )

class Octahedron(Polyhedra):
    """An octahedron."""
    def __init__(
        self,
        side_length=1
    ):
        unit = side_length * np.sqrt(2) / 2
        Polyhedra.__init__(
            self,
            vertices = [
                np.array([unit, 0, 0]),
                np.array([-unit, 0, 0]),
                np.array([0, unit, 0]),
                np.array([0, -unit, 0]),
                np.array([0, 0, unit]),
                np.array([0, 0, -unit])
            ],
            faces = [
                [2, 4, 1],
                [0, 4, 2],
                [4, 3, 0],
                [1, 3, 4],
                [3, 5, 0],
                [1, 5, 3],
                [2, 5, 1],
                [0, 5, 2]
            ]
        )