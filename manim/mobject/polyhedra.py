from copy import copy
from typing import Hashable, List, Tuple, Union

import numpy as np

from .geometry import Polygon
from .graph import Graph
from .three_dimensions import Dot3D
from .types.vectorized_mobject import VMobject, VGroup
from .mobject_update_utils import always

__all__ = ["Polyhedron", "Tetrahedron", "Octahedron", "Icosahedron", "Dodecahedron"]

class Polyhedron(VGroup):
    """An abstract polyhedra class.
    """
    def __init__(
        self,
        vertex_coords: List[np.ndarray],
        faces_list: List[List[Hashable]],
        faces_config = {},
        graph_config = {}
    ):
        VGroup.__init__(self)
        self.vertex_coords = vertex_coords
        self.vertex_indices = list(range(len(self.vertex_coords)))
        self.layout = dict(enumerate(self.vertex_coords))
        self.faces_list = faces_list
        self.face_coords = [[self.layout[j] for j in i] for i in faces_list]
        self.edges = self.get_edges(self.faces_list)
        self.faces = self.create_faces(self.face_coords)
        self.graph = Graph(self.vertex_indices, self.edges, layout=self.layout, vertex_type=Dot3D)
        self.add(self.faces, self.graph)
        self.faces.add_updater(self.update_faces)

    def get_edges(self, faces_list):
        """Creates list of cyclic pairwise tuples."""
        edges = []
        for face in faces_list:
            edges += zip(face, face[1:] + face[:1])
        return edges

    def create_faces(self, face_coords):
        face_group = VGroup()
        for face in face_coords:
            face_group.add(Polygon(*face, fill_opacity=0.5, shade_in_3d=True))
        return face_group

    def update_faces(self, m):
        face_coords = self.extract_face_coords()
        for face, points in zip(m, face_coords):
            face.set_points_as_corners([*points, points[0]])

    def extract_face_coords(self):
        layout = self.graph._layout
        return [[layout[j] for j in i] for i in self.faces_list]


class Tetrahedron(Polyhedron):
    """A tetrahedron."""
    def __init__(
        self,
        side_length=1
    ):
        unit = side_length*np.sqrt(2)/4
        Polyhedron.__init__(
            self,
            vertex_coords = [
                np.array([unit, unit, unit]),
                np.array([unit, -unit, -unit]),
                np.array([-unit, unit, -unit]),
                np.array([-unit, -unit, unit])
            ],
            faces_list = [
                [0, 1, 2],
                [3, 0, 2],
                [0, 1, 3],
                [3, 1, 2]
            ]
        )

class Octahedron(Polyhedron):
    """An octahedron."""
    def __init__(
        self,
        side_length=1
    ):
        unit = side_length * np.sqrt(2) / 2
        Polyhedron.__init__(
            self,
            vertex_coords = [
                np.array([unit, 0, 0]),
                np.array([-unit, 0, 0]),
                np.array([0, unit, 0]),
                np.array([0, -unit, 0]),
                np.array([0, 0, unit]),
                np.array([0, 0, -unit])
            ],
            faces_list = [
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

class Icosahedron(Polyhedron):
    """An icosahedron."""
    def __init__(
        self,
        side_length
    ):
        unit_a = side_length * ((1 + np.sqrt(5))/4)
        unit_b = side_length * (1/2)
        Polyhedron.__init__(
            self,
            vertex_coords = [
                np.array([0, unit_b, unit_a]),
                np.array([0, -unit_b, unit_a]),
                np.array([0, unit_b, -unit_a]),
                np.array([0, -unit_b, -unit_a]),
                np.array([unit_b, unit_a, 0]),
                np.array([unit_b, -unit_a, 0]),
                np.array([-unit_b, unit_a, 0]),
                np.array([-unit_b, -unit_a, 0]),
                np.array([unit_a, 0, unit_b]),
                np.array([unit_a, 0, -unit_b]),
                np.array([-unit_a, 0, unit_b]),
                np.array([-unit_a, 0, -unit_b]),
            ],
            faces_list = [
                [1, 8, 0],
                [1, 5, 7],
                [8, 5, 1],
                [7, 3, 5],
                [5, 9, 3],
                [8, 9, 5],
                [3, 2, 9],
                [9, 4, 2],
                [8, 4, 9],
                [0, 4, 8],
                [6, 4, 0],
                [6, 2, 4],
                [11, 2, 6],
                [3, 11, 2],
                [0, 6, 10],
                [10, 1, 0],
                [10, 7, 1],
                [11, 7, 3],
                [10, 11, 7],
                [10, 11, 6]
            ]
        )

class Dodecahedron(Polyhedron):
    """A dodecahedron."""
    def __init__(
        self,
        side_length=1
    ):
        unit_a = side_length * ((1 + np.sqrt(5))/4)
        unit_b = side_length * ((3 + np.sqrt(5))/4)
        unit_c = side_length * (1/2)
        Polyhedron.__init__(
            self,
            vertex_coords = [
                np.array([unit_a, unit_a, unit_a]),
                np.array([unit_a, unit_a, -unit_a]),
                np.array([unit_a, -unit_a, unit_a]),
                np.array([unit_a, -unit_a, -unit_a]),
                np.array([-unit_a, unit_a, unit_a]),
                np.array([-unit_a, unit_a, -unit_a]),
                np.array([-unit_a, -unit_a, unit_a]),
                np.array([-unit_a, -unit_a, -unit_a]),
                np.array([0, unit_c, unit_b]),
                np.array([0, unit_c, -unit_b]),
                np.array([0, -unit_c, -unit_b]),
                np.array([0, -unit_c, unit_b]),
                np.array([unit_c, unit_b, 0]),
                np.array([-unit_c, unit_b, 0]),
                np.array([unit_c, -unit_b, 0]),
                np.array([-unit_c, -unit_b, 0]),
                np.array([unit_b, 0, unit_c]),
                np.array([-unit_b, 0, unit_c]),
                np.array([unit_b, 0, -unit_c]),
                np.array([-unit_b, 0, -unit_c]),                                              
            ],
            faces_list = [
                [18, 16, 0, 12, 1],
                [3, 18, 16, 2, 14],
                [3, 10, 9, 1, 18],
                [1, 9, 5, 13, 12],
                [0, 8, 4, 13, 12],
                [2, 16, 0, 8, 11],
                [4, 17, 6, 11, 8],
                [17, 19, 5, 13, 4],
                [19, 7, 15, 6, 17],
                [6, 15, 14, 2, 11],
                [19, 5, 9, 10, 7],
                [7, 10, 3, 14, 15]
            ]
        )