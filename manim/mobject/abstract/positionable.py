from typing import Self

import numpy as np

from manim import ORIGIN
from manim.typing import Point3D, Point3D_Array, Point3DLike, Point3DLike_Array


class Positionable:
    def __init__(self, *points: Point3DLike):
        self.points: Point3D_Array = np.asarray(points)

    def get_points(self) -> Point3D_Array:
        if len(self.points) == 0:
            return np.array([ORIGIN])
        return self.points

    def set_points(self, points: Point3DLike_Array) -> Self:
        self.points = np.asarray(points)
        return self

    # Getter
    def length_over_dim(self, dim: int) -> float:
        points = self.get_points()
        values = points[:, dim]
        return values.max() - values.min()

    @property
    def width(self) -> float:
        return self.length_over_dim(0)

    @property
    def height(self) -> float:
        return self.length_over_dim(1)

    @property
    def depth(self) -> float:
        return self.length_over_dim(2)

    def get_bottom(self) -> Point3D:
        points = self.get_points()
        x = (points[:, 0].min() + points[:, 0].max()) / 2
        y = points[:, 1].min()
        z = (points[:, 2].min() + points[:, 2].max()) / 2
        return np.array([x, y, z])

    # Helper Methods
    # align_on_border
    # align_to
    # apply_complex_function
    # apply_function
    # apply_function_to_position
    # apply_matrix
    # apply_over_attr_arrays
    # apply_points_function
    # apply_points_function_about_point
    # center
    # flip
    # get_array_attrs
    # get_bottom
    # get_boundary_point
    # get_bounding_box
    # get_bounding_box_point
    # get_center
    # get_center_of_mass
    # get_continuous_bounding_box_point
    # get_coord
    # get_corner
    # get_critical_point
    # get_depth
    # get_edge_center
    # get_extremum_along_dim
    # get_height
    # get_left
    # get_midpoint
    # get_nadir
    # get_points_defining_boundary
    # get_right
    # get_top
    # get_width
    # get_x
    # get_y
    # get_z
    # get_zenith
    # is_off_screen
    # is_point_touching
    # length_over_dim
    # match_coord
    # match_depth
    # match_dim_size
    # match_height
    # match_width
    # match_x
    # match_y
    # match_z
    # move_to
    # next_to
    # pfp
    # point_from_proportion
    # pose_at_angle
    # proportion_from_point
    # reduce_across_dimension
    # rescale_to_fit
    # rotate
    # rotate_about_origin
    # scale
    # scale_to_fit_depth
    # scale_to_fit_height
    # scale_to_fit_width
    # set_coord
    # set_depth
    # set_height
    # set_width
    # set_x
    # set_y
    # set_z
    # shift
    # shift_onto_screen
    # stretch
    # stretch_about_point
    # stretch_to_fit_depth
    # stretch_to_fit_height
    # stretch_to_fit_width
    # to_corner
    # to_edge


if __name__ == "__main__":
    mob = Positionable(*[(0, -1, 0), (0, -2, 1)])
    print(mob.get_bottom())
