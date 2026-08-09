from typing import Any


class Positionable:
    __slots__ = ()

    def align_on_border(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def align_to(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apply_complex_function(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apply_function(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apply_function_to_position(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apply_matrix(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apply_over_attr_arrays(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apply_points_function(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apply_to_family(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def center(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def depth(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @depth.setter
    def depth(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def flip(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_bottom(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_boundary_point(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_bounding_box(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_center(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_center_of_mass(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_coord(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_corner(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_critical_point(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_depth(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_edge_center(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_extremum_along_dim(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_height(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_left(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_merged_array(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_midpoint(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_nadir(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_right(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_top(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_width(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_x(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_y(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_z(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_zenith(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def height(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @height.setter
    def height(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def is_off_screen(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def is_point_touching(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def length_over_dim(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def match_coord(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def match_depth(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def match_dim_size(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def match_height(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def match_width(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def match_x(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def match_y(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def match_z(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def move_to(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def next_to(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def point_from_proportion(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def pose_at_angle(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def proportion_from_point(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def reduce_across_dimension(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def rescale_to_fit(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def rotate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def rotate_about_origin(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def scale(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def scale_to_fit_depth(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def scale_to_fit_height(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def scale_to_fit_width(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def set_coord(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def set_depth(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def set_height(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def set_width(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def set_x(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def set_y(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def set_z(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def shift(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def shift_onto_screen(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stretch(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stretch_about_point(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stretch_to_fit_depth(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stretch_to_fit_height(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stretch_to_fit_width(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def to_corner(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def to_edge(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def width(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @width.setter
    def width(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
