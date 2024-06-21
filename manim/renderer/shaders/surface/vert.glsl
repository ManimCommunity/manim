#version 330

in vec3 point;
in vec3 du_point;
in vec3 dv_point;
in vec4 color;

out vec3 xyz_coords;
out vec3 v_normal;
out vec4 v_color;

#include "../include/get_gl_Position.glsl"
#include "../include/get_rotated_surface_unit_normal_vector.glsl"
#include "../include/position_point_into_frame.glsl"

void main()
{
    xyz_coords = position_point_into_frame(point);
    v_normal = get_rotated_surface_unit_normal_vector(point, du_point, dv_point);
    v_color = color;
    gl_Position = get_gl_Position(xyz_coords);
}
