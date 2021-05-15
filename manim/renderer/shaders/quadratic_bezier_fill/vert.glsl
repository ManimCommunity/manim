#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in vec3 unit_normal;
in vec4 color;
in float vert_index;

out vec3 bp;  // Bezier control point
out vec3 v_global_unit_normal;
out vec4 v_color;
out float v_vert_index;

// Analog of import for manim only
#include ../include/position_point_into_frame.glsl

void main(){
    bp = position_point_into_frame(point.xyz);
    v_global_unit_normal = rotate_point_into_frame(unit_normal.xyz);
    v_color = color;
    v_vert_index = vert_index;
}
