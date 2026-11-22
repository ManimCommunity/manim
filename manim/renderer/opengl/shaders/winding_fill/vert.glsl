#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in vec2 corner_uv;
in float fill_all;
in vec4 color;

out vec2 uv_coords;
out float v_fill_all;
out vec4 v_color;

#include ../include/position_point_into_frame.glsl
#include ../include/get_gl_Position.glsl

void main(){
    uv_coords = corner_uv;
    v_fill_all = fill_all;
    v_color = color;
    gl_Position = get_gl_Position(position_point_into_frame(point));
}
