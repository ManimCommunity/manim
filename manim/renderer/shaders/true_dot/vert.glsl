#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in vec4 color;

uniform float point_width;

out vec3 v_point;
out float v_point_width;
out vec4 v_color;

#include ../include/position_point_into_frame.glsl

void main(){
    v_point = position_point_into_frame(point);
    v_point_width = point_width;
    v_color = color;
}
