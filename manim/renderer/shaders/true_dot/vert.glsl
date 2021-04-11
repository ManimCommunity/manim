#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in float radius;
in vec4 color;

out vec3 v_point;
out float v_radius;
out vec4 v_color;

#include ../include/position_point_into_frame.glsl

void main(){
    v_point = position_point_into_frame(point);
    v_radius = radius;
    v_color = color;
}
