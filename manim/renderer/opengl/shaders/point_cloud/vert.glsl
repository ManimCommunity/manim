#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in vec4 color;

uniform float point_size;

out vec4 v_color;

#include ../include/position_point_into_frame.glsl
#include ../include/get_gl_Position.glsl

void main(){
    v_color = color;
    gl_PointSize = point_size;
    gl_Position = get_gl_Position(position_point_into_frame(point));
}
