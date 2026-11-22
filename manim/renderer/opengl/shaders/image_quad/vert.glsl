#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in vec2 uv;

out vec2 v_uv;

#include ../include/position_point_into_frame.glsl
#include ../include/get_gl_Position.glsl

void main(){
    v_uv = uv;
    gl_Position = get_gl_Position(position_point_into_frame(point));
}
