#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;

// Analog of import for manim only
#include ../include/get_gl_Position.glsl
#include ../include/position_point_into_frame.glsl

void main(){
    gl_Position = get_gl_Position(position_point_into_frame(point));
}
