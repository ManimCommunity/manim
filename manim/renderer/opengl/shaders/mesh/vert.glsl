#version 330

#include ../include/camera_uniform_declarations.glsl

in vec3 point;
in vec3 normal;
in vec4 color;
in vec2 uv;

out vec3 v_scene_point;
out vec3 v_scene_normal;
out vec4 v_color;
out vec2 v_uv;

#include ../include/position_point_into_frame.glsl
#include ../include/get_gl_Position.glsl

void main(){
    v_scene_point = point;
    v_scene_normal = normal;
    v_color = color;
    v_uv = uv;
    gl_Position = get_gl_Position(position_point_into_frame(point));
}
