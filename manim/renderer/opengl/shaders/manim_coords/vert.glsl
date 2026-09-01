#version 330

in vec4 in_vert;
uniform mat4 u_model_view_matrix;
uniform mat4 u_projection_matrix;

void main() {
    vec4 camera_space_vertex = u_model_view_matrix * in_vert;
    vec4 clip_space_vertex = u_projection_matrix * camera_space_vertex;
    gl_Position = clip_space_vertex;
}
