#version 330

in vec3 in_vert;
uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;

void main() {
    gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(in_vert, 1.0);
}
