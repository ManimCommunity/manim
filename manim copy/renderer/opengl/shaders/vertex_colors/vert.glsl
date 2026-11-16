#version 330

uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;
in vec4 in_vert;
in vec4 in_color;
out vec4 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * in_vert;
}
