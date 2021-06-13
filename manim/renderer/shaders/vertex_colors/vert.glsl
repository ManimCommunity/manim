#version 330

in vec4 in_vert;
in vec4 in_color;
out vec4 v_color;
uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

void main() {
    v_color = in_color;
    gl_Position = projection_matrix *  view_matrix *  model_matrix * in_vert;
}
