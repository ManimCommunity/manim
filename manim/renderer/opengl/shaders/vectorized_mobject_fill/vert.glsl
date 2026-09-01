#version 330

uniform mat4 u_model_view_matrix;
uniform mat4 u_projection_matrix;

in vec3 in_vert;
in vec4 in_color;
in vec2 texture_coords;
in int texture_mode;

out vec4 v_color;
out vec2 v_texture_coords;
flat out int v_texture_mode;

void main() {
    v_color = in_color;
    v_texture_coords = texture_coords;
    v_texture_mode = texture_mode;
    gl_Position = u_projection_matrix * u_model_view_matrix * vec4(in_vert, 1.0);
}
