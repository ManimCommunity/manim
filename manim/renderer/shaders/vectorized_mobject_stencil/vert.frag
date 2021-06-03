#version 330

uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;
in vec3 in_vert;
in vec2 texture_coords;
out vec2 v_texture_coords;

void main() {
    v_texture_coords = texture_coords;
    gl_Position = u_projection_matrix * u_view_matrix * vec4(in_vert, 1.0);
}
