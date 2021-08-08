#version 330

in vec3 point;
in vec4 color;
out vec4 v_color;

uniform mat4 u_model_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;
uniform float point_width;

void main() {
    gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(point, 1.0);
    v_color = color;
    gl_PointSize = point_width;
}
