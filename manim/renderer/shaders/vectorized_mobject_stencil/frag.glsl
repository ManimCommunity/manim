#version 330

in vec2 v_texture_coords;
out vec4 f_color;

void main() {
    if (v_texture_coords[0] * v_texture_coords[0] - v_texture_coords[1] < 0) {
        f_color = vec4(1/255.0, 1/255.0, 1/255.0, 1.0);
    } else {
        discard;
    }
}
