#version 330

in vec4 v_color;
in vec2 v_texture_coords;
flat in int v_texture_mode;

out vec4 frag_color;

void main() {
    float curve_func = v_texture_coords[0] * v_texture_coords[0] - v_texture_coords[1];
    if (v_texture_mode * curve_func >= 0.0) {
        frag_color = v_color;
    } else {
        discard;
    }
}
