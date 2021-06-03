#version 330

in vec2 in_vert;
in vec2 in_texcoord_0;
in float in_texindex;
out vec2 v_text;
out float v_texindex;

void main() {
    v_text = in_texcoord_0;
    v_texindex = in_texindex;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
