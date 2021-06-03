#version 330

uniform sampler2D Texture0;
uniform sampler2D Texture1;
uniform sampler2D Texture2;
uniform sampler2D Texture3;
uniform sampler2D Texture4;
uniform vec4 color;

in vec2 v_text;
in float v_texindex;
out vec4 f_color;

void main() {
    // TODO: Find a way to pass a variable number of textures.
    if (v_texindex == 0.0) {
        f_color = vec4(texture(Texture0, v_text).rgb * 255.0, 1.0);
    } else if (v_texindex == 1.0) {
        f_color = vec4(texture(Texture1, v_text).rgb * 255.0, 1.0);
    } else if (v_texindex == 2.0) {
        f_color = vec4(texture(Texture2, v_text).rgb * 255.0, 1.0);
    } else if (v_texindex == 3.0) {
        f_color = vec4(texture(Texture3, v_text).rgb * 255.0, 1.0);
    } else {
        f_color = vec4(texture(Texture4, v_text).rgb * 255.0, 1.0);
    }

    if (mod(f_color[0], 2) == 1) {
        f_color = color;
    } else {
        discard;
    }
}
