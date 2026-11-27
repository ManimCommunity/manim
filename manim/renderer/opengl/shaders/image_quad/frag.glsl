#version 330

uniform sampler2D image;

in vec2 v_uv;

out vec4 frag_color;

void main(){
    frag_color = texture(image, v_uv);
}
