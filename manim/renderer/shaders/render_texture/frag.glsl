#version 330

uniform sampler2D tex;
in vec2 f_uv;

out vec4 frag_color;

void main()
{
    frag_color = texture(tex, f_uv);
    // frag_color.a = 1.0;
}
