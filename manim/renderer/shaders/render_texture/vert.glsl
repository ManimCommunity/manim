#version 330

in vec2 pos;
in vec2 uv;

out vec2 f_uv;

void main()
{
    gl_Position = vec4(pos, 0.0, 1.0);
    f_uv = uv;
}
