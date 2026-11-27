#version 330

// One oversized triangle covering clip space; the fragment shader reads the
// accumulation buffer by pixel coordinate, so no texture coordinates are needed.
in vec2 corner;

void main(){
    gl_Position = vec4(corner, 0.0, 1.0);
}
