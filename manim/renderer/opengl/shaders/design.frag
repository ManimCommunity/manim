#version 330

out vec4 frag_color;

void main() {
  vec2 st = gl_FragCoord.xy / vec2(854, 360);
  vec3 color = vec3(0.0);

  st *= 3.0;
  st = fract(st);

  color = vec3(st, 0.0);

  frag_color = vec4(color, 1.0);
}
