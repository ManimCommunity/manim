#version 330

#include ../include/camera_uniform_declarations.glsl

in vec4 color;
in float fill_all;  // Either 0 or 1e
in float uv_anti_alias_width;

in float orientation;
in vec2 uv_coords;
in float bezier_degree;

out vec4 frag_color;

#define ANTI_ALIASING

float sdf(){
    if(bezier_degree < 2){
        return abs(uv_coords[1]);
    }
    vec2 p = uv_coords;
    float sgn = orientation;
    float q = (p.x * p.x - p.y);
#ifdef ANTI_ALIASING
    return sgn * q / sqrt(dFdx(q) * dFdx(q) + dFdy(q) * dFdy(q));
#endif
#ifndef ANTI_ALIASING
    return -sgn * q;
#endif
}


void main() {
    if (color.a == 0) discard;
    frag_color = color;
    if (fill_all == 1.0) return;
#ifdef ANTI_ALIASING
    frag_color.a *= 0.5 - sdf(); // Anti-aliasing
#endif
#ifndef ANTI_ALIASING
    frag_color.a *= float(sdf() > 0); // No anti-aliasing
#endif
    if (frag_color.a <= 0.0)
    {
        discard;
    }
}
