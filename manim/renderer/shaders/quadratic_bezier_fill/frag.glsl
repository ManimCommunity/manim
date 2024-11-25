#version 330

#include "../include/camera_uniform_declarations.glsl"

uniform vec2 pixel_shape;
uniform float index;

in vec4 color;
in float fill_all; // Either 0 or 1e

in float orientation;
in vec2 uv_coords;
in float bezier_degree;

uniform sampler2D stencil_texture;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 stencil_value;

#define ANTI_ALIASING

float sdf()
{
    if (bezier_degree < 2)
    {
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

void main()
{
    gl_FragDepth = gl_FragCoord.z;
    if (color.a == 0)
        discard;

    float previous_index =
        texture2D(stencil_texture, vec2(gl_FragCoord.x / pixel_shape.x, gl_FragCoord.y / pixel_shape.y)).r;

    // Check if we are behind another fill and if yes discard the current fragment
    if (previous_index > index)
    {
        discard;
    }
    // If we are on top of a previously drawn fill we need to shift ourselves forward by the index amount to compensate
    // for different shifting and avoid z_fighting
    if (previous_index < index && previous_index != 0)
    {
        gl_FragDepth = gl_FragCoord.z - index / 1000.0;
    }
    stencil_value.rgb = vec3(index);
    stencil_value.a = 1.0;
    frag_color = color;
    if (fill_all == 1.0)
        return;
#ifdef ANTI_ALIASING
    float fac = max(0.0, min(1.0, 0.5 - sdf()));
    frag_color.a *= fac; // Anti-aliasing
#endif
#ifndef ANTI_ALIASING
    frag_color.a *= float(sdf() > 0); // No anti-aliasing
#endif
    if (frag_color.a <= 0.0)
    {
        discard;
    }
}
