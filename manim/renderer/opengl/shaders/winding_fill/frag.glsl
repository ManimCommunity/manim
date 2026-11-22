#version 330

/*
Accumulates the signed coverage of one group of filled paths. Nothing here
decides what is inside a path: the winding number does, summed from which way
each triangle faces once projected, so no triangulation is needed.

Colour is premultiplied by that signed coverage, letting paths of one group
differ in colour and still be resolved from a single buffer wherever they do
not overlap each other. Opacity is accumulated separately, weighted by the same
coverage: a pixel covered twice by one path must still get that path's opacity,
so the resolve step divides it back out instead of adding it up.
*/

in vec2 uv_coords;
in float v_fill_all;
in vec4 v_color;

layout(location = 0) out vec4 coverage;
layout(location = 1) out float weighted_opacity;

void main(){
    float value = 1.0;
    if (v_fill_all < 0.5) {
        // The triangle hugging a curve keeps only the side where the parabola
        // y = x^2 encloses the fragment. Filtering that implicit function gives
        // the boundary its antialiasing; the fan edges stay hard so that
        // adjacent triangles keep tiling exactly.
        float implicit = uv_coords.x * uv_coords.x - uv_coords.y;
        float width = fwidth(implicit);
        value = width > 0.0
            ? clamp(0.5 - implicit / width, 0.0, 1.0)
            : float(implicit < 0.0);
        if (value <= 0.0) discard;
    }
    float signed_coverage = (gl_FrontFacing ? 1.0 : -1.0) * value;
    coverage = vec4(signed_coverage * v_color.rgb, signed_coverage);
    weighted_opacity = signed_coverage * v_color.a;
}
