#version 330

/*
Colours every pixel the accumulated winding says is inside, exactly once, so
that an authored opacity is applied a single time however many triangles of the
group covered the pixel. Colour and opacity are divided back out of their
coverage-weighted sums; their sign cancels with the winding's, so a contour wound
either way keeps the colour it was drawn with.
*/

uniform sampler2DMS accumulated;
uniform sampler2DMS accumulated_opacity;
uniform int samples;

out vec4 frag_color;

void main(){
    // Averaging the samples here is what antialiases the boundary.
    vec4 total = vec4(0.0);
    float total_opacity = 0.0;
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    for (int index = 0; index < samples; index++) {
        total += texelFetch(accumulated, pixel, index);
        total_opacity += texelFetch(accumulated_opacity, pixel, index).r;
    }
    float winding = total.a / float(samples);
    if (winding == 0.0) discard;
    float opacity = total_opacity / total.a;
    // Clamped, so a region enclosed twice is inside once, not twice as opaque.
    float alpha = clamp(abs(winding), 0.0, 1.0) * opacity;
    if (alpha < 1.0 / 512.0) discard;
    frag_color = vec4(total.rgb / total.a, alpha);
}
