// A colour field in scene coordinates (manim.paint.ColorField), sampled where the
// fragment is. field_on is 0 for paths without one.
uniform float field_on;
uniform sampler2D field_texture;
// Texel coordinates (u, v) = field_uv * (gl_FragCoord.xy, 1).
uniform mat3 field_uv;
uniform float field_opacity;

vec4 field_colour(){
    vec2 uv = (field_uv * vec3(gl_FragCoord.xy, 1.0)).xy;
    // Premultiplied in the texture: filtering does not bleed transparent colour.
    vec4 premultiplied = texture(field_texture, uv);
    vec3 rgb = premultiplied.a > 0.0 ? premultiplied.rgb / premultiplied.a : vec3(0.0);
    return vec4(rgb, premultiplied.a * field_opacity);
}
