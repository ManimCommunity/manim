#version 330

#include "../include/camera_uniform_declarations.glsl"
#include "../include/quadratic_bezier_distance.glsl"

uniform vec2 pixel_shape;
uniform float index;
uniform float disable_stencil;

in vec2 uv_coords;
in vec2 uv_b2;

in float uv_stroke_width;
in vec4 color;
in float uv_anti_alias_width;

in float has_prev;
in float has_next;
in float bevel_start;
in float bevel_end;
in float angle_from_prev;
in float angle_to_next;
in float bezier_degree;

uniform sampler2D stencil_texture;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 stencil_value;

float cross2d(vec2 v, vec2 w)
{
    return v.x * w.y - w.x * v.y;
}

float modify_distance_for_endpoints(vec2 p, float dist, float t)
{
    float buff = 0.5 * uv_stroke_width - uv_anti_alias_width;
    // Check the beginning of the curve
    if (t == 0)
    {
        // Clip the start
        if (has_prev == 0)
            return max(dist, -p.x + buff);
        // Bevel start
        if (bevel_start == 1)
        {
            float a = angle_from_prev;
            mat2 rot = mat2(cos(a), sin(a), -sin(a), cos(a));
            // Dist for intersection of two lines
            float bevel_d = max(abs(p.y), abs((rot * p).y));
            // Dist for union of this intersection with the real curve
            // intersected with radius 2 away from curve to smooth out
            // really sharp corners
            return max(min(dist, bevel_d), dist / 2);
        }
        // Otherwise, start will be rounded off
    }
    else if (t == 1)
    {
        // Check the end of the curve
        // TODO, too much code repetition
        vec2 v21 = (bezier_degree == 2) ? vec2(1, 0) - uv_b2 : vec2(-1, 0);
        float len_v21 = length(v21);
        if (len_v21 == 0)
        {
            v21 = -uv_b2;
            len_v21 = length(v21);
        }

        float perp_dist = dot(p - uv_b2, v21) / len_v21;
        if (has_next == 0)
            return max(dist, -perp_dist + buff);
        // Bevel end
        if (bevel_end == 1)
        {
            float a = -angle_to_next;
            mat2 rot = mat2(cos(a), sin(a), -sin(a), cos(a));
            vec2 v21_unit = v21 / length(v21);
            float bevel_d = max(abs(cross2d(p - uv_b2, v21_unit)), abs(cross2d((rot * (p - uv_b2)), v21_unit)));
            return max(min(dist, bevel_d), dist / 2);
        }
        // Otherwise, end will be rounded off
    }
    return dist;
}

void main()
{
    // Use the default value as standard output
    if (disable_stencil == 1.0)
    {
        stencil_value = vec4(0.0);
    }
    else
    {
        stencil_value.rgb = vec3(index);
        stencil_value.a = 1.0;
    }
    gl_FragDepth = gl_FragCoord.z;
    // Get the previous index that was written to this fragment
    float previous_index =
        texture2D(stencil_texture, vec2(gl_FragCoord.x / pixel_shape.x, gl_FragCoord.y / pixel_shape.y)).r;
    // If the index is the same that means we are overlapping with the fill and
    // crossing through so we push the stroke forward a tiny bit
    if (previous_index < index && previous_index != 0)
    {
        gl_FragDepth = gl_FragCoord.z - 1.7 * index / 1000.0;
    }
    if (previous_index == index)
    {
        gl_FragDepth = gl_FragCoord.z - index / 1000.0;
    }
    // If the stroke is overlapping with a shape that is of higher index that
    // means it is behind another mobject on the same plane so we discard the
    // fragment
    if (previous_index > index)
    {
        // But for stroke transparency we shouldn't discard but move the stroke in
        // front so it is not discarded by the depth test
        // TODO: This is highly experimental and should later be rethought and if no
        // good solution is found it should just be a discard;
        if (color.a == 1.0)
            discard;
        else
            gl_FragDepth = gl_FragCoord.z + index / 1000.0;
    }
    if (disable_stencil == 1.0)
    {
        gl_FragDepth = gl_FragCoord.z + 4.5 * index / 1000.0;
    }
    if (uv_stroke_width == 0)
        discard;
    float dist_to_curve = min_dist_to_curve(uv_coords, uv_b2, bezier_degree);
    // An sdf for the region around the curve we wish to color.
    float signed_dist = abs(dist_to_curve) - 0.5 * uv_stroke_width;

    frag_color = color;
    frag_color.a *= smoothstep(0.5, -0.5, signed_dist / uv_anti_alias_width);
    if (frag_color.a <= 0.0)
    {
        discard;
    }
}
