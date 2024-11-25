#version 330

#include "../include/camera_uniform_declarations.glsl"
#include "../include/finalize_color.glsl"
#include "../include/get_gl_Position.glsl"
#include "../include/get_unit_normal.glsl"
#include "../include/mobject_uniform_declarations.glsl"
#include "../include/quadratic_bezier_geometry_functions.glsl"

layout(triangles) in;
layout(triangle_strip, max_vertices = 5) out;

// Needed for get_gl_Position
// uniform vec2 frame_shape;
// uniform float focal_distance;

in vec3 bp[3];
in vec3 v_global_unit_normal[3];
in vec4 v_color[3];
in float v_vert_index[3];

out vec4 color;
out float fill_all;

out float orientation;
out vec2 uv_coords;
out float bezier_degree;

const vec2 uv_coords_arr[3] = vec2[3](vec2(0, 0), vec2(0.5, 0), vec2(1, 1));

void emit_vertex_wrapper(vec3 point, int index)
{
    color = finalize_color(v_color[index], point, v_global_unit_normal[index], light_source_position, gloss, shadow,
                           reflectiveness);
    gl_Position = get_gl_Position(point);
    uv_coords = uv_coords_arr[index];
    EmitVertex();
}

void emit_simple_triangle()
{
    for (int i = 0; i < 3; i++)
    {
        emit_vertex_wrapper(bp[i], i);
    }
    EndPrimitive();
}

void main()
{
    // If vert indices are sequential, don't fill all
    fill_all = float((v_vert_index[1] - v_vert_index[0]) != 1.0 || (v_vert_index[2] - v_vert_index[1]) != 1.0);

    if (fill_all == 1.0)
    {
        emit_simple_triangle();
        return;
    }

    vec3 new_bp[3];
    bezier_degree = get_reduced_control_points(vec3[3](bp[0], bp[1], bp[2]), new_bp);
    vec3 local_unit_normal = get_unit_normal(new_bp);
    orientation = sign(dot(v_global_unit_normal[0], local_unit_normal));

    if (bezier_degree >= 1)
    {
        emit_simple_triangle();
    }
    // Don't emit any vertices for bezier_degree 0
}
