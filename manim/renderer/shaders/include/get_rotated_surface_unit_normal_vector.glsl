#ifndef GET_ROTATED_SURFACE_GLSL
#define GET_ROTATED_SURFACE_GLSL

#include "./camera_uniform_declarations.glsl"

vec3 get_rotated_surface_unit_normal_vector(vec3 point, vec3 du_point, vec3 dv_point)
{
    vec3 cp = cross((du_point - point), (dv_point - point));
    if (length(cp) == 0)
    {
        // Instead choose a normal to just dv_point - point in the direction of point
        vec3 v2 = dv_point - point;
        cp = cross(cross(v2, point), v2);
    }
    return normalize(rotate_point_into_frame(cp));
}
#endif // GET_ROTATED_SURFACE_GLSL
