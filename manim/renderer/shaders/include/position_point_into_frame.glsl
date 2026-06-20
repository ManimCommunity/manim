#ifndef POSITION_POINT_INTO_FRAME_GLSL
#define POSITION_POINT_INTO_FRAME_GLSL

#include "./camera_uniform_declarations.glsl"
#include "./mobject_uniform_declarations.glsl"

vec3 position_point_into_frame(vec3 point)
{
    if (bool(is_fixed_in_frame))
    {
        return point;
    }
    if (bool(is_fixed_orientation))
    {
	return point - camera_rotation * camera_center + camera_rotation * fixed_orientation_center - fixed_orientation_center;
    }
    return camera_rotation*(point - camera_center);
}
#endif // POSITION_POINT_INTO_FRAME_GLSL
