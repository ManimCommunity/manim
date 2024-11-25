#ifndef POSITION_POINT_INTO_FRAME_GLSL
#define POSITION_POINT_INTO_FRAME_GLSL

#include "./camera_uniform_declarations.glsl"
#include "./mobject_uniform_declarations.glsl"

vec3 rotate_point_into_frame(vec3 point){
    if(bool(is_fixed_in_frame)){
        return point;
    }
    return camera_rotation * point;
}


vec3 position_point_into_frame(vec3 point){
    if(bool(is_fixed_in_frame)){
        return point;
    }
    if(bool(is_fixed_orientation)){
        vec3 new_center = rotate_point_into_frame(fixed_orientation_center);
        return point + (new_center - fixed_orientation_center);
    }
    return rotate_point_into_frame(point - camera_center);
}
#endif // POSITION_POINT_INTO_FRAME_GLSL
