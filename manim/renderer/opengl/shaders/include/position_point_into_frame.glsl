// Assumes the following uniforms exist in the surrounding context:
// uniform float is_fixed_in_frame;
// uniform float is_fixed_orientation;
// uniform vec3 fixed_orientation_center;
// uniform vec3 camera_center;
// uniform mat3 camera_rotation;

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
