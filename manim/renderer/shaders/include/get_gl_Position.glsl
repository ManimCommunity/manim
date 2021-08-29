// Assumes the following uniforms exist in the surrounding context:
// uniform vec2 frame_shape;
// uniform float focal_distance;
// uniform float is_fixed_in_frame;

const vec2 DEFAULT_FRAME_SHAPE = vec2(8.0 * 16.0 / 9.0, 8.0);

float perspective_scale_factor(float z, float focal_distance){
    return max(0.0, focal_distance / (focal_distance - z));
}


vec4 get_gl_Position(vec3 point){
    vec4 result = vec4(point, 1.0);
    if(!bool(is_fixed_in_frame)){
        result.x *= 2.0 / frame_shape.x;
        result.y *= 2.0 / frame_shape.y;
    } else{
        result.x *= 2.0 / DEFAULT_FRAME_SHAPE.x;
        result.y *= 2.0 / DEFAULT_FRAME_SHAPE.y;
    }
    result.z *= -1;
    return result;
}
