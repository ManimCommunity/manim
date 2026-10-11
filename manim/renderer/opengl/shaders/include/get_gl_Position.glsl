// Assumes the following uniforms exist in the surrounding context:
// uniform vec2 frame_shape;
// uniform float focal_distance;

float perspective_scale_factor(float z, float focal_distance){
    return max(0.0, focal_distance / (focal_distance - z));
}


vec4 get_gl_Position(vec3 point){
    vec4 result = vec4(point, 1.0);
    result.x *= 2.0 / frame_shape.x;
    result.y *= 2.0 / frame_shape.y;
    float psf = perspective_scale_factor(result.z, focal_distance);
    if (psf > 0){
        result.xy *= psf;
        // Keep vertices far out of frame from getting clipped.
        result.z *= 0.01;
    }
    result.z *= -1;
    return result;
}
