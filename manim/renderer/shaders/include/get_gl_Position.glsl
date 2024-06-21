layout(std140) uniform ubo_camera {
    mat4 view;
    vec2 frame_shape;
    vec3 camera_center;
    mat3 camera_rotation;
    float focal_distance;
    float is_fixed_in_frame;
    float is_fixed_orientation;
    vec3 fixed_orientation_center;
};

const float DEFAULT_FRAME_HEIGHT = 8.0;
const float ASPECT_RATIO = 16.0 / 9.0;
const float X_SCALE = 2.0 / DEFAULT_FRAME_HEIGHT / ASPECT_RATIO;
const float Y_SCALE = 2.0 / DEFAULT_FRAME_HEIGHT;

void emit_gl_Position(vec3 point) {
    vec4 result = vec4(point, 1.0);
    // This allow for smooth transitions between objects fixed and unfixed from frame
    result = mix(view * result, result, is_fixed_in_frame);
    // Essentially a projection matrix
    result.x *= X_SCALE;
    result.y *= Y_SCALE;
    result.z /= focal_distance;
    result.w = 1.0 - result.z;
    // Flip and scale to prevent premature clipping
    result.z *= -0.1;
    gl_Position = result;
}
