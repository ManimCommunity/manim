layout (std140) uniform ubo_camera {
    vec2 frame_shape;
    vec3 camera_center;
    mat3 camera_rotation;
    float is_fixed_in_frame;
    float is_fixed_orientation;
    vec3 fixed_orientation_center;
    float focal_distance;
};
// uniform vec2 frame_shape;
// uniform vec3 camera_center;
// uniform mat3 camera_rotation;
// uniform float is_fixed_in_frame;
// uniform float is_fixed_orientation;
// uniform vec3 fixed_orientation_center;
// uniform float focal_distance;

