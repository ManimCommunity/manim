layout (std140) uniform ubo_camera {
    // mat4 u_projection_view_matrix; # TODO: convert to mat4 instead of the following...
    vec2 frame_shape;
    vec3 camera_center;
    mat3 camera_rotation;
    float focal_distance;
    float is_fixed_in_frame;
    float is_fixed_orientation;
    vec3 fixed_orientation_center;
};
