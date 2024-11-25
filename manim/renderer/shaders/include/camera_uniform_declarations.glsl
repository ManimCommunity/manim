#ifndef CAMERA_GLSL
#define CAMERA_GLSL
layout (std140) uniform ubo_camera {
    // mat4 u_projection_view_matrix; # TODO: convert to mat4 instead of the following...
    vec2 frame_shape;
    vec3 camera_center;
    mat3 camera_rotation;
    float focal_distance;
};
#endif // CAMERA_GLSL
