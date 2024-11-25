#ifndef MOBJECT_GLSL
#define MOBJECT_GLSL

layout (std140) uniform ubo_mobject {
    vec3 light_source_position;
    float gloss;
    float shadow;
    float reflectiveness;
    float flat_stroke;
    float joint_type;
    float is_fixed_in_frame;
    float is_fixed_orientation;
    vec3 fixed_orientation_center;
};

#endif // MOBJECT_GLSL
