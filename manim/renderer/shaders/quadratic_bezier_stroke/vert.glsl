#version 330

#include "../include/get_gl_Position.glsl"

uniform float frame_scale;

in vec3 point;
in vec4 color;
in float stroke_width;
in vec4 joint_product;

// Bezier control point
out vec3 verts;

out vec4 v_joint_product;
out float v_stroke_width;
out vec4 v_color;

const float STROKE_WIDTH_CONVERSION = 0.01;

void main() {
    verts = point;
    v_stroke_width = STROKE_WIDTH_CONVERSION * stroke_width;
    v_stroke_width *= mix(frame_scale, 1, is_fixed_in_frame);
    v_joint_product = joint_product;
    v_color = color;
}
