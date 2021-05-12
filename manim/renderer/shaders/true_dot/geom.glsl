#version 330

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

// Needed for get_gl_Position
uniform vec2 frame_shape;
uniform float focal_distance;
uniform float is_fixed_in_frame;
uniform float anti_alias_width;

in vec3 v_point[1];
in float v_radius[1];
in vec4 v_color[1];

out vec4 color;
out float radius;
out vec2 center;
out vec2 point;

#include ../include/get_gl_Position.glsl

void main() {
    color = v_color[0];
    radius = v_radius[0];
    center = v_point[0].xy;
    
    radius = v_radius[0] / max(1.0 - v_point[0].z / focal_distance / frame_shape.y, 0.0);
    float rpa = radius + anti_alias_width;

    for(int i = 0; i < 4; i++){
        // To account for perspective

        int x_index = 2 * (i % 2) - 1;
        int y_index = 2 * (i / 2) - 1;
        vec3 corner = v_point[0] + vec3(x_index * rpa, y_index * rpa, 0.0);

        gl_Position = get_gl_Position(corner);
        point = corner.xy;
        EmitVertex();
    }
    EndPrimitive();
}
