#version 330

uniform float has_texture;
uniform sampler2D image;
uniform float lit;
uniform float roughness;
uniform float metallic;
uniform vec3 emissive;
// The most lights a frame can have (OpenGLRenderer raises beyond it).
#define MAX_LIGHTS 8
uniform vec3 ambient;
uniform int light_count;
uniform vec3 light_positions[MAX_LIGHTS];
uniform vec3 light_rgbs[MAX_LIGHTS];
uniform float light_intensities[MAX_LIGHTS];
uniform vec3 eye_position;
// Half the width of the faces' outlines in pixels; 0 for none.
uniform float edge_width;

// In scene coordinates, where the lighting is computed.
in vec3 v_scene_point;
in vec3 v_scene_normal;
in vec4 v_color;
in vec2 v_uv;
in vec4 v_edge;
in vec4 v_stroke;

out vec4 frag_color;

#include ../include/lighting.glsl

void main(){
    vec4 color = v_color;
    if(bool(has_texture)){
        vec4 texel = texture(image, v_uv);
        color = vec4(texel.rgb, texel.a * v_color.a);
    }
    // The face is lit; its outline, as on the Cairo renderer, is not.
    vec4 face = vec4(shade(color.rgb, v_scene_normal, v_scene_point), color.a);
    if(edge_width > 0.0){
        // Pixels to the nearest side of the face.
        vec4 sides = v_edge / max(fwidth(v_edge), vec4(1e-6));
        float distance = min(min(sides.x, sides.y), min(sides.z, sides.w));
        float coverage = clamp(edge_width + 0.5 - distance, 0.0, 1.0) * v_stroke.a;
        // The outline over the face.
        float alpha = coverage + face.a * (1.0 - coverage);
        face.rgb = (
            v_stroke.rgb * coverage + face.rgb * face.a * (1.0 - coverage)
        ) / max(alpha, 1e-6);
        face.a = alpha;
    }
    frag_color = face;
}
