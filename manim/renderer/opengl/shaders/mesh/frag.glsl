#version 330

uniform float has_texture;
uniform sampler2D image;
uniform float lit;
uniform float roughness;
uniform float metallic;
uniform vec3 emissive;
uniform float ambient;
uniform vec3 light_position;
uniform vec3 light_rgb;
uniform float light_intensity;
uniform vec3 eye_position;

// In scene coordinates, where the lighting is computed.
in vec3 v_scene_point;
in vec3 v_scene_normal;
in vec4 v_color;
in vec2 v_uv;

out vec4 frag_color;

#include ../include/lighting.glsl

void main(){
    vec4 color = v_color;
    if(bool(has_texture)){
        vec4 texel = texture(image, v_uv);
        color = vec4(texel.rgb, texel.a * v_color.a);
    }
    frag_color = vec4(shade(color.rgb, v_scene_normal, v_scene_point), color.a);
}
