// Shading of lit surfaces: the GLSL version of manim/renderer/_lighting.py's shade().
// Keep the two in step. Colors are sRGB in and out; the shading is in linear RGB.
//
// Assumes the following uniforms exist in the surrounding context:
// uniform float lit;            // a PbrMaterial with a light to shade it: 1, else 0
// uniform float roughness;
// uniform float metallic;
// uniform vec3 emissive;        // linear RGB
// uniform float ambient;
// uniform vec3 light_position;  // in the same space as the points
// uniform vec3 light_rgb;       // linear RGB
// uniform float light_intensity;
// uniform vec3 eye_position;    // in the same space as the points

const float PI_LIGHTING = 3.14159265358979;
const float MIN_ROUGHNESS = 0.03;
const float MIN_DISTANCE_SQUARED = 1e-4;

vec3 srgb_to_linear(vec3 c){
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linear_to_srgb(vec3 c){
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

vec3 shade(vec3 base_srgb, vec3 normal, vec3 point){
    if(!bool(lit)){
        return base_srgb;
    }
    vec3 base = srgb_to_linear(base_srgb);
    float r = max(roughness, MIN_ROUGHNESS);
    float alpha = r * r;
    float exponent = 2.0 / (alpha * alpha) - 2.0;
    vec3 diffuse = base * (1.0 - metallic);
    vec3 f0 = mix(vec3(0.04), base, metallic);
    vec3 result = emissive + ambient * (diffuse + f0);

    float normal_length = length(normal);
    if(normal_length == 0.0){
        return linear_to_srgb(result);
    }
    vec3 n = normal / normal_length;
    vec3 to_eye = eye_position - point;
    to_eye = length(to_eye) > 0.0 ? normalize(to_eye) : vec3(0.0);
    if(dot(n, to_eye) < 0.0){
        n = -n;  // two-sided
    }
    vec3 offset = light_position - point;
    float distance_squared = max(dot(offset, offset), MIN_DISTANCE_SQUARED);
    vec3 to_light = length(offset) > 0.0 ? normalize(offset) : vec3(0.0);
    vec3 radiance = light_rgb * light_intensity / distance_squared;
    float n_dot_l = max(dot(n, to_light), 0.0);
    vec3 half_vector = to_light + to_eye;
    float highlight = 0.0;
    if(length(half_vector) > 0.0){
        float n_dot_h = dot(n, normalize(half_vector));
        // pow(0, 0) is undefined in GLSL (NaN on some drivers); no highlight there.
        if(n_dot_h > 0.0){
            highlight = pow(n_dot_h, exponent);
        }
    }
    vec3 specular = f0 * (exponent + 2.0) / (2.0 * PI_LIGHTING) * highlight;
    result += radiance * n_dot_l * (diffuse / PI_LIGHTING + specular);
    return linear_to_srgb(result);
}
