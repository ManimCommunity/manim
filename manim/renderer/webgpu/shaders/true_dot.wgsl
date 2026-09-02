// WebGPU TrueDot shader — screen-aligned sphere dot rendering.
//
// Each dot is expanded CPU-side into 2 triangles (6 vertices) forming a
// screen-aligned quad.  UV coords span (-1,-1) → (1,1) across the quad.
// The fragment shader treats the quad as a sphere projected onto the screen:
//   • Pixels outside the unit disc are discarded (anti-aliased edge).
//   • The sphere normal is reconstructed from the UV position.
//   • Multi-light Phong shading is applied (same light array as surface shaders).
//     gloss / shadow parameters blend the result toward the Cairo-style look.
//
// The same camera Uniforms struct and bind group layout as the surface
// shaders are reused (binding 0, group 0).
//
// Vertex layout (stride 48 bytes) — must match _TRUE_DOT_DTYPE:
//   location 0 — center  float32x3  offset  0   (12 B)
//   location 1 — color   float32x4  offset 12   (16 B)
//   location 2 — uv      float32x2  offset 28   ( 8 B)
//   location 3 — radius  float32    offset 36   ( 4 B)
//   location 4 — gloss   float32    offset 40   ( 4 B)
//   location 5 — shadow  float32    offset 44   ( 4 B)

// ── Shared uniform (same layout as surface shaders, 656 bytes) ────────────────

const MAX_LIGHTS : u32 = 8u;

struct Light {
    position   : vec3<f32>,
    kind       : u32,
    direction  : vec3<f32>,
    intensity  : f32,
    color      : vec3<f32>,
    cone_angle : f32,
    penumbra   : f32,
    _pad0      : f32,
    _pad1      : f32,
    _pad2      : f32,
};

struct Uniforms {
    projection : mat4x4<f32>,
    view       : mat4x4<f32>,
    num_lights : u32,
    _pad0      : u32,
    _pad1      : u32,
    _pad2      : u32,
    lights     : array<Light, 8>,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) center : vec3<f32>,
    @location(1) color  : vec4<f32>,
    @location(2) uv     : vec2<f32>,
    @location(3) radius : f32,
    @location(4) gloss  : f32,
    @location(5) shadow : f32,
};

struct VertexOutput {
    @builtin(position)              clip_position  : vec4<f32>,
    @location(0)                    v_color        : vec4<f32>,
    @location(1)                    v_uv           : vec2<f32>,
    @location(2) @interpolate(flat) v_gloss        : f32,
    @location(3) @interpolate(flat) v_shadow       : f32,
    @location(4)                    v_center_view  : vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Project dot center into view space.
    let cv = u.view * vec4<f32>(in.center, 1.0);

    // Expand quad in view space: move corner by radius × UV along x/y.
    let expanded = cv + vec4<f32>(in.uv.x * in.radius, in.uv.y * in.radius, 0.0, 0.0);

    var out: VertexOutput;
    out.clip_position = u.projection * expanded;
    out.v_color       = in.color;
    out.v_uv          = in.uv;
    out.v_gloss       = in.gloss;
    out.v_shadow      = in.shadow;
    out.v_center_view = cv.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.v_uv);

    // Anti-aliased disc edge.
    let fw         = fwidth(d);
    let alpha_mult = 1.0 - smoothstep(1.0 - fw, 1.0 + fw, d);
    if alpha_mult <= 0.001 { discard; }

    // Reconstruct sphere surface normal in view space.
    let z2           = max(0.0, 1.0 - d * d);
    let sphere_normal = normalize(vec3<f32>(in.v_uv.x, in.v_uv.y, sqrt(z2)));
    let view_dir      = normalize(-in.v_center_view);
    let view3         = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);

    // ── Multi-light accumulation ──────────────────────────────────────────
    var acc_rgb = vec3<f32>(0.0);

    for (var i = 0u; i < u.num_lights; i++) {
        let L = u.lights[i];

        switch L.kind {
            // Ambient
            case 0u: {
                acc_rgb += in.v_color.rgb * L.color * L.intensity;
            }
            // Directional
            case 1u: {
                let to_light = normalize(-(view3 * L.direction));
                let dot_ln   = clamp(dot(sphere_normal, to_light), 0.0, 1.0);
                // Cairo-style shadow: darken by Lambertian term
                let darkening = mix(1.0, dot_ln, in.v_shadow);
                // Cairo-style specular gloss
                let reflect_l = reflect(-to_light, sphere_normal);
                let dot_rv    = clamp(dot(reflect_l, view_dir), 0.0, 1.0);
                let shine     = in.v_gloss * exp(-3.0 * pow(1.0 - dot_rv, 2.0));
                let lit = darkening * mix(in.v_color.rgb, vec3<f32>(1.0), shine);
                acc_rgb += lit * L.color * L.intensity;
            }
            // Point
            case 2u: {
                let light_vpos = (u.view * vec4<f32>(L.position, 1.0)).xyz;
                let light_dir_v = light_vpos - in.v_center_view;
                let to_light   = normalize(light_dir_v);
                let attenuation = L.intensity / dot(light_dir_v, light_dir_v);
                let dot_ln     = clamp(dot(sphere_normal, to_light), 0.0, 1.0);
                let darkening  = mix(1.0, dot_ln, in.v_shadow);
                let reflect_l  = reflect(-to_light, sphere_normal);
                let dot_rv     = clamp(dot(reflect_l, view_dir), 0.0, 1.0);
                let shine      = in.v_gloss * exp(-3.0 * pow(1.0 - dot_rv, 2.0));
                let lit        = darkening * mix(in.v_color.rgb, vec3<f32>(1.0), shine);
                acc_rgb += lit * L.color * attenuation;
            }
            // Spot
            case 3u: {
                let light_vpos  = (u.view * vec4<f32>(L.position, 1.0)).xyz;
                let light_dir_v = light_vpos - in.v_center_view;
                let to_light    = normalize(light_dir_v);
                let attenuation = L.intensity / dot(light_dir_v, light_dir_v);
                let spot_dir    = normalize(view3 * L.direction);
                let cos_theta   = dot(-to_light, spot_dir);
                let cos_inner   = cos(radians(L.cone_angle));
                let cos_outer   = cos(radians(L.cone_angle + L.penumbra));
                let spot_factor = clamp((cos_theta - cos_outer) / (cos_inner - cos_outer + 1e-6), 0.0, 1.0);
                let dot_ln      = clamp(dot(sphere_normal, to_light), 0.0, 1.0);
                let darkening   = mix(1.0, dot_ln, in.v_shadow);
                let reflect_l   = reflect(-to_light, sphere_normal);
                let dot_rv      = clamp(dot(reflect_l, view_dir), 0.0, 1.0);
                let shine       = in.v_gloss * exp(-3.0 * pow(1.0 - dot_rv, 2.0));
                let lit         = darkening * mix(in.v_color.rgb, vec3<f32>(1.0), shine);
                acc_rgb += lit * L.color * attenuation * spot_factor;
            }
            default: {}
        }
    }

    let out_rgb = clamp(acc_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(out_rgb, in.v_color.a * alpha_mult);
}
