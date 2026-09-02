// WebGPU OIT accumulation shader — Weighted Blended Order-Independent Transparency.
//
// McGuire & Bavoil 2013.  Renders transparent surface fragments into two
// accumulation targets instead of the main framebuffer:
//
//   location 0  accum   rgba16float   weighted colour + alpha sum
//   location 1  reveal  rgba16float   per-channel transmittance product
//
// The main pipeline blend modes for these targets are set to:
//   accum:   {src: one,  dst: one}            — additive accumulation
//   reveal:  {src: zero, dst: one-minus-src-alpha} — transmittance multiplication
//
// A subsequent full-screen composition pass reads both textures and composites
// the result onto the opaque framebuffer.
//
// Vertex layout matches surface_combined.wgsl (stride 84 bytes):
//   location 0 — in_vert            float32x3  offset  0
//   location 1 — in_normal          float32x3  offset 12
//   location 2 — in_fill_color      float32x4  offset 24
//   location 3 — in_stroke_color    float32x4  offset 40
//   location 4 — in_bary            float32x3  offset 56
//   location 5 — stroke_half_px     float32    offset 68
//   location 6 — diffuse_strength   float32    offset 72
//   location 7 — specular_strength  float32    offset 76
//   location 8 — specular_exponent  float32    offset 80

// ── Lighting uniform layout (656 bytes total) ─────────────────────────────
//
//   offset   0 — projection  mat4x4<f32>  64 B
//   offset  64 — view        mat4x4<f32>  64 B
//   offset 128 — num_lights  u32           4 B
//   offset 132 — _pad        u32 × 3      12 B
//   offset 144 — lights      Light × 8   512 B
//
// Light struct (64 bytes):
//   offset  0  position   vec3<f32>  12 B
//   offset 12  kind       u32         4 B  — 0=ambient,1=directional,2=point,3=spot
//   offset 16  direction  vec3<f32>  12 B
//   offset 28  intensity  f32         4 B
//   offset 32  color      vec3<f32>  12 B
//   offset 44  cone_angle f32         4 B
//   offset 48  penumbra   f32         4 B
//   offset 52  _pad0-2    f32 × 3    12 B

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
    @location(0) in_vert            : vec3<f32>,
    @location(1) in_normal          : vec3<f32>,
    @location(2) in_fill_color      : vec4<f32>,
    @location(3) in_stroke_color    : vec4<f32>,
    @location(4) in_bary            : vec3<f32>,
    @location(5) stroke_half_px     : f32,
    @location(6) diffuse_strength   : f32,
    @location(7) specular_strength  : f32,
    @location(8) specular_exponent  : f32,
};

struct VertexOutput {
    @builtin(position)              clip_position  : vec4<f32>,
    @location(0)                    v_fill_color   : vec4<f32>,
    @location(1)                    v_stroke_color : vec4<f32>,
    @location(2)                    v_view_normal  : vec3<f32>,
    @location(3)                    v_view_pos     : vec3<f32>,
    @location(4)                    v_bary         : vec3<f32>,
    @location(5) @interpolate(flat) v_stroke_half  : f32,
    @location(6) @interpolate(flat) v_diffuse      : f32,
    @location(7) @interpolate(flat) v_specular     : f32,
    @location(8) @interpolate(flat) v_spec_exp     : f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let view_pos       = u.view * vec4<f32>(in.in_vert, 1.0);
    out.clip_position  = u.projection * view_pos;
    out.v_view_pos     = view_pos.xyz;
    let view3          = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    out.v_view_normal  = view3 * in.in_normal;
    out.v_fill_color   = in.in_fill_color;
    out.v_stroke_color = in.in_stroke_color;
    out.v_bary         = in.in_bary;
    out.v_stroke_half  = in.stroke_half_px;
    out.v_diffuse      = in.diffuse_strength;
    out.v_specular     = in.specular_strength;
    out.v_spec_exp     = in.specular_exponent;
    return out;
}

// ── Lighting helpers ──────────────────────────────────────────────────────────

fn compute_lighting(
    view_pos    : vec3<f32>,
    view_normal : vec3<f32>,
    base_rgb    : vec3<f32>,
    diff_str    : f32,
    spec_str    : f32,
    spec_exp    : f32,
) -> vec3<f32> {
    let specular_exp = spec_exp;
    let view_dir     = normalize(-view_pos);
    let view3        = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);

    var acc_rgb = vec3<f32>(0.0);

    for (var i = 0u; i < u.num_lights; i++) {
        let L = u.lights[i];

        switch L.kind {
            case 0u: {
                acc_rgb += base_rgb * L.color * L.intensity;
            }
            case 1u: {
                let light_dir  = normalize(-(view3 * L.direction));
                let half_vec   = normalize(light_dir + view_dir);
                let diff       = clamp(dot(view_normal, light_dir), 0.0, 1.0);
                let spec       = pow(max(dot(view_normal, half_vec), 0.0), specular_exp);
                acc_rgb += base_rgb * L.color * (diff_str * diff * L.intensity);
                acc_rgb += L.color            * (spec_str * spec * L.intensity);
            }
            case 2u: {
                let light_view_pos = (u.view * vec4<f32>(L.position, 1.0)).xyz;
                let light_dir_v    = light_view_pos - view_pos;
                let light_dir      = normalize(light_dir_v);
                let half_vec       = normalize(light_dir + view_dir);
                let diff           = clamp(dot(view_normal, light_dir), 0.0, 1.0);
                let spec           = pow(max(dot(view_normal, half_vec), 0.0), specular_exp);
                let attenuation    = L.intensity / dot(light_dir_v, light_dir_v);
                acc_rgb += base_rgb * L.color * (diff_str * diff * attenuation);
                acc_rgb += L.color            * (spec_str * spec * attenuation);
            }
            case 3u: {
                let light_view_pos = (u.view * vec4<f32>(L.position, 1.0)).xyz;
                let light_dir_v    = light_view_pos - view_pos;
                let light_dir      = normalize(light_dir_v);
                let half_vec       = normalize(light_dir + view_dir);
                let diff           = clamp(dot(view_normal, light_dir), 0.0, 1.0);
                let spec           = pow(max(dot(view_normal, half_vec), 0.0), specular_exp);
                let attenuation    = L.intensity / dot(light_dir_v, light_dir_v);
                let spot_dir       = normalize(view3 * L.direction);
                let cos_theta      = dot(-light_dir, spot_dir);
                let cos_inner      = cos(radians(L.cone_angle));
                let cos_outer      = cos(radians(L.cone_angle + L.penumbra));
                let spot_factor    = clamp((cos_theta - cos_outer) / (cos_inner - cos_outer + 1e-6), 0.0, 1.0);
                acc_rgb += base_rgb * L.color * (diff_str * diff * attenuation * spot_factor);
                acc_rgb += L.color            * (spec_str * spec * attenuation * spot_factor);
            }
            default: {}
        }
    }

    return clamp(acc_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

struct FragOutput {
    @location(0) accum  : vec4<f32>,  // weighted colour sum   → rgba16float
    @location(1) reveal : vec4<f32>,  // transmittance product → rgba16float
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragOutput {
    let diffuse_strength  = in.v_diffuse;
    let specular_strength = in.v_specular;

    let raw_normal = select(-in.v_view_normal, in.v_view_normal, front_facing);
    let norm       = normalize(raw_normal);

    let lit_rgb = compute_lighting(
        in.v_view_pos, norm,
        in.v_fill_color.rgb,
        diffuse_strength, specular_strength, in.v_spec_exp,
    );
    let fill_a = in.v_fill_color.a;

    // ── Barycentric wireframe ──────────────────────────────────────────────
    let edge_dist_px = in.v_bary.x / max(fwidth(in.v_bary.x), 1e-6);
    let stroke_cov   = clamp(in.v_stroke_half + 0.5 - edge_dist_px, 0.0, 1.0);
    let stroke_a     = in.v_stroke_color.a * stroke_cov;

    // ── Porter-Duff "over": stroke on top of fill ─────────────────────────
    let total_a = stroke_a + fill_a * (1.0 - stroke_a);
    if total_a <= 0.001 { discard; }
    let out_rgb = (stroke_a * in.v_stroke_color.rgb + fill_a * (1.0 - stroke_a) * lit_rgb) / total_a;

    // ── Weighted Blended OIT ───────────────────────────────────────────────
    let z = in.v_view_pos.z;
    let w = clamp(
        pow(total_a, 3.0) / (1e-5 + pow(abs(z) / 5.0, 4.0)),
        1e-2, 3e3
    );

    var out: FragOutput;
    out.accum  = vec4<f32>(out_rgb * total_a * w, total_a * w);
    out.reveal = vec4<f32>(total_a, total_a, total_a, total_a);
    return out;
}
