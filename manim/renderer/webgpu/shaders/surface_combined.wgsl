// Combined opaque surface fill + barycentric wireframe shader.
//
// One draw call per surface face renders both Phong-lit fill and mesh-grid
// lines without a separate stroke pass.
//
// Technique
// ---------
// Each triangle (centroid, anchor_i, anchor_{i+1}) in the centroid fan
// carries barycentric coordinates:
//   centroid     → bary = (1, 0, 0)   bary.x = 0 on the outer edge
//   anchor_i     → bary = (0, 1, 0)
//   anchor_{i+1} → bary = (0, 0, 1)
//
// bary.x is 0 on the "outer" edge (anchor_i ↔ anchor_{i+1}), which is the
// visible mesh-grid edge.  The inner spoke edges (centroid ↔ anchor_*) have
// bary.y = 0 or bary.z = 0 but are NOT rendered as wireframe.
//
// fwidth(bary.x) gives the screen-space derivative of bary.x, so
//   edge_dist_px = bary.x / fwidth(bary.x)
// is approximately the distance from the outer edge in screen pixels.
// A smooth SDF step at stroke_half_px produces anti-aliased grid lines.
//
// Compositing: wireframe stroke "over" Phong fill (Porter-Duff).
//
// Vertex layout (must match _SURFACE_COMBINED_DTYPE, stride 84 bytes):
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
//   offset 132 — _pad        u32 × 3      12 B  (align array to 16 B)
//   offset 144 — lights      Light × 8   512 B
//
// Light struct (64 bytes):
//   offset  0  position   vec3<f32>  12 B  — point / spot world position
//   offset 12  kind       u32         4 B  — 0=ambient,1=directional,2=point,3=spot
//   offset 16  direction  vec3<f32>  12 B  — directional / spot direction
//   offset 28  intensity  f32         4 B
//   offset 32  color      vec3<f32>  12 B
//   offset 44  cone_angle f32         4 B  — spot inner half-angle (degrees)
//   offset 48  penumbra   f32         4 B  — spot penumbra width (degrees)
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
            // ── Ambient ───────────────────────────────────────────────────
            case 0u: {
                acc_rgb += base_rgb * L.color * L.intensity;
            }
            // ── Directional ───────────────────────────────────────────────
            case 1u: {
                // direction is the direction the light *travels toward* (world space).
                // Transform to view space and negate to get the "to-light" direction.
                let light_dir  = normalize(-(view3 * L.direction));
                let half_vec   = normalize(light_dir + view_dir);
                let diff       = clamp(dot(view_normal, light_dir), 0.0, 1.0);
                let spec       = pow(max(dot(view_normal, half_vec), 0.0), specular_exp);
                acc_rgb += base_rgb * L.color * (diff_str * diff * L.intensity);
                acc_rgb += L.color            * (spec_str * spec * L.intensity);
            }
            // ── Point ─────────────────────────────────────────────────────
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
            // ── Spot ──────────────────────────────────────────────────────
            case 3u: {
                let light_view_pos = (u.view * vec4<f32>(L.position, 1.0)).xyz;
                let light_dir_v    = light_view_pos - view_pos;
                let light_dir      = normalize(light_dir_v);
                let half_vec       = normalize(light_dir + view_dir);
                let diff           = clamp(dot(view_normal, light_dir), 0.0, 1.0);
                let spec           = pow(max(dot(view_normal, half_vec), 0.0), specular_exp);
                let attenuation    = L.intensity / dot(light_dir_v, light_dir_v);

                // Cone falloff: compare angle between -light_dir and spot direction.
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

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
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
    return vec4<f32>(out_rgb, total_a);
}
