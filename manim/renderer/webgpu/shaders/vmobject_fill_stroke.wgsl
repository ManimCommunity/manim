// Combined VMobject fill + stroke shader.
//
// One bounding quad per object; one fragment loop simultaneously accumulates:
//   1. Fill coverage — Slug winding-number algorithm in NDC space
//      (Lengyel 2017, patent-dedicated public domain, code MIT)
//   2. Stroke coverage — SDF minimum distance to each curve in pixel space
//
// Result is Porter-Duff "over" compositing: stroke painted on top of fill.
//
// The quadratic Bezier control points are written by cubic_to_quads.wgsl
// into a single shared storage buffer.  Each object's fill and stroke
// curves occupy separate contiguous regions of that buffer referenced by
// (fill_curve_start, n_fill_curves) and (stroke_curve_start, n_stroke_curves).
//
// Objects with no fill: pass fill_color.a = 0 or n_fill_curves = 0.
// Objects with no stroke: pass stroke_half_ndc = 0 or n_stroke_curves = 0.
//
// Uniform layout (group 0, binding 0) — 656-byte block shared with surface shaders
//   (only the first two fields are used here):
//   offset   0 — projection  mat4x4<f32>  (64 B)
//   offset  64 — view        mat4x4<f32>  (64 B)
//   offset 128 — ...         (lighting data, unused by this shader)
//
// Storage buffer (group 0, binding 1) — array<f32>, 9 floats per quadratic:
//   [p0.x p0.y p0.z  pmid.x pmid.y pmid.z  p2.x p2.y p2.z]
//
// Vertex attributes (must match _FILL_STROKE_DTYPE, stride 68 bytes):
//   location 0 — in_pos             float32x3  offset  0
//   location 1 — in_fill_color      float32x4  offset 12
//   location 2 — in_stroke_color    float32x4  offset 28
//   location 3 — stroke_half_ndc    float32    offset 44
//   location 4 — fill_curve_start   uint32     offset 48
//   location 5 — n_fill_curves      uint32     offset 52
//   location 6 — stroke_curve_start uint32     offset 56
//   location 7 — n_stroke_curves    uint32     offset 60
//   location 8 — fill_rule          uint32     offset 64  (0=nonzero, 1=evenodd)

struct Uniforms {
    projection : mat4x4<f32>,
    view       : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform>        u     : Uniforms;
@group(0) @binding(1) var<storage, read>  quads : array<f32>;

struct VertexInput {
    @location(0) in_pos             : vec3<f32>,
    @location(1) in_fill_color      : vec4<f32>,
    @location(2) in_stroke_color    : vec4<f32>,
    @location(3) stroke_half_ndc    : f32,
    @location(4) fill_curve_start   : u32,
    @location(5) n_fill_curves      : u32,
    @location(6) stroke_curve_start : u32,
    @location(7) n_stroke_curves    : u32,
    @location(8) fill_rule          : u32,
};

struct VertexOutput {
    @builtin(position)              clip_pos          : vec4<f32>,
    @location(0)                    ndc_xy            : vec2<f32>,
    @location(1)                    v_fill_color      : vec4<f32>,
    @location(2)                    v_stroke_color    : vec4<f32>,
    @location(3) @interpolate(flat) v_stroke_half_ndc : f32,
    @location(4) @interpolate(flat) fill_curve_start  : u32,
    @location(5) @interpolate(flat) n_fill_curves     : u32,
    @location(6) @interpolate(flat) stroke_curve_start: u32,
    @location(7) @interpolate(flat) n_stroke_curves   : u32,
    @location(8) @interpolate(flat) fill_rule         : u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let view_pos = u.view * vec4<f32>(in.in_pos, 1.0);
    let clip     = u.projection * view_pos;
    out.clip_pos           = clip;
    // Perspective divide: ortho gives w=1 (no change); perspective maps
    // the vertex to the correct 2-D screen-proportional position.
    out.ndc_xy             = clip.xy / clip.w;
    out.v_fill_color       = in.in_fill_color;
    out.v_stroke_color     = in.in_stroke_color;
    out.v_stroke_half_ndc  = in.stroke_half_ndc;
    out.fill_curve_start   = in.fill_curve_start;
    out.n_fill_curves      = in.n_fill_curves;
    out.stroke_curve_start = in.stroke_curve_start;
    out.n_stroke_curves    = in.n_stroke_curves;
    out.fill_rule          = in.fill_rule;
    return out;
}

// ---------------------------------------------------------------------------
// Slug helpers — winding-number fill (NDC space)
// Adapted from Lengyel 2017 (HLSL → WGSL).
// ---------------------------------------------------------------------------

fn calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = (bitcast<u32>(y1) >> 31u) & 1u;
    let i2 = (bitcast<u32>(y2) >> 30u) & 2u;
    let i3 = (bitcast<u32>(y3) >> 29u) & 4u;
    return (0x2E74u >> (i3 | i2 | i1)) & 0x0101u;
}

fn solve_horiz(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> vec2<f32> {
    let ay = p1.y - 2.0*p2.y + p3.y;
    let by = p1.y - p2.y;
    let ax = p1.x - 2.0*p2.x + p3.x;
    let bx = p1.x - p2.x;
    var t1: f32; var t2: f32;
    if abs(ay) < (1.0 / 65536.0) {
        let denom = select(1.0, by, abs(by) > 1e-10);
        t1 = p1.y * 0.5 / denom; t2 = t1;
    } else {
        let ra = 1.0 / ay;
        let d  = sqrt(max(by*by - ay*p1.y, 0.0));
        t1 = (by - d) * ra; t2 = (by + d) * ra;
    }
    return vec2<f32>((ax*t1 - bx*2.0)*t1 + p1.x, (ax*t2 - bx*2.0)*t2 + p1.x);
}

fn solve_vert(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> vec2<f32> {
    let ax = p1.x - 2.0*p2.x + p3.x;
    let bx = p1.x - p2.x;
    let ay = p1.y - 2.0*p2.y + p3.y;
    let by = p1.y - p2.y;
    var t1: f32; var t2: f32;
    if abs(ax) < (1.0 / 65536.0) {
        let denom = select(1.0, bx, abs(bx) > 1e-10);
        t1 = p1.x * 0.5 / denom; t2 = t1;
    } else {
        let ra = 1.0 / ax;
        let d  = sqrt(max(bx*bx - ax*p1.x, 0.0));
        t1 = (bx - d) * ra; t2 = (bx + d) * ra;
    }
    return vec2<f32>((ay*t1 - by*2.0)*t1 + p1.y, (ay*t2 - by*2.0)*t2 + p1.y);
}

fn calc_coverage(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32) -> f32 {
    let blended = abs(xcov*xwgt + ycov*ywgt) / max(xwgt + ywgt, 1.0/65536.0);
    return clamp(max(blended, min(abs(xcov), abs(ycov))), 0.0, 1.0);
}

// Even-odd fill coverage: triangle wave — inside when winding count is odd.
// The raw winding accumulator (xcov or ycov) is a signed integer at stable
// interiors.  A triangle wave with period 2 maps even integers → 0 (outside)
// and odd integers → 1 (inside), with half-pixel AA transitions.
fn calc_coverage_evenodd(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32) -> f32 {
    let w     = (xcov*xwgt + ycov*ywgt) / max(xwgt + ywgt, 1.0/65536.0);
    // Triangle wave: period 2, peak at odd integers.
    let tri   = 1.0 - abs(2.0 * fract(abs(w) * 0.5) - 1.0);
    // Fallback: take the max of both axis coverages independently.
    let tx    = 1.0 - abs(2.0 * fract(abs(xcov) * 0.5) - 1.0);
    let ty    = 1.0 - abs(2.0 * fract(abs(ycov) * 0.5) - 1.0);
    return clamp(max(tri, min(tx, ty)), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Stroke SDF helper — min distance from origin to a 2-D quadratic Bezier.
//
// B(t) = a·t² + b·t + c,  a = p1−2·p2+p3,  b = 2(p2−p1),  c = p1.
// Fragment is at the origin, so B(t)−origin = B(t).
//
// Minimise |B(t)|² by Newton on f(t) = B(t)·B'(t).
// f'(t) = |B'(t)|² + 2a·B(t).
// ---------------------------------------------------------------------------

fn min_dist_to_quad_px(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> f32 {
    let a = p1 - 2.0*p2 + p3;
    let b = 2.0*(p2 - p1);
    let c = p1;

    // Coarse: sample t = 0, 0.25, 0.5, 0.75, 1.0.
    var best_t  = 0.0;
    var best_d2 = dot(c, c);
    for (var i = 1u; i <= 4u; i++) {
        let t   = f32(i) * 0.25;
        let bt  = a*t*t + b*t + c;
        let d2  = dot(bt, bt);
        if d2 < best_d2 { best_d2 = d2; best_t = t; }
    }

    // Newton refinement (4 iterations).
    for (var iter = 0u; iter < 4u; iter++) {
        let t   = clamp(best_t, 0.0, 1.0);
        let Bt  = a*t*t + b*t + c;
        let Bpt = 2.0*a*t + b;
        let f   = dot(Bt, Bpt);
        let fp  = dot(Bpt, Bpt) + dot(2.0*a, Bt);
        if abs(fp) < 1e-10 { break; }
        best_t  = clamp(t - f/fp, 0.0, 1.0);
    }

    let t_f = clamp(best_t, 0.0, 1.0);
    let Bf  = a*t_f*t_f + b*t_f + c;
    return sqrt(max(dot(Bf, Bf), 0.0));
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // NDC units per screen pixel (non-zero denominator guard).
    let ndc_per_pixel  = fwidth(in.ndc_xy);
    let pixels_per_ndc = 1.0 / max(ndc_per_pixel, vec2<f32>(1e-9));

    let pv = u.projection * u.view;

    // ── Fill: Slug winding-number accumulation in NDC space ───────────────
    var xcov = 0.0; var xwgt = 0.0;
    var ycov = 0.0; var ywgt = 0.0;

    for (var i = 0u; i < in.n_fill_curves; i++) {
        let f  = (in.fill_curve_start + i) * 9u;
        let p1w = vec3<f32>(quads[f      ], quads[f + 1u], quads[f + 2u]);
        let p2w = vec3<f32>(quads[f + 3u], quads[f + 4u], quads[f + 5u]);
        let p3w = vec3<f32>(quads[f + 6u], quads[f + 7u], quads[f + 8u]);

        // Transform world → NDC, shift so the current fragment is origin.
        let c1 = pv * vec4<f32>(p1w, 1.0);
        let c2 = pv * vec4<f32>(p2w, 1.0);
        let c3 = pv * vec4<f32>(p3w, 1.0);
        let p1 = c1.xy/c1.w - in.ndc_xy;
        let p2 = c2.xy/c2.w - in.ndc_xy;
        let p3 = c3.xy/c3.w - in.ndc_xy;

        // Horizontal ray (x-coverage accumulation).
        let hcode = calc_root_code(p1.y, p2.y, p3.y);
        if hcode != 0u {
            let r = solve_horiz(p1, p2, p3) * pixels_per_ndc.x;
            if (hcode & 1u) != 0u {
                xcov += clamp(r.x + 0.5, 0.0, 1.0);
                xwgt  = max(xwgt, clamp(1.0 - abs(r.x)*2.0, 0.0, 1.0));
            }
            if hcode > 1u {
                xcov -= clamp(r.y + 0.5, 0.0, 1.0);
                xwgt  = max(xwgt, clamp(1.0 - abs(r.y)*2.0, 0.0, 1.0));
            }
        }

        // Vertical ray (y-coverage accumulation).
        let vcode = calc_root_code(p1.x, p2.x, p3.x);
        if vcode != 0u {
            let r = solve_vert(p1, p2, p3) * pixels_per_ndc.y;
            if (vcode & 1u) != 0u {
                ycov -= clamp(r.x + 0.5, 0.0, 1.0);
                ywgt  = max(ywgt, clamp(1.0 - abs(r.x)*2.0, 0.0, 1.0));
            }
            if vcode > 1u {
                ycov += clamp(r.y + 0.5, 0.0, 1.0);
                ywgt  = max(ywgt, clamp(1.0 - abs(r.y)*2.0, 0.0, 1.0));
            }
        }
    }

    var fill_cov = 0.0;
    if in.n_fill_curves > 0u {
        if in.fill_rule == 1u {
            fill_cov = calc_coverage_evenodd(xcov, ycov, xwgt, ywgt);
        } else {
            fill_cov = calc_coverage(xcov, ycov, xwgt, ywgt);
        }
    }

    // ── Stroke: SDF minimum distance in physical pixel space ──────────────
    // stroke_half_ndc is in NDC units; pixels_per_ndc.x converts to pixels.
    // (stroke_half_ndc was calibrated using pm[0,0], the NDC x-scale.)
    let stroke_half_px = in.v_stroke_half_ndc * pixels_per_ndc.x;
    var min_dist_px = 1e9;

    for (var i = 0u; i < in.n_stroke_curves; i++) {
        let f   = (in.stroke_curve_start + i) * 9u;
        let p1w = vec3<f32>(quads[f      ], quads[f + 1u], quads[f + 2u]);
        let p2w = vec3<f32>(quads[f + 3u], quads[f + 4u], quads[f + 5u]);
        let p3w = vec3<f32>(quads[f + 6u], quads[f + 7u], quads[f + 8u]);

        let c1 = pv * vec4<f32>(p1w, 1.0);
        let c2 = pv * vec4<f32>(p2w, 1.0);
        let c3 = pv * vec4<f32>(p3w, 1.0);
        // NDC-relative coordinates (fragment at origin), then scaled to pixels.
        let n1 = c1.xy/c1.w - in.ndc_xy;
        let n2 = c2.xy/c2.w - in.ndc_xy;
        let n3 = c3.xy/c3.w - in.ndc_xy;
        let p1_px = n1 * pixels_per_ndc;
        let p2_px = n2 * pixels_per_ndc;
        let p3_px = n3 * pixels_per_ndc;

        let d = min_dist_to_quad_px(p1_px, p2_px, p3_px);
        min_dist_px = min(min_dist_px, d);
    }

    // Smooth SDF: 1 within stroke, 0 outside, ½-pixel anti-aliased transition.
    var stroke_cov = 0.0;
    if in.n_stroke_curves > 0u && stroke_half_px > 0.0 {
        stroke_cov = clamp(stroke_half_px + 0.5 - min_dist_px, 0.0, 1.0);
    }

    // ── Porter-Duff "over": stroke on top of fill ─────────────────────────
    let fill_a   = in.v_fill_color.a   * fill_cov;
    let stroke_a = in.v_stroke_color.a * stroke_cov;
    let total_a  = stroke_a + fill_a * (1.0 - stroke_a);

    if total_a <= 0.001 { discard; }

    let fill_rgb   = in.v_fill_color.rgb;
    let stroke_rgb = in.v_stroke_color.rgb;
    let out_rgb    = (stroke_a * stroke_rgb + fill_a * (1.0 - stroke_a) * fill_rgb) / total_a;

    return vec4<f32>(out_rgb, total_a);
}
