// GPU compute shader: cubic Bezier → quadratic approximations.
//
// Each thread converts one cubic Bezier (4 × 3-D control points) into four
// quadratic Beziers (3 points each) using two levels of de Casteljau
// subdivision at t = 0.5 followed by midpoint degree-reduction.
//
// This runs in the same command encoder as the render pass (before it),
// so WebGPU's implicit pass ordering gives a barrier — the render shader
// safely reads the output quads without an explicit synchronisation step.
//
// Buffer layout
// -------------
// binding 0  in_cubics  read-only-storage  array<f32>
//   12 floats per cubic: b0.xyz, b1.xyz, b2.xyz, b3.xyz (tightly packed)
//
// binding 1  out_quads  storage (read_write)  array<f32>
//   36 floats per input cubic (4 quads × 9 floats):
//   quad k: p0.xyz, pmid.xyz, p1.xyz  (start, control, end)
//   Order: [sub-cubic 0, sub-cubic 1, sub-cubic 2, sub-cubic 3]
//
// binding 2  params  uniform
//   offset 0: n_cubics  u32
//   (padded to 16 bytes)
//
// Dispatch: ceil(n_cubics / 64) workgroups × 1 × 1, workgroup_size = 64.

struct Params { n_cubics : u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read>       in_cubics : array<f32>;
@group(0) @binding(1) var<storage, read_write> out_quads : array<f32>;
@group(0) @binding(2) var<uniform>             params    : Params;

// Write one quadratic (p0, pmid, p2) as 9 consecutive floats at base.
fn write_quad(base: u32, p0: vec3<f32>, pmid: vec3<f32>, p2: vec3<f32>) {
    out_quads[base     ] = p0.x;   out_quads[base + 1u] = p0.y;   out_quads[base + 2u] = p0.z;
    out_quads[base + 3u] = pmid.x; out_quads[base + 4u] = pmid.y; out_quads[base + 5u] = pmid.z;
    out_quads[base + 6u] = p2.x;   out_quads[base + 7u] = p2.y;   out_quads[base + 8u] = p2.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_cubics { return; }

    // Read 4 control points of this cubic.
    let bi = idx * 12u;
    let b0 = vec3<f32>(in_cubics[bi     ], in_cubics[bi +  1u], in_cubics[bi +  2u]);
    let b1 = vec3<f32>(in_cubics[bi +  3u], in_cubics[bi +  4u], in_cubics[bi +  5u]);
    let b2 = vec3<f32>(in_cubics[bi +  6u], in_cubics[bi +  7u], in_cubics[bi +  8u]);
    let b3 = vec3<f32>(in_cubics[bi +  9u], in_cubics[bi + 10u], in_cubics[bi + 11u]);

    // ── Level 1: split [b0, b1, b2, b3] at t = 0.5 ──────────────────────
    // Left half:  [b0, m01, m012, m0123]
    // Right half: [m0123, m123, m23, b3]
    let m01   = (b0 + b1) * 0.5;
    let m12   = (b1 + b2) * 0.5;
    let m23   = (b2 + b3) * 0.5;
    let m012  = (m01 + m12) * 0.5;
    let m123  = (m12 + m23) * 0.5;
    let m0123 = (m012 + m123) * 0.5;

    // ── Level 2a: split left half [b0, m01, m012, m0123] at t = 0.5 ──────
    let lm01   = (b0 + m01) * 0.5;
    let lm12   = (m01 + m012) * 0.5;
    let lm23   = (m012 + m0123) * 0.5;
    let lm012  = (lm01 + lm12) * 0.5;
    let lm123  = (lm12 + lm23) * 0.5;
    let lm0123 = (lm012 + lm123) * 0.5;
    // Quad 0: [b0, lm01, lm012, lm0123]
    // Quad 1: [lm0123, lm123, lm23, m0123]

    // ── Level 2b: split right half [m0123, m123, m23, b3] at t = 0.5 ─────
    let rm01   = (m0123 + m123) * 0.5;
    let rm12   = (m123 + m23) * 0.5;
    let rm23   = (m23 + b3) * 0.5;
    let rm012  = (rm01 + rm12) * 0.5;
    let rm123  = (rm12 + rm23) * 0.5;
    let rm0123 = (rm012 + rm123) * 0.5;
    // Quad 2: [m0123, rm01, rm012, rm0123]
    // Quad 3: [rm0123, rm123, rm23, b3]

    // Write 4 quadratics.  Degree reduction: mid-handle = (h0 + h1) * 0.5.
    let bo = idx * 36u;
    write_quad(bo      , b0,     (lm01  + lm012) * 0.5, lm0123);
    write_quad(bo +  9u, lm0123, (lm123 + lm23 ) * 0.5, m0123 );
    write_quad(bo + 18u, m0123,  (rm01  + rm012) * 0.5, rm0123);
    write_quad(bo + 27u, rm0123, (rm123 + rm23 ) * 0.5, b3    );
}
