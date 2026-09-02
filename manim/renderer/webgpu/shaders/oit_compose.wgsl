// WebGPU OIT composition shader.
//
// Reads the two OIT accumulation textures produced by surface_oit.wgsl and
// composites the transparent geometry result onto the existing opaque framebuffer.
//
// No vertex buffer is needed — three vertices are generated from the built-in
// vertex index, covering the full screen with a single oversized triangle.
//
// The output is alpha-blended onto the main render texture using standard
// {src-alpha, one-minus-src-alpha} blending, so the pipeline must be
// configured with that blend mode.

@group(0) @binding(0) var oit_accum  : texture_2d<f32>;  // rgba16float
@group(0) @binding(1) var oit_reveal : texture_2d<f32>;  // rgba16float

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
};

// Full-screen triangle: 3 vertices cover [-1,1]x[-1,1] without a VBO.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos[vi], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coord  = vec2<i32>(in.clip_position.xy);
    let accum  = textureLoad(oit_accum,  coord, 0);
    let reveal = textureLoad(oit_reveal, coord, 0).r;

    // Nothing accumulated at this pixel — don't touch the framebuffer.
    if (accum.a < 1e-5) { discard; }

    // Weighted average colour.
    let avg_color = accum.rgb / accum.a;

    // Overall opacity: 1 − ∏(1 − αᵢ).
    // `reveal` started at 1 and each fragment multiplied it by (1 − α),
    // so reveal == ∏(1 − αᵢ) and 1 − reveal is the accumulated opacity.
    let alpha = clamp(1.0 - reveal, 0.0, 1.0);

    // Output with premultiplied alpha so the {src-alpha, one-minus-src-alpha}
    // pipeline blend correctly composites over the opaque framebuffer.
    return vec4<f32>(avg_color, alpha);
}
