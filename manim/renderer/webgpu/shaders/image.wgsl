// Image quad shader.
//
// Renders a textured quad from four world-space corner vertices.
// Used by WebGPURenderer to draw ImageMobject instances.
//
// Uniform layout (group 0, binding 0) — same 656-byte block as the surface
// shaders; only projection and view are used here:
//   offset  0 — projection  mat4x4<f32>  64 B
//   offset 64 — view        mat4x4<f32>  64 B
//   (remaining bytes are lighting fields, unused by this shader)
//
// Texture / sampler (group 1):
//   binding 0 — texture_2d<f32>  (rgba8unorm uploaded as f32 [0,1] per channel)
//   binding 1 — sampler          (linear, clamp-to-edge)
//
//   UV origin is top-left (matches WebGPU texture layout and Manim pixel_array
//   row-major order: row 0 = top).  No y-flip is needed.
//
// Tint uniform (group 2, binding 0) — 16-byte block:
//   offset  0 — rgb   vec3<f32>  12 B   per-channel colour multiplier
//   offset 12 — _pad  f32         4 B   alignment padding (unused)
//
//   Default white (1, 1, 1) is identity — texture is returned unchanged.
//   Set via mob.color; populated by the renderer from mob.color.to_rgb().
//
// Vertex attributes (stride 20 bytes):
//   location 0 — in_pos  float32x3  offset  0
//   location 1 — in_uv   float32x2  offset 12

struct Uniforms {
    projection : mat4x4<f32>,
    view       : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

@group(1) @binding(0) var img_texture : texture_2d<f32>;
@group(1) @binding(1) var img_sampler : sampler;

struct TintUniforms {
    rgb  : vec3<f32>,
    _pad : f32,
};
@group(2) @binding(0) var<uniform> tint : TintUniforms;

struct VertexInput {
    @location(0) in_pos : vec3<f32>,
    @location(1) in_uv  : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0)       uv       : vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.projection * u.view * vec4<f32>(in.in_pos, 1.0);
    out.uv = in.in_uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(img_texture, img_sampler, in.uv);
    // Multiply RGB by the tint; alpha is taken from the texture as-is.
    return vec4<f32>(sample.rgb * tint.rgb, sample.a);
}
