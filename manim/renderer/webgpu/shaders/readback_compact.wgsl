// GPU compact-readback compute shader.
//
// Reads every pixel from the bgra8unorm render texture and writes tightly-packed
// RGBA bytes into a storage buffer — one u32 per pixel, little-endian:
//   byte 0 = R,  byte 1 = G,  byte 2 = B,  byte 3 = A
//
// Two CPU operations are eliminated in a single pass:
//
//   1. Row-padding strip
//      copy_texture_to_buffer requires bytes_per_row to be a multiple of 256.
//      The CPU previously looped over every row to remove the padding bytes.
//      Here each thread writes directly to the tight index  y*width + x,
//      so the output buffer is already compact — no post-processing needed.
//
//   2. B↔R channel swap
//      The render texture is bgra8unorm (GPU memory layout: B G R A).
//      textureLoad() always returns components as (r, g, b, a) regardless of
//      the physical layout, so the output is already in RGBA byte order.
//      The CPU numpy channel-swap is no longer required.
//
// Workgroup size 16×16 = 256 threads.  Each thread handles one pixel.
// Caller dispatches ceil(width/16) × ceil(height/16) workgroups; the
// out-of-bounds guard below is a no-op for tiles that fit exactly.

@group(0) @binding(0) var  src_tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(src_tex);

    // Discard threads outside the image boundary (last tile edge).
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    // textureLoad returns (r, g, b, a) as normalized f32 regardless of bgra
    // memory layout — no manual component swap needed.
    let c = textureLoad(src_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);

    let r = u32(clamp(c.r * 255.0 + 0.5, 0.0, 255.0));
    let g = u32(clamp(c.g * 255.0 + 0.5, 0.0, 255.0));
    let b = u32(clamp(c.b * 255.0 + 0.5, 0.0, 255.0));
    let a = u32(clamp(c.a * 255.0 + 0.5, 0.0, 255.0));

    // Pack as little-endian u32: byte0=R, byte1=G, byte2=B, byte3=A.
    // NumPy / PIL both read this as RGBA when the buffer is reinterpreted as
    // uint8 in row-major order.
    dst[gid.y * dims.x + gid.x] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
