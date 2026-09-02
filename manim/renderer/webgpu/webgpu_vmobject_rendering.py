"""WebGPU draw calls for VMobject fill + stroke + surface rendering.

Fill + Stroke (combined pipeline)
----------------------------------
``cubic_to_quads.wgsl`` — GPU compute shader converts raw cubic Bezier
control points to quadratic approximations (4 per cubic, two-level de
Casteljau subdivision).

``vmobject_fill_stroke.wgsl`` — combined render shader: one bounding quad
per object, one fragment loop accumulates both Slug winding-number fill
coverage (in NDC space) and SDF stroke distance (in pixel space).  Porter-
Duff "over" compositing produces the final colour.

Closing segments
~~~~~~~~~~~~~~~~
Every open subpath gets a linear closing cubic (degree-elevated line from
the last anchor back to the first) appended to the fill cubic list.  This
makes the winding-number integral correct for partial paths (e.g. during
``Create`` animations).  The closing cubic is NOT added to the stroke cubic
list — strokes should follow the visible part of the curve only.

Surfaces
--------
Parametric surfaces use a combined triangle-mesh pipeline
(``surface_combined.wgsl`` / ``surface_oit.wgsl``) with Phong lighting and
barycentric wireframe in a single draw call.  The centroid vertex of each
triangle fan carries bary=(1,0,0); the outer edge (anchor_i ↔ anchor_{i+1})
has bary.x=0 — this is the visible mesh-grid edge.  Transparent surfaces go
through the OIT accumulation + composition passes.

Batching
--------
``collect_frame_data`` tessellates *all* scene mobjects on the CPU, uploads
one cubics buffer (fill then stroke, all objects) and one vertex buffer (one
bounding quad per object), then returns a ``_FrameData`` ready for the GPU.
``draw_frame_data`` records draw calls into the active render pass.
"""

from __future__ import annotations

import struct
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from manim.mobject.three_d.dot_cloud import DotCloud3D
from manim.mobject.three_d.three_dimensions import Surface
from manim.mobject.types.vectorized_mobject import VMobject

if TYPE_CHECKING:
    import wgpu as wgpu_t

    from manim.renderer.webgpu.webgpu_renderer import WebGPURenderer


# ---------------------------------------------------------------------------
# Combined surface vertex layout — must match surface_combined.wgsl /
# surface_oit.wgsl locations.
#
#   location 0 — in_vert            float32x3  offset  0  (12 B)
#   location 1 — in_normal          float32x3  offset 12  (12 B)
#   location 2 — in_fill_color      float32x4  offset 24  (16 B)
#   location 3 — in_stroke_color    float32x4  offset 40  (16 B)
#   location 4 — in_bary            float32x3  offset 56  (12 B)
#   location 5 — stroke_half_px     float32    offset 68  ( 4 B)
#   location 6 — diffuse_strength   float32    offset 72  ( 4 B)
#   location 7 — specular_strength  float32    offset 76  ( 4 B)
#   location 8 — specular_exponent  float32    offset 80  ( 4 B)
#   stride: 84 bytes
# ---------------------------------------------------------------------------

_SURFACE_COMBINED_DTYPE = np.dtype(
    [
        ("in_vert", np.float32, (3,)),
        ("in_normal", np.float32, (3,)),
        ("in_fill_color", np.float32, (4,)),
        ("in_stroke_color", np.float32, (4,)),
        ("in_bary", np.float32, (3,)),
        ("stroke_half_px", np.float32),
        ("diffuse_strength", np.float32),
        ("specular_strength", np.float32),
        ("specular_exponent", np.float32),
    ]
)
_SURFACE_COMBINED_STRIDE: int = _SURFACE_COMBINED_DTYPE.itemsize  # 84 bytes

_SURFACE_COMBINED_OFFSETS: dict[str, int] = {
    name: _SURFACE_COMBINED_DTYPE.fields[name][1]  # type: ignore[index]
    for name in _SURFACE_COMBINED_DTYPE.names
}

SURFACE_COMBINED_VERTEX_LAYOUT: dict = {
    "array_stride": _SURFACE_COMBINED_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        {
            "format": "float32x3",
            "offset": _SURFACE_COMBINED_OFFSETS["in_vert"],
            "shader_location": 0,
        },
        {
            "format": "float32x3",
            "offset": _SURFACE_COMBINED_OFFSETS["in_normal"],
            "shader_location": 1,
        },
        {
            "format": "float32x4",
            "offset": _SURFACE_COMBINED_OFFSETS["in_fill_color"],
            "shader_location": 2,
        },
        {
            "format": "float32x4",
            "offset": _SURFACE_COMBINED_OFFSETS["in_stroke_color"],
            "shader_location": 3,
        },
        {
            "format": "float32x3",
            "offset": _SURFACE_COMBINED_OFFSETS["in_bary"],
            "shader_location": 4,
        },
        {
            "format": "float32",
            "offset": _SURFACE_COMBINED_OFFSETS["stroke_half_px"],
            "shader_location": 5,
        },
        {
            "format": "float32",
            "offset": _SURFACE_COMBINED_OFFSETS["diffuse_strength"],
            "shader_location": 6,
        },
        {
            "format": "float32",
            "offset": _SURFACE_COMBINED_OFFSETS["specular_strength"],
            "shader_location": 7,
        },
        {
            "format": "float32",
            "offset": _SURFACE_COMBINED_OFFSETS["specular_exponent"],
            "shader_location": 8,
        },
    ],
}


# ---------------------------------------------------------------------------
# Combined fill+stroke vertex layout — must match vmobject_fill_stroke.wgsl.
#
#   location 0 — in_pos             float32x3  offset  0   (12 B)
#   location 1 — in_fill_color      float32x4  offset 12   (16 B)
#   location 2 — in_stroke_color    float32x4  offset 28   (16 B)
#   location 3 — stroke_half_ndc    float32    offset 44   ( 4 B)
#   location 4 — fill_curve_start   uint32     offset 48   ( 4 B)
#   location 5 — n_fill_curves      uint32     offset 52   ( 4 B)
#   location 6 — stroke_curve_start uint32     offset 56   ( 4 B)
#   location 7 — n_stroke_curves    uint32     offset 60   ( 4 B)
#   location 8 — fill_rule          uint32     offset 64   ( 4 B)  0=nonzero, 1=evenodd
#   stride: 68 bytes
# ---------------------------------------------------------------------------

_FILL_STROKE_DTYPE = np.dtype(
    [
        ("in_pos", np.float32, (3,)),
        ("in_fill_color", np.float32, (4,)),
        ("in_stroke_color", np.float32, (4,)),
        ("stroke_half_ndc", np.float32),
        ("fill_curve_start", np.uint32),
        ("n_fill_curves", np.uint32),
        ("stroke_curve_start", np.uint32),
        ("n_stroke_curves", np.uint32),
        ("fill_rule", np.uint32),
    ]
)
_FILL_STROKE_STRIDE: int = _FILL_STROKE_DTYPE.itemsize  # 64 bytes

_FILL_STROKE_OFFSETS: dict[str, int] = {
    name: _FILL_STROKE_DTYPE.fields[name][1]  # type: ignore[index]
    for name in _FILL_STROKE_DTYPE.names
}

FILL_STROKE_VERTEX_LAYOUT: dict = {
    "array_stride": _FILL_STROKE_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        {
            "format": "float32x3",
            "offset": _FILL_STROKE_OFFSETS["in_pos"],
            "shader_location": 0,
        },
        {
            "format": "float32x4",
            "offset": _FILL_STROKE_OFFSETS["in_fill_color"],
            "shader_location": 1,
        },
        {
            "format": "float32x4",
            "offset": _FILL_STROKE_OFFSETS["in_stroke_color"],
            "shader_location": 2,
        },
        {
            "format": "float32",
            "offset": _FILL_STROKE_OFFSETS["stroke_half_ndc"],
            "shader_location": 3,
        },
        {
            "format": "uint32",
            "offset": _FILL_STROKE_OFFSETS["fill_curve_start"],
            "shader_location": 4,
        },
        {
            "format": "uint32",
            "offset": _FILL_STROKE_OFFSETS["n_fill_curves"],
            "shader_location": 5,
        },
        {
            "format": "uint32",
            "offset": _FILL_STROKE_OFFSETS["stroke_curve_start"],
            "shader_location": 6,
        },
        {
            "format": "uint32",
            "offset": _FILL_STROKE_OFFSETS["n_stroke_curves"],
            "shader_location": 7,
        },
        {
            "format": "uint32",
            "offset": _FILL_STROKE_OFFSETS["fill_rule"],
            "shader_location": 8,
        },
    ],
}


# ---------------------------------------------------------------------------
# TrueDot vertex layout — must match true_dot.wgsl locations.
#
#   location 0 — center  float32x3  offset  0  (12 B)
#   location 1 — color   float32x4  offset 12  (16 B)
#   location 2 — uv      float32x2  offset 28  ( 8 B)
#   location 3 — radius  float32    offset 36  ( 4 B)
#   location 4 — gloss   float32    offset 40  ( 4 B)
#   location 5 — shadow  float32    offset 44  ( 4 B)
#   stride: 48 bytes
# ---------------------------------------------------------------------------

_TRUE_DOT_DTYPE = np.dtype(
    [
        ("center", np.float32, (3,)),
        ("color", np.float32, (4,)),
        ("uv", np.float32, (2,)),
        ("radius", np.float32),
        ("gloss", np.float32),
        ("shadow", np.float32),
    ]
)
_TRUE_DOT_STRIDE: int = _TRUE_DOT_DTYPE.itemsize  # 48 bytes

_TRUE_DOT_OFFSETS: dict[str, int] = {
    name: _TRUE_DOT_DTYPE.fields[name][1]  # type: ignore[index]
    for name in _TRUE_DOT_DTYPE.names
}

TRUE_DOT_VERTEX_LAYOUT: dict = {
    "array_stride": _TRUE_DOT_STRIDE,
    "step_mode": "vertex",
    "attributes": [
        {
            "format": "float32x3",
            "offset": _TRUE_DOT_OFFSETS["center"],
            "shader_location": 0,
        },
        {
            "format": "float32x4",
            "offset": _TRUE_DOT_OFFSETS["color"],
            "shader_location": 1,
        },
        {
            "format": "float32x2",
            "offset": _TRUE_DOT_OFFSETS["uv"],
            "shader_location": 2,
        },
        {
            "format": "float32",
            "offset": _TRUE_DOT_OFFSETS["radius"],
            "shader_location": 3,
        },
        {
            "format": "float32",
            "offset": _TRUE_DOT_OFFSETS["gloss"],
            "shader_location": 4,
        },
        {
            "format": "float32",
            "offset": _TRUE_DOT_OFFSETS["shadow"],
            "shader_location": 5,
        },
    ],
}

# Corner UV offsets for the two triangles that form a screen-aligned quad:
#   triangle 0: (BL, BR, TL)  → corners 0,1,2
#   triangle 1: (BR, TR, TL)  → corners 1,3,2
# x_sign: -1 +1 -1 +1   y_sign: -1 -1 +1 +1
_QUAD_UVS = np.array(
    [
        [-1.0, -1.0],  # BL (0)
        [1.0, -1.0],  # BR (1)
        [-1.0, 1.0],  # TL (2)
        [1.0, -1.0],  # BR (1)  ← repeated for 2nd triangle
        [1.0, 1.0],  # TR (3)
        [-1.0, 1.0],  # TL (2)  ← repeated
    ],
    dtype=np.float32,
)  # shape (6, 2)


def build_true_dot_vbo(
    mob: DotCloud3D,
) -> np.ndarray | None:
    """Expand a ``DotCloud3D`` into a flat vertex array for TrueDot rendering.

    Each point becomes 6 vertices (2 triangles) forming a screen-aligned quad.
    UV coords span (−1,−1) → (1,1); the radius is in world-space scene units.

    Returns ``None`` if the mob has no renderable points.
    """
    pts = mob.get_cloud_points()
    rgbas = mob.get_rgbas()
    radius = mob.dot_radius
    gloss = mob.gloss
    shadow = mob.shadow

    pts = np.asarray(pts, dtype=np.float32)  # (N, 3)
    N = len(pts)
    if N == 0:
        return None

    # Broadcast rgbas to (N, 4).
    if rgbas is None or len(rgbas) == 0:
        rgba = np.ones((N, 4), dtype=np.float32)
    else:
        rgbas = np.asarray(rgbas, dtype=np.float32)
        if len(rgbas) == 1:
            rgba = np.repeat(rgbas[:1], N, axis=0)
        elif len(rgbas) < N:
            # Resize with interpolation (matches OpenGL behaviour).
            indices = np.round(np.linspace(0, len(rgbas) - 1, N)).astype(int)
            rgba = rgbas[indices]
        else:
            rgba = rgbas[:N]

    # Expand N points → N×6 vertices.
    pts_rep = np.repeat(pts, 6, axis=0)  # (N*6, 3)
    rgba_rep = np.repeat(rgba, 6, axis=0)  # (N*6, 4)
    uvs = np.tile(_QUAD_UVS, (N, 1))  # (N*6, 2)

    arr = np.zeros(N * 6, dtype=_TRUE_DOT_DTYPE)
    arr["center"] = pts_rep
    arr["color"] = rgba_rep
    arr["uv"] = uvs
    arr["radius"] = radius
    arr["gloss"] = gloss
    arr["shadow"] = shadow
    return arr


# ---------------------------------------------------------------------------
# Per-frame data container
# ---------------------------------------------------------------------------


@dataclass
class _FrameData:
    """All GPU-ready data for one group of mobjects (one camera bind group).

    Produced by ``collect_frame_data``; consumed by ``draw_frame_data`` and
    the caller's OIT / fixed-frame passes.
    """

    # VMobject fill+stroke via combined pipeline
    fs_parts: list[np.ndarray]  # _FILL_STROKE_DTYPE arrays, one per draw call
    fs_buf: wgpu_t.GPUBuffer | None  # concatenated vertex buffer
    fs_byte_offsets: list[int]  # byte offset of each part in fs_buf

    # GPU compute: cubic → quadratic conversion
    cubics_buf: wgpu_t.GPUBuffer | None  # input (12 floats/cubic), all objects
    quads_out_buf: wgpu_t.GPUBuffer | None  # output (36 floats/cubic = 4 quads × 9)
    n_cubics_total: int
    compute_bg: wgpu_t.GPUBindGroup | None  # compute pass bind group
    render_bg: wgpu_t.GPUBindGroup | None  # fragment bind group (camera + quads)

    # Parametric surfaces (combined fill + barycentric wireframe pipeline)
    surface_parts: list[np.ndarray]
    surface_buf: wgpu_t.GPUBuffer | None
    surface_byte_offsets: list[int]

    # Ordered draw commands:
    #   "fill_stroke_2d"  — 2-D VMobject (no depth write)
    #   "fill_stroke_3d"  — shade_in_3d VMobject (depth write + test)
    #   "surface_opaque"  — opaque parametric surface
    #   "surface_oit"     — transparent parametric surface (OIT pass, caller handles)
    draw_plan: list[tuple[str, int]]

    # Indices into surface_parts that need OIT (handled by the caller).
    oit_indices: list[int]


# ---------------------------------------------------------------------------
# Geometry caches
# ---------------------------------------------------------------------------

# fill_stroke_cache: vmobject → (points_hash, (fill_cubics, stroke_cubics))
#   fill_cubics  : (N, 4, 3) float32 — includes closing segments for winding
#   stroke_cubics: (M, 4, 3) float32 — no closing segments (visible curve only)
# Geometry only; colors/widths are fetched fresh every frame.
_fill_stroke_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# surface_mob_cache: Surface mob →
#   (geom_hash, color_hash, big_template, seg_starts, seg_ends,
#    sw_arr, sa_arr, has_sw, draw_cmds)
#
#   geom_hash   : bytes — hash of patch geometry + material ONLY (no colors).
#                 Unchanged when FadeIn/set_fill changes opacity.
#   color_hash  : bytes — hash of fill_rgba, stroke_rgba, stroke_width per patch.
#                 Changes on FadeIn/set_fill without requiring retessellation.
#   big_template: single concatenated _SURFACE_COMBINED_DTYPE array with
#                 already-smoothed normals; colors reflect the last stored frame;
#                 stroke_half_px == 0.0 (recomputed per-frame on each hit).
#   seg_starts/ends: numpy intp arrays of part boundaries in big_template.
#   sw_arr/sa_arr/has_sw: stroke metadata, updated in-place on color misses.
#   draw_cmds   : list[str] — "surface_opaque" or "surface_oit" per part;
#                 updated in-place on color misses (opacity class can change).
#
# Cache hit states:
#   geom HIT + color HIT  → just recompute stroke_half_px (camera rotation path)
#   geom HIT + color MISS → patch colors in big_template + recompute stroke_half_px
#                           (FadeIn / set_fill without geometry change; O(N_parts))
#   geom MISS             → full retessellation + normal smoothing
_surface_mob_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# surface_color_memo: Surface mob →
#   (fill_cols, stroke_cols, sw_arr, sa_arr)
#   fill_cols   : float32 (N_active, 4) — fill RGBA per active part
#   stroke_cols : float32 (N_active, 4) — stroke RGBA per active part
#   sw_arr      : float32 (N_active,)   — stroke width per active part
#   sa_arr      : float32 (N_active,)   — stroke alpha per active part
#
# Populated by _surface_hash_pair whenever the color hash is recomputed
# (color-only miss or full miss path).  The color-only update path in
# collect_frame_data reads from here instead of re-iterating all submobjects,
# saving ~2 full O(N_submobs) passes per Surface per color-changing frame.
_surface_color_memo: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _points_hash(vmobject: VMobject) -> int:
    pts = vmobject.points
    if pts.size == 0:
        return 0
    # Cast to float32 before hashing so that sub-epsilon float64 noise
    # (e.g. from FadeIn / Transform's straight_path interpolation, which
    # produces ~1e-17 differences when start == end) does not create
    # spurious cache misses.  Genuine geometry changes are at least
    # float32-epsilon (~1e-7) in magnitude and are still detected.
    return hash(pts.astype(np.float32).tobytes())


# _surface_geom_hash_memo: Surface mob → (fast_geom_id, fast_color_id, geom_hash, color_hash)
# fast_geom_id  — XOR of id(s.points) for all submobs.
# fast_color_id — XOR of id(fill_rgbas) ^ id(stroke_rgbas) for all submobs.
# Separately tracking the two fast IDs lets us skip recomputing the geometry hash
# on a color-only change without missing a genuine geometry update.
_surface_geom_hash_memo: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _surface_hash_pair(
    mob: Surface,
    submobs: list | None = None,
) -> tuple[bytes, bytes]:
    """Return ``(geom_hash, color_hash)`` for *mob*.

    ``geom_hash``  covers patch point positions + material params (diffuse,
    specular, specular_exp).  It does NOT include fill/stroke colors, so a
    FadeIn that only changes opacity does not invalidate it.

    ``color_hash`` is ``fast_color_id`` packed as 8 bytes, where
    ``fast_color_id`` is the XOR of ``id(fill_rgbas)`` / ``id(stroke_rgbas)``
    for all submobject patches.  It changes whenever Manim replaces any color
    array (which happens on every FadeIn / set_fill frame).

    Two-level memoisation avoids full per-submob rehashing on every frame:
    the fast IDs (XOR of array ``id()``s) detect changes in O(N_submobs)
    without touching array data; the slow geometry hash runs only on a
    geometry miss (first call or actual point change).

    *submobs* — optional pre-computed ``mob.family_members_with_points()``.
    Pass this from ``collect_frame_data`` to avoid a redundant tree walk.
    """
    if submobs is None:
        submobs = mob.family_members_with_points()

    fast_geom_id = 0
    fast_color_id = 0
    for s in submobs:
        fast_geom_id ^= id(s.points)
        fast_color_id ^= id(getattr(s, "fill_rgbas", None))
        fast_color_id ^= id(getattr(s, "stroke_rgbas", None))

    memo = _surface_geom_hash_memo.get(mob)
    if memo is not None:
        cached_fgi, cached_fci, cached_gh, cached_ch = memo
        if cached_fgi == fast_geom_id and cached_fci == fast_color_id:
            # Both geometry and colors unchanged.
            return cached_gh, cached_ch
        if cached_fgi == fast_geom_id:
            # Geometry unchanged, colors changed — use fast_color_id as the
            # color hash (no submob method calls needed here).  The actual
            # color arrays are read lazily by collect_frame_data when it
            # applies the per-vertex update, so we skip the second O(N_submobs)
            # iteration entirely.
            new_ch = struct.pack("<q", fast_color_id)
            _surface_geom_hash_memo[mob] = (
                fast_geom_id,
                fast_color_id,
                cached_gh,
                new_ch,
            )
            # Invalidate stale color arrays so collect_frame_data reads fresh data.
            _surface_color_memo.pop(mob, None)
            return cached_gh, new_ch

    # Full recompute (first call or geometry change).
    geom_parts: list[bytes] = []
    for submob in submobs:
        phash = _points_hash(submob)
        geom_parts.append(phash.to_bytes(8, "little", signed=True))
        geom_parts.append(
            struct.pack(
                "<fff",
                float(getattr(submob, "diffuse_strength", 0.8)),
                float(getattr(submob, "specular_strength", 0.9)),
                float(getattr(submob, "specular_exponent", 16.0)),
            )
        )
    gh = b"".join(geom_parts)
    ch = struct.pack("<q", fast_color_id)
    _surface_geom_hash_memo[mob] = (fast_geom_id, fast_color_id, gh, ch)
    # Invalidate stale color arrays so collect_frame_data reads fresh data.
    _surface_color_memo.pop(mob, None)
    return gh, ch


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _fd_fingerprint(
    mobjects: list,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    center_view_matrix: np.ndarray | None = None,
) -> bytes:
    """Compute a compact fingerprint of the mobject set + camera state.

    Captures: view/projection matrices, per-submobject geometry hash, fill
    color, stroke color, and stroke width.  Two frames with identical
    fingerprints are guaranteed to produce pixel-identical renders.

    *center_view_matrix* — when provided (fixed-orientation path), it is
    included in the fingerprint so that camera rotation invalidates the cache
    even though *view_matrix* (the stripped fixed_view) is constant.

    Cost: O(n) over all leaf submobjects, but only does scalar reads and bytes
    operations — no numpy matrix math or buffer allocations.  Much cheaper
    than a full tessellation pass.
    """
    parts: list[bytes] = [view_matrix.tobytes(), proj_matrix.tobytes()]
    if center_view_matrix is not None:
        parts.append(center_view_matrix.tobytes())
    for mob in mobjects:
        for submob in mob.family_members_with_points():
            phash = _points_hash(submob)
            fill_rgba = submob.get_fill_rgbas()
            stroke_rgba = submob.get_stroke_rgbas()
            sw = float(submob.get_stroke_width()) if stroke_rgba.shape[0] > 0 else 0.0
            parts.append(struct.pack("<q", phash))
            parts.append(fill_rgba.tobytes())
            parts.append(stroke_rgba.tobytes())
            parts.append(struct.pack("<f", sw))
    return b"".join(parts)


def collect_frame_data(
    renderer: WebGPURenderer,
    mobjects: list,
    camera_uniform_buf: wgpu_t.GPUBuffer,
    view_matrix_override: np.ndarray | None = None,
    proj_matrix_override: np.ndarray | None = None,
    center_view_matrix: np.ndarray | None = None,
    cache_slot: str | None = None,
) -> _FrameData | None:
    """Tessellate *mobjects*, upload to GPU, return a ``_FrameData``.

    Does NOT record any GPU commands — only uploads buffers and creates bind
    groups.  The caller must run the compute pass (via ``_FrameData.compute_bg``)
    before the render pass.

    *camera_uniform_buf* is the 656-byte uniform buffer for this camera group.
    It is stored in the render bind group so the fragment shader can project
    world-space curve data into the correct NDC space.

    *view_matrix_override* / *proj_matrix_override* replace the camera's normal
    view and projection matrices for CPU-side bounding-quad computation.  Use
    these for fixed-in-frame and fixed-orientation mobjects so that the
    world-space quad vertices are consistent with the bind group the GPU will
    use to rasterise them.

    *center_view_matrix* — when provided, enables fixed-orientation rendering.
    Each submobject's bezier control points are pre-translated by
    ``(R_full - I) @ center_w`` so the object appears at the 3D-projected
    position of its world center while its local orientation stays upright
    (no camera rotation applied to the local shape).  The GPU shader still
    uses *view_matrix_override* (typically the rotation-stripped fixed_view),
    and the pre-translation makes the combined effect equivalent to Cairo's
    ``transform_points_pre_display`` for fixed-orientation objects.

    *cache_slot* — when not None, enables ``_FrameData`` caching for this
    call.  On a fingerprint hit the cached ``_FrameData`` is returned
    immediately, skipping all tessellation and GPU buffer uploads.  The
    ``camera_uniform_buf`` must be a *persistent* buffer (same Python object
    across frames) so that cached ``render_bg`` bind groups remain valid.
    """
    import wgpu

    view_matrix: np.ndarray = (
        view_matrix_override
        if view_matrix_override is not None
        else renderer.camera.view_matrix
    )
    proj_matrix: np.ndarray = (
        proj_matrix_override
        if proj_matrix_override is not None
        else renderer.camera.projection_matrix
    )

    # ── _FrameData cache check ────────────────────────────────────────────
    # When all mobs are Surface objects the _surface_mob_cache handles
    # geometry caching.  Skip _fd_fingerprint (it always misses when the
    # camera rotates) and disable _FrameData caching for this call — the
    # surface GPU buffer must be regenerated each frame to update
    # stroke_half_px.  Non-surface or mixed calls still use _fd_fingerprint.
    _all_surface_call: bool = bool(mobjects) and all(
        isinstance(m, Surface) for m in mobjects if isinstance(m, VMobject)
    )
    fp: bytes = b""  # populated below on non-surface-only paths
    if cache_slot is not None and mobjects and not _all_surface_call:
        fp = _fd_fingerprint(mobjects, view_matrix, proj_matrix, center_view_matrix)
        cached = renderer._fd_cache.get(cache_slot)
        if cached is not None and cached[0] == fp:
            # Scene + camera unchanged — return the cached GPU data directly.
            # The compute pass will re-dispatch into the same quads_out_buf
            # (safe: identical input → identical output; the render pass reads
            # it after the compute pass completes within the same encoder).
            return cached[1]
        # Cache miss — tessellate below, then store result before returning.

    # Per-draw-call data collected across all mobjects.
    fs_parts: list[np.ndarray] = []
    # Cubics: fill first (all objects), then stroke (all objects).
    all_fill_cubics: list[np.ndarray] = []  # (Ni, 4, 3) per draw call
    all_stroke_cubics: list[np.ndarray] = []  # (Mi, 4, 3) per draw call
    n_fill_cubics_per: list[int] = []  # Ni per draw call
    n_stroke_cubics_per: list[int] = []  # Mi per draw call

    surface_parts: list[np.ndarray] = []
    draw_plan: list[tuple[str, int]] = []

    # Tracks newly-tessellated Surface mobs (cache misses) so we can store
    # their parts in _surface_mob_cache after _smooth_surface_normals runs.
    # Each entry: (mob, geom_hash, parts_start_idx, parts_end_idx)
    _new_surface_mobs: list[tuple] = []

    # Guard against double-processing the same submobject.  This can happen
    # when mob_list is a flat family list (e.g. moving_mobjects from
    # begin_animations) that contains both a VGroup and its children: without
    # the guard, family_members_with_points() on the VGroup would process the
    # children, and then those children would be processed again individually.
    _seen_submobs: set[int] = set()

    use_z_index: bool = renderer.camera.use_z_index

    for mob in mobjects:
        if not isinstance(mob, VMobject):
            continue

        # ── Parametric Surface ────────────────────────────────────────────
        if isinstance(mob, Surface):
            surface_submobs = mob.family_members_with_points()
            if use_z_index:
                surface_submobs = sorted(surface_submobs, key=lambda m: m.z_index)
            # Read material params from the parent Surface as defaults; each
            # submobject patch may override them individually by carrying its
            # own diffuse_strength / specular_strength / specular_exponent
            # instance attribute (set via set_*_by_func or direct assignment).
            surf_diffuse = float(getattr(mob, "diffuse_strength", 0.8))
            surf_specular = float(getattr(mob, "specular_strength", 0.9))
            surf_spec_exp = float(getattr(mob, "specular_exponent", 16.0))

            # ── Surface geometry cache ────────────────────────────────────
            # The geometry (verts, normals, bary, colors, material) does NOT
            # depend on view_matrix / proj_matrix — only stroke_half_px does.
            # Cache the fully-tessellated + smoothed arrays per mob so that
            # a camera-only change (ambient rotation, etc.) skips the expensive
            # per-patch Python loop and just updates stroke_half_px.
            geom_hash, color_hash = _surface_hash_pair(mob, submobs=surface_submobs)
            cached_entry = _surface_mob_cache.get(mob)

            if cached_entry is not None and cached_entry[0] == geom_hash:
                # ── Geometry HIT ──────────────────────────────────────────
                # cached_entry = (geom_hash, color_hash, big_template,
                #                 seg_starts, seg_ends,
                #                 sw_arr, sa_arr, has_sw, draw_cmds)
                (
                    _,
                    cached_color_hash,
                    big_template,
                    seg_starts,
                    seg_ends,
                    sw_arr,
                    sa_arr,
                    has_sw,
                    draw_cmds,
                ) = cached_entry

                if cached_color_hash != color_hash:
                    # ── Color-only miss (FadeIn / set_fill / set_stroke) ──
                    # Read current colors from submobjects directly using fast
                    # attribute access (avoids method call overhead).  Build
                    # per-part color arrays then write them in a single
                    # vectorized np.repeat call instead of N_parts slice writes.
                    fill_list: list[np.ndarray] = []
                    stroke_list: list[np.ndarray] = []
                    sw_list: list[float] = []
                    sa_list: list[float] = []
                    for submob in surface_submobs:
                        if id(submob) in _seen_submobs:
                            continue
                        f_rgba = getattr(submob, "fill_rgbas", None)
                        s_rgba = getattr(submob, "stroke_rgbas", None)
                        if f_rgba is None or f_rgba.shape[0] == 0:
                            continue
                        if float(f_rgba[0, 3]) <= 0.0:
                            continue
                        has_stroke = s_rgba is not None and s_rgba.shape[0] > 0
                        fill_list.append(f_rgba[0].astype(np.float32))
                        stroke_list.append(
                            s_rgba[0].astype(np.float32)
                            if has_stroke
                            else np.zeros(4, dtype=np.float32)
                        )
                        sw_list.append(
                            float(submob.stroke_width) if has_stroke else 0.0
                        )
                        sa_list.append(float(s_rgba[0, 3]) if has_stroke else 0.0)

                    if len(fill_list) == len(draw_cmds):
                        fill_cols = np.array(fill_list, dtype=np.float32)
                        stroke_cols = np.array(stroke_list, dtype=np.float32)
                        sw_arr = np.array(sw_list, dtype=np.float32)
                        sa_arr = np.array(sa_list, dtype=np.float32)
                        has_sw = (sw_arr > 0.0) & (sa_arr > 0.001)
                        draw_cmds = [
                            "surface_opaque"
                            if float(fill_cols[i, 3]) >= 0.99
                            else "surface_oit"
                            for i in range(len(draw_cmds))
                        ]
                        # Vectorized color write: expand per-part colors to
                        # per-vertex with repeat counts, then assign in one op.
                        rep_counts = (seg_ends - seg_starts).astype(np.intp)
                        big_template["in_fill_color"] = np.repeat(
                            fill_cols, rep_counts, axis=0
                        )
                        big_template["in_stroke_color"] = np.repeat(
                            stroke_cols, rep_counts, axis=0
                        )
                        _surface_mob_cache[mob] = (
                            geom_hash,
                            color_hash,
                            big_template,
                            seg_starts,
                            seg_ends,
                            sw_arr,
                            sa_arr,
                            has_sw,
                            draw_cmds,
                        )
                    else:
                        # Part count changed — treat as full miss.
                        cached_entry = None

            if cached_entry is not None and cached_entry[0] == geom_hash:
                # ── Full HIT: copy template and recompute stroke_half_px ──
                big_copy = big_template.copy()
                vm = view_matrix.astype(np.float32)
                pm = proj_matrix.astype(np.float32)
                R, t = vm[:3, :3], vm[:3, 3]
                from manim import config as _cfg

                px_half = _cfg.pixel_width * 0.5
                pm_00 = abs(float(pm[0, 0]))
                pm_32 = float(pm[3, 2])
                pm_33 = float(pm[3, 3])

                # Vectorized stroke_half_px: one matrix multiply + reduceat
                # instead of per-part Python slice+mean inside a loop.
                z_vals = ((R @ big_copy["in_vert"].T).T + t)[:, 2]
                part_sizes = (seg_ends - seg_starts).astype(np.float32)

                # Per-part average view-space z via reduceat sum / count.
                z_sums = np.add.reduceat(z_vals, seg_starts)
                avg_z = z_sums / part_sizes  # (n_parts,)
                clip_w = pm_32 * avg_z + pm_33  # (n_parts,)
                clip_w = np.where(np.abs(clip_w) < 1e-8, 1.0, clip_w)

                shp = np.where(
                    has_sw,
                    0.004 * sw_arr * pm_00 / np.abs(clip_w) * px_half,
                    0.0,
                ).astype(np.float32)  # (n_parts,)

                # Write per-part stroke_half_px into the copy using index ranges.
                for i in range(len(draw_cmds)):
                    big_copy["stroke_half_px"][seg_starts[i] : seg_ends[i]] = shp[i]

                # Slice views for draw_plan / surface_parts.
                for i, cmd in enumerate(draw_cmds):
                    draw_plan.append((cmd, len(surface_parts)))
                    surface_parts.append(big_copy[seg_starts[i] : seg_ends[i]])

                # Mark submobs as seen so they aren't re-processed as VMobjects.
                for submob in surface_submobs:
                    _seen_submobs.add(id(submob))
                continue

            # ── Full MISS: tessellation + smoothing (original path) ───────
            # Collect (stroke_width, stroke_color_alpha) per part so we can
            # recompute stroke_half_px on future cache hits.
            new_parts_start = len(surface_parts)
            stroke_per_part_new: list[tuple[float, float]] = []

            for submob in surface_submobs:
                if id(submob) in _seen_submobs:
                    continue
                _seen_submobs.add(id(submob))
                stroke_rgba_sub = submob.get_stroke_rgbas()
                sw_sub = (
                    float(submob.get_stroke_width())
                    if stroke_rgba_sub.shape[0] > 0
                    else 0.0
                )
                s_alpha_sub = (
                    float(stroke_rgba_sub[0, 3])
                    if stroke_rgba_sub.shape[0] > 0
                    else 0.0
                )
                data = _collect_surface_geometry(
                    submob,
                    view_matrix,
                    proj_matrix,
                    diffuse_strength=float(
                        getattr(submob, "diffuse_strength", surf_diffuse)
                    ),
                    specular_strength=float(
                        getattr(submob, "specular_strength", surf_specular)
                    ),
                    specular_exponent=float(
                        getattr(submob, "specular_exponent", surf_spec_exp)
                    ),
                )
                if data is not None:
                    cls = _surface_opacity_class(data)
                    cmd = "surface_opaque" if cls == "opaque" else "surface_oit"
                    draw_plan.append((cmd, len(surface_parts)))
                    surface_parts.append(data)
                    stroke_per_part_new.append((sw_sub, s_alpha_sub))

            # Record this mob so we can cache its smoothed parts later.
            _new_surface_mobs.append(
                (
                    mob,
                    geom_hash,
                    color_hash,
                    new_parts_start,
                    len(surface_parts),
                    stroke_per_part_new,
                )
            )
            continue

        # ── Regular VMobject (2-D or shade_in_3d) ────────────────────────
        vmob_submobs = mob.family_members_with_points()
        if use_z_index:
            vmob_submobs = sorted(vmob_submobs, key=lambda m: m.z_index)
        for submob in vmob_submobs:
            if id(submob) in _seen_submobs:
                continue
            _seen_submobs.add(id(submob))
            phash = _points_hash(submob)
            cached = _fill_stroke_cache.get(submob)
            if cached is None or cached[0] != phash:
                result = _collect_cubics(submob)
                if result is not None:
                    _fill_stroke_cache[submob] = (phash, result)
                else:
                    _fill_stroke_cache.pop(submob, None)
                    continue
            cached = _fill_stroke_cache.get(submob)
            if cached is None:
                continue
            fill_cubics, stroke_cubics = cached[1]

            # Fixed-orientation pre-transform: translate control points so the
            # submob appears at its full 3D-projected center position while
            # preserving local orientation (no rotation of the local shape).
            #
            # Cairo's equivalent: transform_points_pre_display() computes
            #   new_center = project_point(center)   (full camera rotation)
            #   points     = points + (new_center - center)
            # i.e. translate all points by the difference between the
            # camera-space center and the world-space center.
            #
            # In WebGPU, with t_full == t_fixed == [0,0,-11], the offset
            # simplifies to:
            #   offset = R_full @ center_w - center_w  = (R_full - I) @ center_w
            # After adding this offset, the GPU shader applies fixed_view
            # (identity rotation + z-translation), giving:
            #   view_pos = (point_w + offset) + [0,0,-11]
            #            = (point_w - center_w) + (R_full @ center_w + [0,0,-11])
            # which is the local shape centred at the full-projection center. ✓
            if center_view_matrix is not None:
                R_full = center_view_matrix[:3, :3].astype(np.float32)
                c_w = submob.get_center().astype(np.float32)
                offset = R_full @ c_w - c_w  # shape (3,)
                fill_cubics = fill_cubics + offset  # broadcast (N,4,3)+(3,)
                stroke_cubics = stroke_cubics + offset

            # Fetch current colors every frame (they change during animations).
            fill_rgba = submob.get_fill_rgbas()
            stroke_rgba = submob.get_stroke_rgbas()
            fill_color = (
                fill_rgba[0].astype(np.float32)
                if fill_rgba.shape[0] > 0
                else np.zeros(4, dtype=np.float32)
            )
            stroke_color = (
                stroke_rgba[0].astype(np.float32)
                if stroke_rgba.shape[0] > 0
                else np.zeros(4, dtype=np.float32)
            )
            stroke_width = (
                float(submob.get_stroke_width()) if stroke_rgba.shape[0] > 0 else 0.0
            )

            # Skip entirely invisible objects (both fill and stroke transparent).
            if fill_color[3] < 0.001 and (
                stroke_color[3] < 0.001 or stroke_width < 0.001
            ):
                continue

            # 0 = nonzero (default), 1 = evenodd (set by SVG parser)
            fill_rule = int(getattr(submob, "fill_rule", 0))

            # Gradient fill: pass all colour stops and gradient axis endpoints.
            gradient_start = gradient_end = None
            if fill_rgba.shape[0] > 1:
                try:
                    gradient_start, gradient_end = (
                        submob.get_gradient_start_and_end_points()
                    )
                except Exception:
                    pass  # fall back to solid fill_color[0]

            # Gradient stroke: same pattern as fill gradient.
            stroke_gradient_start = stroke_gradient_end = None
            if stroke_rgba.shape[0] > 1:
                try:
                    stroke_gradient_start, stroke_gradient_end = (
                        submob.get_gradient_start_and_end_points()
                    )
                except Exception:
                    pass  # fall back to solid stroke_color[0]

            # Build bounding quad with placeholder curve indices.
            quad_verts = _build_fill_stroke_quad(
                fill_cubics=fill_cubics,
                stroke_cubics=stroke_cubics,
                fill_color=fill_color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                fill_curve_start=0,  # assigned below after all objects are collected
                stroke_curve_start=0,  # assigned below
                view_matrix=view_matrix,
                proj_matrix=proj_matrix,
                fill_rule=fill_rule,
                fill_rgbas=fill_rgba if fill_rgba.shape[0] > 1 else None,
                gradient_start=gradient_start,
                gradient_end=gradient_end,
                stroke_rgbas=stroke_rgba if stroke_rgba.shape[0] > 1 else None,
                stroke_gradient_start=stroke_gradient_start,
                stroke_gradient_end=stroke_gradient_end,
            )
            if len(quad_verts) == 0:
                continue

            is_3d = getattr(submob, "shade_in_3d", False)
            draw_plan.append(
                ("fill_stroke_3d" if is_3d else "fill_stroke_2d", len(fs_parts))
            )
            fs_parts.append(quad_verts)
            all_fill_cubics.append(fill_cubics)
            all_stroke_cubics.append(stroke_cubics)
            n_fill_cubics_per.append(len(fill_cubics))
            n_stroke_cubics_per.append(len(stroke_cubics))

    if not draw_plan:
        return None

    device: wgpu_t.GPUDevice = renderer.device

    # ── Assign global curve start indices ────────────────────────────────
    # Cubics buffer layout: [fill_cubics_obj0, fill_cubics_obj1, ...,
    #                        stroke_cubics_obj0, stroke_cubics_obj1, ...]
    # Quads output layout:  [fill_quads_obj0, fill_quads_obj1, ...,
    #                        stroke_quads_obj0, stroke_quads_obj1, ...]
    total_fill_cubics = sum(n_fill_cubics_per)
    total_stroke_cubics = sum(n_stroke_cubics_per)
    n_cubics_total = total_fill_cubics + total_stroke_cubics

    fill_global = 0  # running fill cubic index
    stroke_global = total_fill_cubics  # stroke cubics follow all fill cubics

    for i, part in enumerate(fs_parts):
        part["fill_curve_start"] = fill_global * 4
        part["n_fill_curves"] = n_fill_cubics_per[i] * 4
        part["stroke_curve_start"] = stroke_global * 4
        part["n_stroke_curves"] = n_stroke_cubics_per[i] * 4
        fill_global += n_fill_cubics_per[i]
        stroke_global += n_stroke_cubics_per[i]

    # ── Upload vertex data ───────────────────────────────────────────────
    fs_buf, fs_byte_offsets = None, []
    if fs_parts:
        fs_buf, fs_byte_offsets = _batch_upload(device, fs_parts)
        renderer.frame_vbos.append(fs_buf)

    # ── Upload cubics and create compute/render bind groups ──────────────
    cubics_buf = quads_out_buf = compute_bg = render_bg = None

    if n_cubics_total > 0:
        # Build flat float32 array: [all fill cubics..., all stroke cubics...]
        fill_arrays = [c for c in all_fill_cubics if len(c) > 0]
        stroke_arrays = [c for c in all_stroke_cubics if len(c) > 0]
        all_arrays = fill_arrays + stroke_arrays
        all_cubics = np.concatenate(all_arrays, axis=0)  # (N, 4, 3)
        cubics_flat = all_cubics.astype(np.float32).ravel()  # N*12 floats

        cubics_buf = device.create_buffer_with_data(
            data=cubics_flat.tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        renderer.frame_vbos.append(cubics_buf)

        quads_size = n_cubics_total * 36 * 4  # 4 quads × 9 floats × 4 bytes
        quads_out_buf = device.create_buffer(
            size=max(quads_size, 16),  # WebGPU minimum binding size
            usage=wgpu.BufferUsage.STORAGE,
        )
        renderer.frame_vbos.append(quads_out_buf)

        # Params uniform (n_cubics, padded to 16 bytes for WebGPU alignment).
        params_bytes = struct.pack("<4I", n_cubics_total, 0, 0, 0)
        params_buf = device.create_buffer_with_data(
            data=params_bytes,
            usage=wgpu.BufferUsage.UNIFORM,
        )
        renderer.frame_vbos.append(params_buf)

        compute_bg = device.create_bind_group(
            layout=renderer._compute_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": cubics_buf,
                        "offset": 0,
                        "size": cubics_buf.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": quads_out_buf,
                        "offset": 0,
                        "size": quads_out_buf.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {"buffer": params_buf, "offset": 0, "size": 16},
                },
            ],
        )

        render_bg = device.create_bind_group(
            layout=renderer._fill_stroke_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": camera_uniform_buf,
                        "offset": 0,
                        "size": camera_uniform_buf.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": quads_out_buf,
                        "offset": 0,
                        "size": quads_out_buf.size,
                    },
                },
            ],
        )

    # ── Upload surface data ──────────────────────────────────────────────
    # Apply normal smoothing only to parts from cache-miss mobs; cached
    # parts already carry correctly smoothed normals.
    surface_buf, surface_byte_offsets = None, []
    if surface_parts:
        if _new_surface_mobs:
            # Smooth normals for newly-tessellated slices.
            # _new_surface_mobs entries:
            #   (mob, geom_hash, color_hash, start, end, stroke_per_part_new)
            new_slices: list[np.ndarray] = []
            for mob, geom_hash, color_hash, start, end, _ in _new_surface_mobs:
                new_slices.extend(surface_parts[start:end])
            _smooth_surface_normals(new_slices)

            # Cache each newly-tessellated mob's smoothed parts.
            for (
                mob,
                geom_hash,
                color_hash,
                start,
                end,
                stroke_per_part_new,
            ) in _new_surface_mobs:
                parts_for_mob = surface_parts[start:end]
                if not parts_for_mob:
                    continue
                # Collect draw commands for this mob's global part indices.
                # Must filter on cmd type to exclude VMobject ("fill_stroke_*")
                # entries: draw_plan is shared between VMobject and Surface paths,
                # and both index from 0 (fs_parts vs surface_parts respectively),
                # so index-range-only filtering incorrectly includes VMobject entries
                # whose fs_parts index happens to fall inside [start, end).
                draw_cmds_for_mob = [
                    cmd
                    for cmd, idx in draw_plan
                    if start <= idx < end and cmd in ("surface_opaque", "surface_oit")
                ]
                # Cache as a SINGLE concatenated array so a hit can copy the
                # whole mob's geometry in one numpy operation.  stroke_half_px
                # is zeroed in the template; it is recomputed on every hit.
                # Colors in big_template reflect the current frame's colors so
                # that a color-only miss can patch them in O(N_parts).
                big_template = np.concatenate(parts_for_mob, axis=0)
                big_template["stroke_half_px"] = 0.0
                # Part boundary offsets as numpy arrays (avoid Python list ops on hit).
                sizes = np.array([len(p) for p in parts_for_mob], dtype=np.intp)
                starts = np.concatenate([[0], np.cumsum(sizes[:-1])]).astype(np.intp)
                ends = starts + sizes
                # Precompute stroke metadata as numpy arrays for vectorised hit path.
                sw_arr_c = np.array(
                    [sw for sw, _ in stroke_per_part_new], dtype=np.float32
                )
                sa_arr_c = np.array(
                    [alpha for _, alpha in stroke_per_part_new], dtype=np.float32
                )
                has_sw_c = (sw_arr_c > 0.0) & (sa_arr_c > 0.001)
                _surface_mob_cache[mob] = (
                    geom_hash,
                    color_hash,
                    big_template,
                    starts,
                    ends,
                    sw_arr_c,
                    sa_arr_c,
                    has_sw_c,
                    draw_cmds_for_mob,
                )

        surface_buf, surface_byte_offsets = _batch_upload(device, surface_parts)
        renderer.frame_vbos.append(surface_buf)

    oit_indices = [idx for cmd, idx in draw_plan if cmd == "surface_oit"]

    result = _FrameData(
        fs_parts=fs_parts,
        fs_buf=fs_buf,
        fs_byte_offsets=fs_byte_offsets,
        cubics_buf=cubics_buf,
        quads_out_buf=quads_out_buf,
        n_cubics_total=n_cubics_total,
        compute_bg=compute_bg,
        render_bg=render_bg,
        surface_parts=surface_parts,
        surface_buf=surface_buf,
        surface_byte_offsets=surface_byte_offsets,
        draw_plan=draw_plan,
        oit_indices=oit_indices,
    )

    # Store in cache so the NEXT frame can skip tessellation on a fingerprint hit.
    # frame_vbos are NOT added for cached buffers — the cache itself is the owner.
    # Remove the just-uploaded buffers from frame_vbos so they aren't released at
    # end-of-frame (the cache needs them to survive across frames).
    # _all_surface_call paths are excluded: the surface GPU buffer changes every
    # frame (stroke_half_px update), so the _FrameData cache cannot help there.
    if cache_slot is not None and not _all_surface_call:
        cached_bufs = {
            id(result.fs_buf),
            id(result.cubics_buf),
            id(result.quads_out_buf),
            id(result.surface_buf),
        } - {id(None)}
        renderer.frame_vbos = [
            b for b in renderer.frame_vbos if id(b) not in cached_bufs
        ]
        renderer._fd_cache[cache_slot] = (fp, result)

    return result


def draw_frame_data(
    renderer: WebGPURenderer,
    fd: _FrameData,
    cam_bg: wgpu_t.GPUBindGroup,
) -> None:
    """Record draw commands for *fd* into ``renderer.current_render_pass``.

    Draw order
    ----------
    1. 2-D fill+stroke objects — interleaved in ``draw_plan`` order (painter's
       algorithm; no depth write so objects paint over each other correctly).
    2. 3-D fill+stroke objects — depth write + test (shade_in_3d).
    3. Opaque parametric surfaces — depth write (includes barycentric wireframe).

    OIT surfaces are NOT drawn here; the caller reads ``fd.oit_indices`` and
    handles them in a separate accumulation pass.
    """
    rp = renderer.current_render_pass

    # ── 1. 2-D fill+stroke: painter's algorithm ───────────────────────────
    # All 2-D quads live in fs_buf in their original draw_plan order.
    # Instead of N separate set_vertex_buffer+draw calls we issue one draw
    # per *contiguous run* of "fill_stroke_2d" entries, dramatically reducing
    # the number of wgpu API calls (typically from 800 to 1 for a pure-2D scene).
    if fd.fs_buf is not None and fd.render_bg is not None:
        rp.set_pipeline(renderer.fill_stroke_pipeline)
        rp.set_bind_group(0, fd.render_bg, [], 0, 0)
        rp.set_vertex_buffer(0, fd.fs_buf)

        # Walk draw_plan and batch consecutive fill_stroke_2d entries.
        run_first_vertex: int = -1
        run_vertex_count: int = 0

        for cmd, idx in fd.draw_plan:
            if cmd != "fill_stroke_2d":
                # Flush the current 2D run (if any) before breaking the batch.
                if run_vertex_count > 0:
                    rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                    run_vertex_count = 0
                    run_first_vertex = -1
                continue

            arr = fd.fs_parts[idx]
            byte_offset = fd.fs_byte_offsets[idx]
            first_vert = byte_offset // _FILL_STROKE_STRIDE

            if run_first_vertex < 0:
                # Start a new run.
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
            elif first_vert == run_first_vertex + run_vertex_count:
                # Extend the current contiguous run.
                run_vertex_count += len(arr)
            else:
                # Gap in the buffer (shouldn't happen for pure-2D scenes but
                # guard anyway).  Flush old run, start new one.
                rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                run_first_vertex = first_vert
                run_vertex_count = len(arr)

        # Flush final run.
        if run_vertex_count > 0:
            rp.draw(run_vertex_count, 1, run_first_vertex, 0)

    # ── 2. 3-D fill+stroke: depth-tested and depth-written ────────────────
    # Same batching strategy for shade_in_3d VMobjects.
    if fd.fs_buf is not None and fd.render_bg is not None:
        rp.set_pipeline(renderer.fill_stroke_3d_pipeline)
        rp.set_bind_group(0, fd.render_bg, [], 0, 0)
        rp.set_vertex_buffer(0, fd.fs_buf)

        run_first_vertex = -1
        run_vertex_count = 0

        for cmd, idx in fd.draw_plan:
            if cmd != "fill_stroke_3d":
                if run_vertex_count > 0:
                    rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                    run_vertex_count = 0
                    run_first_vertex = -1
                continue

            arr = fd.fs_parts[idx]
            byte_offset = fd.fs_byte_offsets[idx]
            first_vert = byte_offset // _FILL_STROKE_STRIDE

            if run_first_vertex < 0:
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
            elif first_vert == run_first_vertex + run_vertex_count:
                run_vertex_count += len(arr)
            else:
                rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                run_first_vertex = first_vert
                run_vertex_count = len(arr)

        if run_vertex_count > 0:
            rp.draw(run_vertex_count, 1, run_first_vertex, 0)

    # ── 3. Opaque parametric surfaces ─────────────────────────────────────
    if fd.surface_buf is not None:
        rp.set_pipeline(renderer.surface_pipeline)
        rp.set_bind_group(0, cam_bg, [], 0, 0)
        rp.set_vertex_buffer(0, fd.surface_buf)

        run_first_vertex = -1
        run_vertex_count = 0

        for cmd, idx in fd.draw_plan:
            if cmd != "surface_opaque":
                if run_vertex_count > 0:
                    rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                    run_vertex_count = 0
                    run_first_vertex = -1
                continue

            arr = fd.surface_parts[idx]
            byte_offset = fd.surface_byte_offsets[idx]
            first_vert = byte_offset // _SURFACE_COMBINED_STRIDE

            if run_first_vertex < 0:
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
            elif first_vert == run_first_vertex + run_vertex_count:
                run_vertex_count += len(arr)
            else:
                rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                run_first_vertex = first_vert
                run_vertex_count = len(arr)

        if run_vertex_count > 0:
            rp.draw(run_vertex_count, 1, run_first_vertex, 0)


def draw_frame_data_subcam(
    rp: Any,
    fd: _FrameData,
    sub_fill_render_bg: Any,
    sub_cam_bg: Any,
    fill_2d_pipeline: Any,
    fill_3d_pipeline: Any,
    surf_pipeline: Any,
) -> None:
    """Draw *fd* into render pass *rp* using sub-camera pipelines and bind groups.

    Mirrors ``draw_frame_data`` but accepts explicit pipeline objects and bind
    groups instead of reading them from the renderer.  Used by
    ``_render_sub_camera_pass`` to reuse cached geometry (quads buffer, vertex
    buffer, surface buffer) with a different camera uniform.

    Parameters
    ----------
    rp
        Active render pass encoder targeting the sub-camera texture.
    fd
        Cached geometry from the main frame's ``collect_frame_data`` call.
    sub_fill_render_bg
        Bind group with sub-camera uniform (binding 0) + quads storage (binding 1).
        Replaces ``fd.render_bg`` for fill-stroke draw calls.
    sub_cam_bg
        Camera-only bind group with sub-camera uniform (binding 0).
        Used for surface draw calls.
    fill_2d_pipeline / fill_3d_pipeline / surf_pipeline
        Sub-camera render pipelines targeting the sub-camera texture format.
    """
    # ── 2-D fill+stroke ───────────────────────────────────────────────────
    if fd.fs_buf is not None and sub_fill_render_bg is not None:
        rp.set_pipeline(fill_2d_pipeline)
        rp.set_bind_group(0, sub_fill_render_bg, [], 0, 0)
        rp.set_vertex_buffer(0, fd.fs_buf)

        run_first_vertex: int = -1
        run_vertex_count: int = 0
        for cmd, idx in fd.draw_plan:
            if cmd != "fill_stroke_2d":
                if run_vertex_count > 0:
                    rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                    run_vertex_count = 0
                    run_first_vertex = -1
                continue
            arr = fd.fs_parts[idx]
            byte_offset = fd.fs_byte_offsets[idx]
            first_vert = byte_offset // _FILL_STROKE_STRIDE
            if run_first_vertex < 0:
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
            elif first_vert == run_first_vertex + run_vertex_count:
                run_vertex_count += len(arr)
            else:
                rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
        if run_vertex_count > 0:
            rp.draw(run_vertex_count, 1, run_first_vertex, 0)

    # ── 3-D fill+stroke ───────────────────────────────────────────────────
    if fd.fs_buf is not None and sub_fill_render_bg is not None:
        rp.set_pipeline(fill_3d_pipeline)
        rp.set_bind_group(0, sub_fill_render_bg, [], 0, 0)
        rp.set_vertex_buffer(0, fd.fs_buf)

        run_first_vertex = -1
        run_vertex_count = 0
        for cmd, idx in fd.draw_plan:
            if cmd != "fill_stroke_3d":
                if run_vertex_count > 0:
                    rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                    run_vertex_count = 0
                    run_first_vertex = -1
                continue
            arr = fd.fs_parts[idx]
            byte_offset = fd.fs_byte_offsets[idx]
            first_vert = byte_offset // _FILL_STROKE_STRIDE
            if run_first_vertex < 0:
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
            elif first_vert == run_first_vertex + run_vertex_count:
                run_vertex_count += len(arr)
            else:
                rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
        if run_vertex_count > 0:
            rp.draw(run_vertex_count, 1, run_first_vertex, 0)

    # ── Opaque surfaces ───────────────────────────────────────────────────
    if fd.surface_buf is not None and surf_pipeline is not None:
        rp.set_pipeline(surf_pipeline)
        rp.set_bind_group(0, sub_cam_bg, [], 0, 0)
        rp.set_vertex_buffer(0, fd.surface_buf)

        run_first_vertex = -1
        run_vertex_count = 0
        for cmd, idx in fd.draw_plan:
            if cmd != "surface_opaque":
                if run_vertex_count > 0:
                    rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                    run_vertex_count = 0
                    run_first_vertex = -1
                continue
            arr = fd.surface_parts[idx]
            byte_offset = fd.surface_byte_offsets[idx]
            first_vert = byte_offset // _SURFACE_COMBINED_STRIDE
            if run_first_vertex < 0:
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
            elif first_vert == run_first_vertex + run_vertex_count:
                run_vertex_count += len(arr)
            else:
                rp.draw(run_vertex_count, 1, run_first_vertex, 0)
                run_first_vertex = first_vert
                run_vertex_count = len(arr)
        if run_vertex_count > 0:
            rp.draw(run_vertex_count, 1, run_first_vertex, 0)


# ---------------------------------------------------------------------------
# GPU upload helpers
# ---------------------------------------------------------------------------


def _batch_upload(
    device: wgpu_t.GPUDevice,
    arrays: list[np.ndarray],
) -> tuple[wgpu_t.GPUBuffer, list[int]]:
    """Concatenate *arrays* into one bytes blob and upload as a VERTEX buffer."""
    import wgpu

    byte_offsets: list[int] = []
    parts: list[bytes] = []
    offset = 0
    for arr in arrays:
        byte_offsets.append(offset)
        b = arr.tobytes()
        parts.append(b)
        offset += len(b)

    buf = device.create_buffer_with_data(
        data=b"".join(parts),
        usage=wgpu.BufferUsage.VERTEX,
    )
    return buf, byte_offsets


# ---------------------------------------------------------------------------
# VMobject cubic collector — geometry cache
# ---------------------------------------------------------------------------


def _collect_cubics(
    vmobject: VMobject,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(fill_cubics, stroke_cubics)`` for the GPU compute shader.

    *fill_cubics*  — ``(N, 4, 3)`` float32.  All subpath cubics **plus** one
        linear closing cubic per open subpath (required for correct winding-
        number coverage during partial animations such as ``Create``).

    *stroke_cubics* — ``(M, 4, 3)`` float32.  Only the actual subpath cubics,
        no closing segment — the stroke should follow the visible curve only.

    Colors are NOT stored here; they are fetched fresh every frame in
    ``collect_frame_data`` so that opacity animations work correctly.

    Returns ``None`` if the vmobject has no usable bezier curves.
    """
    nppcc = vmobject.n_points_per_cubic_curve

    fill_cubics_list: list[np.ndarray] = []
    stroke_cubics_list: list[np.ndarray] = []

    for subpath in vmobject.get_subpaths():
        n_curves = len(subpath) // nppcc
        if n_curves == 0:
            continue
        pts = subpath[: n_curves * nppcc]
        b0s = pts[0::nppcc].astype(np.float32)
        h0s = pts[1::nppcc].astype(np.float32)
        h1s = pts[2::nppcc].astype(np.float32)
        b3s = pts[3::nppcc].astype(np.float32)

        cubics = np.stack([b0s, h0s, h1s, b3s], axis=1)  # (n, 4, 3)
        stroke_cubics_list.append(cubics)
        fill_cubics_list.append(cubics)

        # Closing segment: linear cubic from the last anchor back to the first.
        # Degree-elevation from a line (last→first) to a cubic:
        #   b0 = last, b1 = last + (first-last)/3,
        #   b2 = last + 2*(first-last)/3, b3 = first.
        first = b0s[0]
        last = b3s[-1]
        if not np.allclose(first, last, atol=1e-6):
            diff = first - last
            closing = np.array(
                [[last, last + diff * (1.0 / 3.0), last + diff * (2.0 / 3.0), first]],
                dtype=np.float32,
            )
            fill_cubics_list.append(closing)

    if not fill_cubics_list and not stroke_cubics_list:
        return None

    fill_cubics = (
        np.concatenate(fill_cubics_list, axis=0)
        if fill_cubics_list
        else np.empty((0, 4, 3), dtype=np.float32)
    )
    stroke_cubics = (
        np.concatenate(stroke_cubics_list, axis=0)
        if stroke_cubics_list
        else np.empty((0, 4, 3), dtype=np.float32)
    )
    return fill_cubics, stroke_cubics


# ---------------------------------------------------------------------------
# Bounding-quad builder
# ---------------------------------------------------------------------------


def _build_fill_stroke_quad(
    fill_cubics: np.ndarray,
    stroke_cubics: np.ndarray,
    fill_color: np.ndarray,
    stroke_color: np.ndarray,
    stroke_width: float,
    fill_curve_start: int,
    stroke_curve_start: int,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    fill_rule: int = 0,
    fill_rgbas: np.ndarray | None = None,
    gradient_start: np.ndarray | None = None,
    gradient_end: np.ndarray | None = None,
    stroke_rgbas: np.ndarray | None = None,
    stroke_gradient_start: np.ndarray | None = None,
    stroke_gradient_end: np.ndarray | None = None,
) -> np.ndarray:
    """Build a ``_FILL_STROKE_DTYPE`` bounding quad (6 vertices) for one object.

    The bounding box is computed in NDC space (clip.xy / clip.w) from the
    anchor points of both fill and stroke cubics, then mapped back to world
    space at the average view-space Z.  This is correct for both orthographic
    (w = 1) and perspective projections.

    *stroke_half_ndc* is the stroke half-width in NDC units, computed from
    the current projection matrix and average clip-w so that stroke width is
    consistent across perspective depths.

    *fill_rgbas* — if provided and has more than one row, enables gradient fill.
    *gradient_start* / *gradient_end* — world-space endpoints of the fill gradient axis.
    *stroke_rgbas* — if provided and has more than one row, enables gradient stroke.
    *stroke_gradient_start* / *stroke_gradient_end* — world-space endpoints of the stroke gradient axis.
    """
    # Gather all anchor points (b0 and b3 of every cubic).
    anchor_lists: list[np.ndarray] = []
    if len(fill_cubics) > 0:
        anchor_lists.append(fill_cubics[:, 0])
        anchor_lists.append(fill_cubics[:, 3])
    if len(stroke_cubics) > 0:
        anchor_lists.append(stroke_cubics[:, 0])
        anchor_lists.append(stroke_cubics[:, 3])

    if not anchor_lists:
        return np.empty(0, dtype=_FILL_STROKE_DTYPE)

    anchors = np.concatenate(anchor_lists, axis=0).astype(np.float32)  # (N, 3)

    vm = view_matrix.astype(np.float32)
    pm = proj_matrix.astype(np.float32)
    R, t = vm[:3, :3], vm[:3, 3]

    pts_v = (R @ anchors.T).T + t  # (N, 3) view space
    avg_z_v = float(pts_v[:, 2].mean())

    # Perspective divide → NDC.
    ones = np.ones((len(pts_v), 1), dtype=np.float32)
    clips = (pm @ np.hstack([pts_v, ones]).T).T  # (N, 4)
    w = clips[:, 3:4]
    w_s = np.where(np.abs(w) > 1e-8, w, np.sign(w + 1e-38) * 1e-8)
    ndcs = clips[:, :2] / w_s  # (N, 2) NDC

    PAD = 0.05
    ndc_min = ndcs.min(axis=0) - PAD
    ndc_max = ndcs.max(axis=0) + PAD

    # Stroke half-width in NDC.
    # v_thickness = 0.004 * stroke_width  (view-space, matching vmobject_stroke.wgsl)
    # stroke_half_ndc = v_thickness * pm[0,0] / avg_clip_w
    #   where avg_clip_w = pm[3,2]*avg_z + pm[3,3]
    avg_clip_w = float(pm[3, 2] * avg_z_v + pm[3, 3])
    avg_clip_w = avg_clip_w if abs(avg_clip_w) > 1e-8 else 1.0
    stroke_half_ndc = 0.0
    if stroke_width > 0.0 and float(stroke_color[3]) > 0.001:
        stroke_half_ndc = float(0.004 * stroke_width * abs(pm[0, 0]) / abs(avg_clip_w))
        # Add stroke padding so the bounding quad covers the stroke edges.
        ndc_min -= stroke_half_ndc * 2.0
        ndc_max += stroke_half_ndc * 2.0

    # Invert NDC bounding corners to view space.
    inv_px = 1.0 / (pm[0, 0] if abs(pm[0, 0]) > 1e-8 else 1.0)
    inv_py = 1.0 / (pm[1, 1] if abs(pm[1, 1]) > 1e-8 else 1.0)
    x0_v = (float(ndc_min[0]) * avg_clip_w - float(pm[0, 3])) * inv_px
    x1_v = (float(ndc_max[0]) * avg_clip_w - float(pm[0, 3])) * inv_px
    y0_v = (float(ndc_min[1]) * avg_clip_w - float(pm[1, 3])) * inv_py
    y1_v = (float(ndc_max[1]) * avg_clip_w - float(pm[1, 3])) * inv_py

    corners_v = np.array(
        [
            [x0_v, y0_v, avg_z_v],
            [x1_v, y0_v, avg_z_v],
            [x0_v, y1_v, avg_z_v],
            [x1_v, y1_v, avg_z_v],
        ],
        dtype=np.float32,
    )
    R_inv = R.T
    t_inv = -(R_inv @ t)
    corners_w = (R_inv @ corners_v.T).T + t_inv  # (4, 3) world space
    quad_pos = corners_w[[0, 1, 2, 1, 3, 2]]  # (6, 3) two CCW triangles

    # ── Per-vertex fill colours (gradient support) ────────────────────────────
    # If fill_rgbas has >1 colour row, interpolate along the gradient axis.
    # gradient_start / gradient_end are world-space endpoints of the axis.
    if (
        fill_rgbas is not None
        and fill_rgbas.shape[0] > 1
        and gradient_start is not None
        and gradient_end is not None
    ):
        gs = np.asarray(gradient_start, dtype=np.float32)
        ge = np.asarray(gradient_end, dtype=np.float32)
        axis = ge - gs
        axis_len2 = float(np.dot(axis, axis))
        if axis_len2 > 1e-12:
            # Project each of the 4 corners onto the gradient axis → t ∈ [0, 1].
            t_corners = np.clip(
                np.dot(corners_w - gs, axis) / axis_len2, 0.0, 1.0
            )  # (4,)
            n_stops = fill_rgbas.shape[0]
            # Interpolate: t_corners maps to colour stop indices.
            idx_f = t_corners * (n_stops - 1)  # float indices
            idx_lo = np.floor(idx_f).astype(int).clip(0, n_stops - 2)
            idx_hi = idx_lo + 1
            frac = (idx_f - idx_lo)[:, None]  # (4, 1)
            corner_colors = (
                fill_rgbas[idx_lo].astype(np.float32) * (1.0 - frac)
                + fill_rgbas[idx_hi].astype(np.float32) * frac
            )  # (4, 4)
            # Map corners [0,1,2,3] → quad vertices [0,1,2,1,3,2].
            per_vertex_fill = corner_colors[[0, 1, 2, 1, 3, 2]]  # (6, 4)
        else:
            per_vertex_fill = np.broadcast_to(fill_color, (6, 4)).copy()
    else:
        per_vertex_fill = np.broadcast_to(fill_color, (6, 4)).copy()

    # ── Per-vertex stroke colours (gradient support) ─────────────────────────
    if (
        stroke_rgbas is not None
        and stroke_rgbas.shape[0] > 1
        and stroke_gradient_start is not None
        and stroke_gradient_end is not None
    ):
        sgs = np.asarray(stroke_gradient_start, dtype=np.float32)
        sge = np.asarray(stroke_gradient_end, dtype=np.float32)
        s_axis = sge - sgs
        s_axis_len2 = float(np.dot(s_axis, s_axis))
        if s_axis_len2 > 1e-12:
            t_corners = np.clip(
                np.dot(corners_w - sgs, s_axis) / s_axis_len2, 0.0, 1.0
            )  # (4,)
            n_stops = stroke_rgbas.shape[0]
            idx_f = t_corners * (n_stops - 1)
            idx_lo = np.floor(idx_f).astype(int).clip(0, n_stops - 2)
            idx_hi = idx_lo + 1
            frac = (idx_f - idx_lo)[:, None]
            corner_colors = (
                stroke_rgbas[idx_lo].astype(np.float32) * (1.0 - frac)
                + stroke_rgbas[idx_hi].astype(np.float32) * frac
            )  # (4, 4)
            per_vertex_stroke = corner_colors[[0, 1, 2, 1, 3, 2]]  # (6, 4)
        else:
            per_vertex_stroke = np.broadcast_to(stroke_color, (6, 4)).copy()
    else:
        per_vertex_stroke = np.broadcast_to(stroke_color, (6, 4)).copy()

    n_fill_quads = len(fill_cubics) * 4  # 4 quadratics per cubic
    n_stroke_quads = len(stroke_cubics) * 4

    verts = np.empty(6, dtype=_FILL_STROKE_DTYPE)
    verts["in_pos"] = quad_pos
    verts["in_fill_color"] = per_vertex_fill
    verts["in_stroke_color"] = per_vertex_stroke
    verts["stroke_half_ndc"] = stroke_half_ndc
    verts["fill_curve_start"] = fill_curve_start
    verts["n_fill_curves"] = n_fill_quads
    verts["stroke_curve_start"] = stroke_curve_start
    verts["n_stroke_curves"] = n_stroke_quads
    verts["fill_rule"] = fill_rule
    return verts


# ---------------------------------------------------------------------------
# Surface geometry collectors (unchanged from original)
# ---------------------------------------------------------------------------


def _surface_opacity_class(part: np.ndarray) -> str:
    alphas = part["in_fill_color"][:, 3]
    return "opaque" if float(alphas.min()) >= 0.99 else "oit"


def _collect_surface_geometry(
    vmobject: VMobject,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    diffuse_strength: float = 0.8,
    specular_strength: float = 0.9,
    specular_exponent: float = 16.0,
) -> np.ndarray | None:
    """Return a ``_SURFACE_COMBINED_DTYPE`` array for a shade_in_3d VMobject.

    Material parameters are passed from the parent :class:`~.Surface` so
    that ``diffuse_strength``, ``specular_strength``, and
    ``specular_exponent`` set on the parent are applied to every submobject
    patch.

    Barycentric coordinates are assigned per triangle in the centroid fan:
      centroid     → bary = (1, 0, 0)   (bary.x = 0 on outer edge)
      anchor_i     → bary = (0, 1, 0)
      anchor_{i+1} → bary = (0, 0, 1)

    ``stroke_half_px`` is computed from the stroke width, projection matrix
    and average clip-w of the surface anchors so that wireframe line width
    is consistent across perspective depths.
    """
    from manim import config

    fill_rgba = vmobject.get_fill_rgbas()
    if fill_rgba.shape[0] == 0 or fill_rgba[0, 3] == 0:
        return None

    fill_color = fill_rgba[0].astype(np.float32)
    stroke_rgba = vmobject.get_stroke_rgbas()
    stroke_color = (
        stroke_rgba[0].astype(np.float32)
        if stroke_rgba.shape[0] > 0
        else np.zeros(4, dtype=np.float32)
    )
    stroke_width = (
        float(vmobject.get_stroke_width()) if stroke_rgba.shape[0] > 0 else 0.0
    )

    nppcc = vmobject.n_points_per_cubic_curve

    all_verts: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_bary: list[np.ndarray] = []

    for subpath in vmobject.get_subpaths():
        n_curves = len(subpath) // nppcc
        if n_curves < 2:
            continue
        anchors = subpath[0::nppcc]
        last = subpath[n_curves * nppcc - 1 : n_curves * nppcc]
        if len(last) and not np.allclose(anchors[-1], last[0], atol=1e-6):
            anchors = np.vstack([anchors, last])

        n_pts = len(anchors)
        if n_pts < 3:
            continue

        centroid = anchors.mean(axis=0)
        v0 = anchors[0] - centroid
        v1 = anchors[1] - centroid
        raw_normal = np.cross(v1, v0).astype(np.float64)
        norm_len = np.linalg.norm(raw_normal)
        normal = (
            (raw_normal / norm_len).astype(np.float32)
            if norm_len > 1e-9
            else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )

        # Triangle fan: (centroid, anchor_i, anchor_{i+1})
        fan_verts = np.empty((n_pts * 3, 3), dtype=np.float32)
        fan_verts[0::3] = centroid.astype(np.float32)
        fan_verts[1::3] = anchors.astype(np.float32)
        fan_verts[2::3] = np.roll(anchors, -1, axis=0).astype(np.float32)

        # Barycentric coords: centroid=(1,0,0), anchor_i=(0,1,0), next=(0,0,1)
        bary_block = np.zeros((n_pts * 3, 3), dtype=np.float32)
        bary_block[0::3] = [1.0, 0.0, 0.0]
        bary_block[1::3] = [0.0, 1.0, 0.0]
        bary_block[2::3] = [0.0, 0.0, 1.0]

        all_verts.append(fan_verts)
        all_normals.append(np.tile(normal, (n_pts * 3, 1)))
        all_bary.append(bary_block)

    if not all_verts:
        return None

    verts = np.concatenate(all_verts, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    bary = np.concatenate(all_bary, axis=0)
    n_total = len(verts)

    # Compute stroke_half_px: half the wireframe line width in screen pixels.
    # Formula matches _build_fill_stroke_quad: 0.004 * width * |pm[0,0]| / |avg_clip_w|
    # then multiplied by pixel_width/2 to convert NDC to pixels.
    stroke_half_px = 0.0
    if stroke_width > 0.0 and float(stroke_color[3]) > 0.001:
        pm = proj_matrix.astype(np.float32)
        vm = view_matrix.astype(np.float32)
        R, t = vm[:3, :3], vm[:3, 3]
        pts_v = (R @ verts.T).T + t  # (N, 3) view space
        avg_z_v = float(pts_v[:, 2].mean())
        avg_clip_w = float(pm[3, 2] * avg_z_v + pm[3, 3])
        avg_clip_w = avg_clip_w if abs(avg_clip_w) > 1e-8 else 1.0
        stroke_half_ndc = float(0.004 * stroke_width * abs(pm[0, 0]) / abs(avg_clip_w))
        stroke_half_px = stroke_half_ndc * config.pixel_width * 0.5

    attrs = np.empty(n_total, dtype=_SURFACE_COMBINED_DTYPE)
    attrs["in_vert"] = verts
    attrs["in_normal"] = normals
    attrs["in_fill_color"] = fill_color
    attrs["in_stroke_color"] = stroke_color
    attrs["in_bary"] = bary
    attrs["stroke_half_px"] = stroke_half_px
    attrs["diffuse_strength"] = float(diffuse_strength)
    attrs["specular_strength"] = float(specular_strength)
    attrs["specular_exponent"] = float(specular_exponent)
    return attrs


def _smooth_surface_normals(surface_parts: list[np.ndarray]) -> None:
    """Average normals at shared vertex positions (modifies in-place)."""
    if not surface_parts:
        return

    all_verts = np.concatenate([p["in_vert"] for p in surface_parts], axis=0)
    all_norms = np.concatenate([p["in_normal"] for p in surface_parts], axis=0)

    PREC = 1e-5
    quantized = np.round(all_verts.astype(np.float64) / PREC).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)

    n_unique = int(inverse.max()) + 1
    smooth = np.zeros((n_unique, 3), dtype=np.float64)
    np.add.at(smooth, inverse, all_norms.astype(np.float64))

    lengths = np.linalg.norm(smooth, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-9, 1.0, lengths)
    smooth = (smooth / lengths).astype(np.float32)

    idx = 0
    for part in surface_parts:
        n = len(part)
        part["in_normal"] = smooth[inverse[idx : idx + n]]
        idx += n
