"""Mobjects made of triangles: surfaces with lighting and textures.

The OpenGL renderer draws them as meshes, with depth testing and lighting per pixel;
the Cairo renderer as one polygon per face, sorted far to near and lit per face.
:class:`.Surface` and the 3D primitives (:class:`.Sphere`, :class:`.Torus`, ...) are
parametric surfaces of this kind.

A mesh has a :mod:`material <manim.appearance.material>`: :class:`.PbrMaterial`
(the default) lights it under a :class:`.ThreeDCamera`'s light,
:class:`.UnlitMaterial` shows its colors as they are.
"""

from __future__ import annotations

__all__ = ["MeshMobject", "ParametricSurface"]

import itertools as it
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from manim.appearance.material import Material, PbrMaterial
from manim.mobject.mobject import Mobject
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.color import (
    BLUE_D,
    BLUE_E,
    LIGHT_GREY,
    ManimColor,
    ParsableManimColor,
    color_to_rgba,
    interpolate_color,
)
from manim.utils.images import get_full_raster_image_path
from manim.utils.iterables import stretch_array_to_length
from manim.utils.versioned_fields import ArrayField, Field

if TYPE_CHECKING:
    from typing import Self

    import numpy.typing as npt

    from manim.mobject.graphing.coordinate_systems import CoordinateSystem
    from manim.typing import (
        FloatRGBA_Array,
        FloatRGBALike_Array,
        PixelArray,
        Point3D,
        Point3D_Array,
        UV_Array,
        UVLike_Array,
    )


class MeshMobject(Mobject):
    """A mobject made of triangles.

    Parameters
    ----------
    points
        The vertices, shape ``(n, 3)``.
    triangles
        Indices into ``points``, three per triangle, shape ``(m, 3)``.
    color, opacity
        One color for the whole mesh. :meth:`set_vertex_colors` sets one per vertex.
    material
        How the mesh is lit: :class:`.PbrMaterial` (by default ``PbrMaterial()``) or
        :class:`.UnlitMaterial`.
    depth_test
        Whether the mesh hides what is behind it, independently of drawing order.
    texture
        An image drawn onto the mesh, a path or an RGBA pixel array. Needs ``uvs``.
    uvs
        Texture coordinates of the vertices, shape ``(n, 2)``, from ``(0, 0)`` at the
        upper left of the image to ``(1, 1)`` at the lower right.
    stroke_color, stroke_width, stroke_opacity
        The outlines of the faces (see :attr:`faces`); no outlines by default.
    """

    #: Vertex indices, three per triangle (int32, as renderers upload them).

    # Meshes have the styling methods of VMobjects that groups call, so they can be
    # in a VGroup (spheres as the vertices of a graph, for example).
    _vmobject_like = True
    triangles: ArrayField[np.ndarray] = ArrayField(np.int32)
    #: Polygons, as vertex indices in order around each, that the mesh is made of
    #: (quads for parametric surfaces). Empty means the triangles.
    faces: ArrayField[np.ndarray] = ArrayField(np.int32)
    #: Colours: one row for the whole mesh, or one per vertex.
    rgbas: ArrayField[FloatRGBA_Array] = ArrayField(np.float64)
    uvs: ArrayField[UV_Array] = ArrayField(np.float64)
    #: RGBA pixels, one byte per channel; empty without a texture.
    texture: ArrayField[PixelArray] = ArrayField(np.uint8)
    material: Field[Material] = Field()
    depth_test: Field[bool] = Field()
    #: Outline colors of the faces: one for all, or one per face.
    stroke_rgbas: ArrayField[np.ndarray] = ArrayField(np.float64)
    stroke_width: Field[float] = Field()

    def __init__(
        self,
        points: Point3D_Array | None = None,
        triangles: np.ndarray | None = None,
        color: ParsableManimColor = BLUE_D,
        opacity: float = 1.0,
        material: Material | None = None,
        depth_test: bool = True,
        texture: str | Path | PixelArray | None = None,
        uvs: UVLike_Array | None = None,
        stroke_color: ParsableManimColor = LIGHT_GREY,
        stroke_width: float = 0,
        stroke_opacity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.triangles = np.zeros((0, 3)) if triangles is None else triangles
        self.faces = np.zeros((0, 3))
        self.stroke_rgbas = [color_to_rgba(stroke_color, stroke_opacity)]
        self.stroke_width = stroke_width
        self.rgbas = [color_to_rgba(color, opacity)]
        self.uvs = np.zeros((0, 2)) if uvs is None else uvs
        self.texture = _load_texture(texture)
        self.material = PbrMaterial() if material is None else material
        self.depth_test = depth_test
        super().__init__(color=color, **kwargs)
        if points is not None:
            self.points = points

    # Colors

    def set_color(
        self,
        color: ParsableManimColor = BLUE_D,
        alpha: float | None = None,
        family: bool = True,
    ) -> Self:
        """Give the mesh one colour (and, with ``family``, its submobjects, such
        as the caps of a :class:`.Cylinder`); ``alpha`` sets its opacity too.
        """
        opacity = self.rgbas[:, 3].max() if alpha is None else alpha
        self.rgbas = [color_to_rgba(color, opacity)]
        self.color = ManimColor(color)
        if family:
            for submob in self.submobjects:
                if isinstance(submob, MeshMobject):
                    submob.set_color(color, alpha)
                else:
                    submob.set_color(color, family=True)
        return self

    def set_fill(
        self,
        color: ParsableManimColor | None = None,
        opacity: float | None = None,
        family: bool = True,
    ) -> Self:
        """Set one colour, or one opacity, for the whole mesh (and, with
        ``family``, its submobjects).
        """
        if color is not None:
            self.set_color(color, opacity, family=family)
        elif opacity is not None:
            self.set_opacity(opacity, family=family)
        return self

    def set_style(
        self,
        fill_color: ParsableManimColor | None = None,
        fill_opacity: float | None = None,
        stroke_color: ParsableManimColor | None = None,
        stroke_width: float | None = None,
        stroke_opacity: float | None = None,
        family: bool = True,
        **vmobject_style: Any,
    ) -> Self:
        """Set the colour of the faces and of their outlines.

        The other arguments of :meth:`.VMobject.set_style` (background strokes,
        sheen, background images) have no effect on meshes.
        """
        self.set_fill(fill_color, fill_opacity, family=family)
        return self.set_stroke(
            stroke_color, stroke_width, stroke_opacity, family=family
        )

    @property
    def fill_color(self) -> ManimColor:
        """The colour of the first vertex."""
        return ManimColor(self.rgbas[0, :3])

    @property
    def fill_opacity(self) -> float:
        """The highest opacity of the vertices."""
        return float(self.rgbas[:, 3].max())

    def get_fill_color(self) -> ManimColor:
        return self.fill_color

    def get_fill_opacity(self) -> float:
        return self.fill_opacity

    @property
    def stroke_color(self) -> ManimColor:
        """The colour of the first face's outline."""
        return ManimColor(self.stroke_rgbas[0, :3])

    @property
    def stroke_opacity(self) -> float:
        return float(self.stroke_rgbas[:, 3].max())

    def get_stroke_color(self) -> ManimColor:
        return self.stroke_color

    def get_stroke_width(self) -> float:
        return self.stroke_width

    def set_opacity(self, opacity: float, family: bool = True) -> Self:
        rgbas = np.array(self.rgbas)
        rgbas[:, 3] = opacity
        self.rgbas = rgbas
        if family:
            for submob in self.submobjects:
                submob.set_opacity(opacity, family=True)
        return self

    def fade(self, darkness: float = 0.5, family: bool = True) -> Self:
        rgbas = np.array(self.rgbas)
        rgbas[:, 3] *= 1 - darkness
        self.rgbas = rgbas
        return super().fade(darkness, family)

    def set_stroke(
        self,
        color: ParsableManimColor | None = None,
        width: float | None = None,
        opacity: float | None = None,
        background: bool = False,
        family: bool = True,
    ) -> Self:
        """Set the outlines of the faces (and, with ``family``, those of its
        submobjects). Meshes have no background strokes.
        """
        if family:
            for submob in self.submobjects:
                if hasattr(submob, "set_stroke"):
                    submob.set_stroke(
                        color, width, opacity, background=background, family=True
                    )
        if background:
            return self
        if color is not None or opacity is not None:
            rgbas = np.array(self.stroke_rgbas)
            if color is not None:
                rgbas = np.repeat([color_to_rgba(color)], len(rgbas), axis=0)
                rgbas[:, 3] = self.stroke_rgbas[:, 3]
            if opacity is not None:
                rgbas[:, 3] = opacity
            self.stroke_rgbas = rgbas
        if width is not None:
            self.stroke_width = width
        return self

    # What VMobject methods ask of the members of a group.

    def set_sheen(self, *args: Any, **kwargs: Any) -> Self:
        """Meshes have no sheen; for groups of vectorized mobjects and meshes."""
        return self

    def color_using_background_image(self, *args: Any, **kwargs: Any) -> Self:
        """Meshes have no background image; for groups of vectorized mobjects and
        meshes.
        """
        return self

    def get_faces(self) -> np.ndarray:
        """The polygons of the mesh: :attr:`faces`, or else the triangles."""
        return self.faces if len(self.faces) else self.triangles

    def set_vertex_colors(self, rgbas: FloatRGBALike_Array) -> Self:
        """Give every vertex its own color, as an ``(n, 4)`` array of RGBA values."""
        self.rgbas = rgbas
        return self

    def set_color_by_xyz_func(
        self,
        func: Callable[[Point3D_Array], npt.NDArray[np.float64]],
        colorscale: Sequence[ParsableManimColor],
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> Self:
        """Color the vertices by a function of their position.

        Parameters
        ----------
        func
            Maps the vertices, shape ``(n, 3)``, to one value each.
        colorscale
            Colors evenly spread from ``min_value`` to ``max_value``.
        """
        values = np.asarray(func(self.points), dtype=float)
        return self.set_vertex_colors(
            _colorscale(values, colorscale, min_value, max_value, self.rgbas[0, 3])
        )

    # Animation

    def align_points(self, mobject: Mobject) -> Self:
        if not isinstance(mobject, MeshMobject) or (
            self.points.shape != mobject.points.shape
            or not np.array_equal(self.triangles, mobject.triangles)
        ):
            raise ValueError(
                "Meshes can only be transformed into meshes with the same triangles "
                "(for parametric surfaces: the same resolution)."
            )
        return self

    def interpolate_color(
        self, mobject1: Mobject, mobject2: Mobject, alpha: float
    ) -> Self:
        assert isinstance(mobject1, MeshMobject)
        assert isinstance(mobject2, MeshMobject)
        self.rgbas = _mixed_rgbas(mobject1.rgbas, mobject2.rgbas, alpha)
        self.stroke_rgbas = _mixed_rgbas(
            mobject1.stroke_rgbas, mobject2.stroke_rgbas, alpha
        )
        start_width, end_width = mobject1.stroke_width, mobject2.stroke_width
        # Equal widths stay the value they are: mixing them can round it.
        self.stroke_width = (
            end_width
            if start_width == end_width
            else float((1 - alpha) * start_width + alpha * end_width)
        )
        self.material = _interpolate_material(
            mobject1.material, mobject2.material, alpha
        )
        return self

    def set_material(self, material: Material, family: bool = True) -> Self:
        """Set how the mesh (and, with ``family``, its submeshes) is lit."""
        for mob in self.get_family() if family else [self]:
            if isinstance(mob, MeshMobject):
                mob.material = material
        return self

    def pointwise_become_partial(self, mobject: Mobject, a: float, b: float) -> Self:
        """Show the triangles of ``mobject`` from proportion ``a`` to ``b``, in order."""
        assert isinstance(mobject, MeshMobject)
        count = len(mobject.triangles)
        low, high = int(a * count), int(np.ceil(b * count))
        self.points = mobject.points
        self.triangles = mobject.triangles[low:high]
        faces = mobject.faces
        if len(faces):
            # Faces made of triangles in order, the same number each.
            per_face = count // len(faces)
            shown = slice(low // per_face, -(-high // per_face))
            self.faces = faces[shown]
            if len(mobject.stroke_rgbas) == len(faces) > 1:
                self.stroke_rgbas = mobject.stroke_rgbas[shown]
        return self

    # Shape

    def get_wireframe(
        self, color: ParsableManimColor = LIGHT_GREY, stroke_width: float = 1
    ) -> VGroup:
        """The edges of the triangles, as lines drawn by both renderers."""
        edges = np.sort(self.triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        lines = VGroup()
        for start, end in np.unique(edges, axis=0):
            lines.add(VMobject().set_points_as_corners(self.points[[start, end]]))
        return lines.set_stroke(color, stroke_width)


class ParametricSurface(MeshMobject):
    """A surface given by a function of two parameters, drawn as a smooth mesh.

    Parameters
    ----------
    func
        Maps parameters ``(u, v)`` to a point in space.
    u_range, v_range
        The ranges of the parameters.
    resolution
        Number of steps in ``u`` and ``v``.
    flat
        Give every quad its own vertices, so that faces can have their own color
        (needed for checkerboards) and are shaded flat.
    texture
        An image drawn onto the surface: ``u`` runs left to right across it, ``v``
        bottom to top.

    Examples
    --------
    .. code-block:: python

        sphere = ParametricSurface(
            lambda u, v: [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)],
            u_range=[0, TAU],
            v_range=[0, PI],
        )
    """

    def __init__(
        self,
        func: Callable[[float, float], Point3D],
        u_range: Sequence[float] = (0, 1),
        v_range: Sequence[float] = (0, 1),
        resolution: int | Sequence[int] = 32,
        flat: bool = False,
        **kwargs: Any,
    ) -> None:
        self.func = func
        self.u_range: tuple[float, float] = _pair(u_range, "u_range")
        self.v_range: tuple[float, float] = _pair(v_range, "v_range")
        self.resolution: tuple[int, int] = (
            (resolution, resolution)
            if isinstance(resolution, int)
            else _pair(resolution, "resolution")
        )
        self.flat = flat
        super().__init__(**kwargs)

    def generate_points(self) -> Self:
        nu, nv = self.resolution
        us = np.linspace(*self.u_range, nu + 1)
        vs = np.linspace(*self.v_range, nv + 1)
        grid = np.array([[self.func(u, v) for v in vs] for u in us], dtype=float)
        return self._build(grid)

    def _build(self, grid: Point3D_Array) -> Self:
        """Make the mesh through ``grid``, the points at the corners of the parameter
        grid (shape ``(nu + 1, nv + 1, 3)``): smooth, or with :attr:`flat` quads.
        """
        nu, nv = self.resolution
        us = np.linspace(*self.u_range, nu + 1)
        vs = np.linspace(*self.v_range, nv + 1)
        uv = np.stack(np.meshgrid(us, vs, indexing="ij"), axis=-1)
        if self.flat:
            # Four vertices per quad: (u, v), (u + 1, v), (u, v + 1), (u + 1, v + 1).
            quads = [(i, j) for i in range(nu) for j in range(nv)]
            corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
            points = np.array(
                [grid[i + di, j + dj] for i, j in quads for di, dj in corners]
            )
            params = np.array(
                [uv[i + di, j + dj] for i, j in quads for di, dj in corners]
            )
            base = 4 * np.arange(len(quads))[:, None]
            triangles = np.concatenate(
                (base + [0, 1, 2], base + [1, 3, 2]), axis=1
            ).reshape(-1, 3)
            faces = base + [0, 1, 3, 2]
        else:
            points = grid.reshape(-1, 3)
            params = uv.reshape(-1, 2)
            index = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
            a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
            c, d = index[:-1, 1:].ravel(), index[1:, 1:].ravel()
            # Two triangles per quad, in order, so that a quad's triangles stay
            # together when part of the surface is shown.
            triangles = np.stack((a, b, c, b, d, c), axis=1).reshape(-1, 3)
            faces = np.stack((a, b, d, c), axis=1)
        self.points = points
        self.triangles = triangles
        self.faces = faces
        if len(self.texture):
            u0, u1 = self.u_range
            v0, v1 = self.v_range
            self.uvs = np.stack(
                ((params[:, 0] - u0) / (u1 - u0), 1 - (params[:, 1] - v0) / (v1 - v0)),
                axis=1,
            )
        return self

    def set_fill_by_checkerboard(
        self, *colors: ParsableManimColor, opacity: float | None = None
    ) -> Self:
        """Color the quads alternately with ``colors`` (default: two blues).

        A smooth surface gets :attr:`flat` quads through its current points.
        """
        if not self.flat:
            # From the surface as it is now (it may have been moved or deformed),
            # not from ``func``.
            grid = self._grid()
            self.flat = True
            self._build(grid)
        colors = colors or (BLUE_D, BLUE_E)
        alpha = self.rgbas[:, 3].max() if opacity is None else opacity
        nu, nv = self.resolution
        pattern = [
            color_to_rgba(colors[(i + j) % len(colors)], alpha)
            for i, j in it.product(range(nu), range(nv))
        ]
        return self.set_vertex_colors(np.repeat(pattern, 4, axis=0))

    def set_fill_by_value(
        self,
        axes: CoordinateSystem,
        colorscale: Sequence[ParsableManimColor],
        axis: int = 2,
    ) -> Self:
        """Color the surface by one coordinate of its points, measured on ``axes``.

        Parameters
        ----------
        axes
            The axes the surface is plotted on.
        colorscale
            Colors evenly spread over the range of that coordinate on the axes.
        axis
            The coordinate: 0, 1 or 2 for x, y or z.
        """
        values = np.asarray(axes.point_to_coords(self.points))[:, axis]
        low, high = getattr(axes, ("x_range", "y_range", "z_range")[axis])[:2]
        return self.set_vertex_colors(
            _colorscale(values, colorscale, low, high, self.rgbas[0, 3])
        )

    def pointwise_become_partial(self, mobject: Mobject, a: float, b: float) -> Self:
        """Show the part of ``mobject`` from proportion ``a`` to ``b`` of ``v``.

        Points outside are moved onto the edge of the shown part, so the triangles stay
        the same and the partial surface can be animated like the whole one.
        """
        assert isinstance(mobject, ParametricSurface)
        if mobject.flat:
            return super().pointwise_become_partial(mobject, a, b)
        nu, nv = mobject.resolution
        grid = mobject.points.reshape(nu + 1, nv + 1, 3).copy()
        low, high = a * nv, b * nv
        for bound, side in ((low, "low"), (high, "high")):
            index = min(int(bound), nv - 1)
            residue = bound - index
            edge = (1 - residue) * grid[:, index] + residue * grid[:, index + 1]
            if side == "low":
                grid[:, : index + 1] = edge[:, None]
            else:
                grid[:, index + 1 :] = edge[:, None]
        self.points = grid.reshape(-1, 3)
        self.triangles = mobject.triangles
        self.faces = mobject.faces
        return self

    def get_wireframe(
        self, color: ParsableManimColor = LIGHT_GREY, stroke_width: float = 1
    ) -> VGroup:
        """The parameter lines of constant ``u`` and ``v``, through the surface's
        current points.
        """
        grid = self._grid()
        lines = VGroup()
        for line in (*grid, *grid.transpose(1, 0, 2)):
            lines.add(VMobject().set_points_as_corners(line))
        return lines.set_stroke(color, stroke_width)

    def _grid(self) -> npt.NDArray[np.float64]:
        """The surface's current points at the corners of its parameter grid, shape
        ``(nu + 1, nv + 1, 3)``.

        A :attr:`flat` surface has every inner grid point four times, once per
        quad around it: the point is taken from the quad it is the first corner
        of (from the last quads along the far edges).
        """
        nu, nv = self.resolution
        points = self.points
        if not self.flat and len(points) == (nu + 1) * (nv + 1):
            return points.reshape(nu + 1, nv + 1, 3)
        if self.flat and len(points) == 4 * nu * nv:
            quads = points.reshape(nu, nv, 4, 3)
            grid = np.empty((nu + 1, nv + 1, 3))
            grid[:nu, :nv] = quads[:, :, 0]
            grid[nu, :nv] = quads[-1, :, 1]
            grid[:nu, nv] = quads[:, -1, 2]
            grid[nu, nv] = quads[-1, -1, 3]
            return grid
        raise ValueError(
            f"The {len(points)} points of the surface are not those of a "
            f"{'flat' if self.flat else 'smooth'} surface of resolution "
            f"{self.resolution}."
        )


def _interpolate_material(start: Material, end: Material, alpha: float) -> Material:
    """The material at progress ``alpha`` from ``start`` to ``end``.

    Two :class:`.PbrMaterial` materials interpolate their parameters; at ``alpha`` 1 the
    result is ``end`` itself. Between other materials, the material switches to
    ``end`` when ``alpha`` reaches 1.
    """
    if alpha >= 1 or start == end:
        return end
    if isinstance(start, PbrMaterial) and isinstance(end, PbrMaterial):
        if alpha <= 0:
            return start
        return PbrMaterial(
            roughness=(1 - alpha) * start.roughness + alpha * end.roughness,
            metallic=(1 - alpha) * start.metallic + alpha * end.metallic,
            emissive=(
                (1 - alpha) * start.emissive[0] + alpha * end.emissive[0],
                (1 - alpha) * start.emissive[1] + alpha * end.emissive[1],
                (1 - alpha) * start.emissive[2] + alpha * end.emissive[2],
            ),
        )
    return start


def _pair(values: Sequence[Any], name: str) -> tuple[Any, Any]:
    """``values`` as a pair; raises if it does not have two entries."""
    if len(values) != 2:
        raise ValueError(f"{name} needs two values, not {len(values)}: {values!r}")
    return values[0], values[1]


def _mixed_rgbas(
    start: FloatRGBA_Array, end: FloatRGBA_Array, alpha: float
) -> FloatRGBA_Array:
    """The colors at progress ``alpha`` from ``start`` to ``end``; the value itself
    where both are the same array, so assigning it keeps a field's version.
    """
    if start is end:
        return end
    if len(start) != len(end):
        length = max(len(start), len(end))
        start = stretch_array_to_length(start, length)
        end = stretch_array_to_length(end, length)
    return (1 - alpha) * start + alpha * end


def _load_texture(texture: str | Path | PixelArray | None) -> PixelArray:
    if texture is None:
        return np.zeros((0, 0, 4), dtype=np.uint8)
    if isinstance(texture, (str, Path)):
        path = get_full_raster_image_path(texture)
        return np.asarray(Image.open(path).convert("RGBA"))
    pixels = np.asarray(texture)
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        alpha = np.full(pixels.shape[:2] + (1,), 255, dtype=pixels.dtype)
        pixels = np.concatenate((pixels, alpha), axis=2)
    return pixels


def _colorscale(
    values: np.ndarray,
    colorscale: Iterable[ParsableManimColor],
    low: float,
    high: float,
    alpha: float,
) -> FloatRGBA_Array:
    """RGBA values for ``values``, from colors evenly spread from ``low`` to ``high``."""
    colors = [ManimColor(c) for c in colorscale]
    t = np.clip((values - low) / (high - low), 0, 1) * (len(colors) - 1)
    index = np.minimum(t.astype(int), len(colors) - 2)
    rgbas = np.array(
        [
            interpolate_color(colors[i], colors[i + 1], f).to_rgba()
            for i, f in zip(index, t - index, strict=True)
        ]
    )
    rgbas[:, 3] = alpha
    return rgbas
