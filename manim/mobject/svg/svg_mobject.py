"""Mobjects generated from an SVG file."""

from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import svgelements as se

from manim import config, logger

from ...constants import RIGHT
from ...utils.bezier import get_quadratic_approximation_of_cubic
from ...utils.images import get_full_vector_image_path
from ...utils.iterables import hash_obj
from ..geometry.arc import Circle
from ..geometry.line import Line
from ..geometry.polygram import Polygon, Rectangle, RoundedRectangle
from ..opengl.opengl_compatibility import ConvertToOpenGL
from ..types.vectorized_mobject import VMobject

__all__ = ["SVGMobject", "VMobjectFromSVGPath"]


SVG_HASH_TO_MOB_MAP: dict[int, VMobject] = {}


def _convert_point_to_3d(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 0.0])


class SVGMobject(VMobject, metaclass=ConvertToOpenGL):
    """A vectorized mobject created from importing an SVG file.

    Parameters
    ----------
    file_name
        The path to the SVG file.
    should_center
        Whether or not the mobject should be centered after
        being imported.
    height
        The target height of the mobject, set to 2 Manim units by default.
        If the height and width are both set to ``None``, the mobject
        is imported without being scaled.
    width
        The target width of the mobject, set to ``None`` by default. If
        the height and the width are both set to ``None``, the mobject
        is imported without being scaled.
    color
        The color (both fill and stroke color) of the mobject. If
        ``None`` (the default), the colors set in the SVG file
        are used.
    opacity
        The opacity (both fill and stroke opacity) of the mobject.
        If ``None`` (the default), the opacity set in the SVG file
        is used.
    fill_color
        The fill color of the mobject. If ``None`` (the default),
        the fill colors set in the SVG file are used.
    fill_opacity
        The fill opacity of the mobject. If ``None`` (the default),
        the fill opacities set in the SVG file are used.
    stroke_color
        The stroke color of the mobject. If ``None`` (the default),
        the stroke colors set in the SVG file are used.
    stroke_opacity
        The stroke opacity of the mobject. If ``None`` (the default),
        the stroke opacities set in the SVG file are used.
    stroke_width
        The stroke width of the mobject. If ``None`` (the default),
        the stroke width values set in the SVG file are used.
    svg_default
        A dictionary in which fallback values for unspecified
        properties of elements in the SVG file are defined. If
        ``None`` (the default), ``color``, ``opacity``, ``fill_color``
        ``fill_opacity``, ``stroke_color``, and ``stroke_opacity``
        are set to ``None``, and ``stroke_width`` is set to 0.
    path_string_config
        A dictionary with keyword arguments passed to
        :class:`.VMobjectFromSVGPath` used for importing path elements.
        If ``None`` (the default), no additional arguments are passed.
    use_svg_cache
        If True (default), the svg inputs (e.g. file_name, settings)
        will be used as a key and a copy of the created mobject will
        be saved using that key to be quickly retrieved if the same
        inputs need be processed later. For large SVGs which are used
        only once, this can be omitted to improve performance.
    kwargs
        Further arguments passed to the parent class.
    """

    def __init__(
        self,
        file_name: str | os.PathLike | None = None,
        should_center: bool = True,
        height: float | None = 2,
        width: float | None = None,
        color: str | None = None,
        opacity: float | None = None,
        fill_color: str | None = None,
        fill_opacity: float | None = None,
        stroke_color: str | None = None,
        stroke_opacity: float | None = None,
        stroke_width: float | None = None,
        svg_default: dict | None = None,
        path_string_config: dict | None = None,
        use_svg_cache: bool = True,
        **kwargs,
    ):
        super().__init__(color=None, stroke_color=None, fill_color=None, **kwargs)

        # process keyword arguments
        self.file_name = Path(file_name) if file_name is not None else None

        self.should_center = should_center
        self.svg_height = height
        self.svg_width = width
        self.color = color
        self.opacity = opacity
        self.fill_color = fill_color
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_opacity = stroke_opacity
        self.stroke_width = stroke_width

        if svg_default is None:
            svg_default = {
                "color": None,
                "opacity": None,
                "fill_color": None,
                "fill_opacity": None,
                "stroke_width": 0,
                "stroke_color": None,
                "stroke_opacity": None,
            }
        self.svg_default = svg_default

        if path_string_config is None:
            path_string_config = {}
        self.path_string_config = path_string_config

        self.init_svg_mobject(use_svg_cache=use_svg_cache)

        self.set_style(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_color=stroke_color,
            stroke_opacity=stroke_opacity,
            stroke_width=stroke_width,
        )
        self.move_into_position()

    def init_svg_mobject(self, use_svg_cache: bool) -> None:
        """Checks whether the SVG has already been imported and
        generates it if not.

        See also
        --------
        :meth:`.SVGMobject.generate_mobject`
        """
        if use_svg_cache:
            hash_val = hash_obj(self.hash_seed)
            if hash_val in SVG_HASH_TO_MOB_MAP:
                mob = SVG_HASH_TO_MOB_MAP[hash_val].copy()
                self.add(*mob)
                return

        self.generate_mobject()
        if use_svg_cache:
            SVG_HASH_TO_MOB_MAP[hash_val] = self.copy()

    @property
    def hash_seed(self) -> tuple:
        """A unique hash representing the result of the generated
        mobject points.

        Used as keys in the ``SVG_HASH_TO_MOB_MAP`` caching dictionary.
        """
        return (
            self.__class__.__name__,
            self.svg_default,
            self.path_string_config,
            self.file_name,
            config.renderer,
        )

    def generate_mobject(self) -> None:
        """Parse the SVG and translate its elements to submobjects."""
        file_path = self.get_file_path()
        element_tree = ET.parse(file_path)
        new_tree = self.modify_xml_tree(element_tree)
        # Create a temporary svg file to dump modified svg to be parsed
        modified_file_path = file_path.with_name(f"{file_path.stem}_{file_path.suffix}")
        new_tree.write(modified_file_path)

        svg = se.SVG.parse(modified_file_path)
        modified_file_path.unlink()

        mobjects = self.get_mobjects_from(svg)
        self.add(*mobjects)
        self.flip(RIGHT)  # Flip y

    def get_file_path(self) -> Path:
        """Search for an existing file based on the specified file name."""
        if self.file_name is None:
            raise ValueError("Must specify file for SVGMobject")
        return get_full_vector_image_path(self.file_name)

    def modify_xml_tree(self, element_tree: ET.ElementTree) -> ET.ElementTree:
        """Modifies the SVG element tree to include default
        style information.

        Parameters
        ----------
        element_tree
            The parsed element tree from the SVG file.
        """
        config_style_dict = self.generate_config_style_dict()
        style_keys = (
            "fill",
            "fill-opacity",
            "stroke",
            "stroke-opacity",
            "stroke-width",
            "style",
        )
        root = element_tree.getroot()
        root_style_dict = {k: v for k, v in root.attrib.items() if k in style_keys}

        new_root = ET.Element("svg", {})
        config_style_node = ET.SubElement(new_root, "g", config_style_dict)
        root_style_node = ET.SubElement(config_style_node, "g", root_style_dict)
        root_style_node.extend(root)
        return ET.ElementTree(new_root)

    def generate_config_style_dict(self) -> dict[str, str]:
        """Generate a dictionary holding the default style information."""
        keys_converting_dict = {
            "fill": ("color", "fill_color"),
            "fill-opacity": ("opacity", "fill_opacity"),
            "stroke": ("color", "stroke_color"),
            "stroke-opacity": ("opacity", "stroke_opacity"),
            "stroke-width": ("stroke_width",),
        }
        svg_default_dict = self.svg_default
        result = {}
        for svg_key, style_keys in keys_converting_dict.items():
            for style_key in style_keys:
                if svg_default_dict[style_key] is None:
                    continue
                result[svg_key] = str(svg_default_dict[style_key])
        return result

    def get_mobjects_from(self, svg: se.SVG) -> list[VMobject]:
        """Convert the elements of the SVG to a list of mobjects.

        Parameters
        ----------
        svg
            The parsed SVG file.
        """
        result = []
        for shape in svg.elements():
            # can we combine the two continue cases into one?
            if isinstance(shape, se.Group):  # noqa: SIM114
                continue
            elif isinstance(shape, se.Path):
                mob = self.path_to_mobject(shape)
            elif isinstance(shape, se.SimpleLine):
                mob = self.line_to_mobject(shape)
            elif isinstance(shape, se.Rect):
                mob = self.rect_to_mobject(shape)
            elif isinstance(shape, (se.Circle, se.Ellipse)):
                mob = self.ellipse_to_mobject(shape)
            elif isinstance(shape, se.Polygon):
                mob = self.polygon_to_mobject(shape)
            elif isinstance(shape, se.Polyline):
                mob = self.polyline_to_mobject(shape)
            elif isinstance(shape, se.Text):
                mob = self.text_to_mobject(shape)
            elif isinstance(shape, se.Use) or type(shape) is se.SVGElement:
                continue
            else:
                logger.warning(f"Unsupported element type: {type(shape)}")
                continue
            if mob is None or not mob.has_points():
                continue
            self.apply_style_to_mobject(mob, shape)
            if isinstance(shape, se.Transformable) and shape.apply:
                self.handle_transform(mob, shape.transform)
            result.append(mob)
        return result

    @staticmethod
    def handle_transform(mob: VMobject, matrix: se.Matrix) -> VMobject:
        """Apply SVG transformations to the converted mobject.

        Parameters
        ----------
        mob
            The converted mobject.
        matrix
            The transformation matrix determined from the SVG
            transformation.
        """
        mat = np.array([[matrix.a, matrix.c], [matrix.b, matrix.d]])
        vec = np.array([matrix.e, matrix.f, 0.0])
        mob.apply_matrix(mat)
        mob.shift(vec)
        return mob

    @staticmethod
    def apply_style_to_mobject(mob: VMobject, shape: se.GraphicObject) -> VMobject:
        """Apply SVG style information to the converted mobject.

        Parameters
        ----------
        mob
            The converted mobject.
        shape
            The parsed SVG element.
        """
        mob.set_style(
            stroke_width=shape.stroke_width,
            stroke_color=shape.stroke.hexrgb,
            stroke_opacity=shape.stroke.opacity,
            fill_color=shape.fill.hexrgb,
            fill_opacity=shape.fill.opacity,
        )
        return mob

    def path_to_mobject(self, path: se.Path) -> VMobjectFromSVGPath:
        """Convert a path element to a vectorized mobject.

        Parameters
        ----------
        path
            The parsed SVG path.
        """
        return VMobjectFromSVGPath(path, **self.path_string_config)

    @staticmethod
    def line_to_mobject(line: se.Line) -> Line:
        """Convert a line element to a vectorized mobject.

        Parameters
        ----------
        line
            The parsed SVG line.
        """
        return Line(
            start=_convert_point_to_3d(line.x1, line.y1),
            end=_convert_point_to_3d(line.x2, line.y2),
        )

    @staticmethod
    def rect_to_mobject(rect: se.Rect) -> Rectangle:
        """Convert a rectangle element to a vectorized mobject.

        Parameters
        ----------
        rect
            The parsed SVG rectangle.
        """
        if rect.rx == 0 or rect.ry == 0:
            mob = Rectangle(
                width=rect.width,
                height=rect.height,
            )
        else:
            mob = RoundedRectangle(
                width=rect.width,
                height=rect.height * rect.rx / rect.ry,
                corner_radius=rect.rx,
            )
            mob.stretch_to_fit_height(rect.height)
        mob.shift(
            _convert_point_to_3d(rect.x + rect.width / 2, rect.y + rect.height / 2)
        )
        return mob

    @staticmethod
    def ellipse_to_mobject(ellipse: se.Ellipse | se.Circle) -> Circle:
        """Convert an ellipse or circle element to a vectorized mobject.

        Parameters
        ----------
        ellipse
            The parsed SVG ellipse or circle.
        """
        mob = Circle(radius=ellipse.rx)
        if ellipse.rx != ellipse.ry:
            mob.stretch_to_fit_height(2 * ellipse.ry)
        mob.shift(_convert_point_to_3d(ellipse.cx, ellipse.cy))
        return mob

    @staticmethod
    def polygon_to_mobject(polygon: se.Polygon) -> Polygon:
        """Convert a polygon element to a vectorized mobject.

        Parameters
        ----------
        polygon
            The parsed SVG polygon.
        """
        points = [_convert_point_to_3d(*point) for point in polygon]
        return Polygon(*points)

    def polyline_to_mobject(self, polyline: se.Polyline) -> VMobject:
        """Convert a polyline element to a vectorized mobject.

        Parameters
        ----------
        polyline
            The parsed SVG polyline.
        """
        points = [_convert_point_to_3d(*point) for point in polyline]
        vmobject_class = self.get_mobject_type_class()
        return vmobject_class().set_points_as_corners(points)

    @staticmethod
    def text_to_mobject(text: se.Text):
        """Convert a text element to a vectorized mobject.

        .. warning::

            Not yet implemented.

        Parameters
        ----------
        text
            The parsed SVG text.
        """
        logger.warning(f"Unsupported element type: {type(text)}")
        return

    def move_into_position(self) -> None:
        """Scale and move the generated mobject into position."""
        if self.should_center:
            self.center()
        if self.svg_height is not None:
            self.set(height=self.svg_height)
        if self.svg_width is not None:
            self.set(width=self.svg_width)


class VMobjectFromSVGPath(VMobject, metaclass=ConvertToOpenGL):
    """A vectorized mobject representing an SVG path.

    .. note::

        The ``long_lines``, ``should_subdivide_sharp_curves``,
        and ``should_remove_null_curves`` keyword arguments are
        only respected with the OpenGL renderer.

    Parameters
    ----------
    path_obj
        A parsed SVG path object.
    long_lines
        Whether or not straight lines in the vectorized mobject
        are drawn in one or two segments.
    should_subdivide_sharp_curves
        Whether or not to subdivide subcurves further in case
        two segments meet at an angle that is sharper than a
        given threshold.
    should_remove_null_curves
        Whether or not to remove subcurves of length 0.
    kwargs
        Further keyword arguments are passed to the parent
        class.
    """

    def __init__(
        self,
        path_obj: se.Path,
        long_lines: bool = False,
        should_subdivide_sharp_curves: bool = False,
        should_remove_null_curves: bool = False,
        **kwargs,
    ):
        # Get rid of arcs
        path_obj.approximate_arcs_with_quads()
        self.path_obj = path_obj

        self.long_lines = long_lines
        self.should_subdivide_sharp_curves = should_subdivide_sharp_curves
        self.should_remove_null_curves = should_remove_null_curves

        super().__init__(**kwargs)

    def init_points(self) -> None:
        # TODO: cache mobject in a re-importable way

        self.handle_commands()

        if config.renderer == "opengl":
            if self.should_subdivide_sharp_curves:
                # For a healthy triangulation later
                self.subdivide_sharp_curves()
            if self.should_remove_null_curves:
                # Get rid of any null curves
                self.set_points(self.get_points_without_null_curves())

    generate_points = init_points

    def handle_commands(self) -> None:
        all_points: list[np.ndarray] = []
        last_move = None
        curve_start = None
        last_true_move = None

        def move_pen(pt, *, true_move: bool = False):
            nonlocal last_move, curve_start, last_true_move
            last_move = pt
            if curve_start is None:
                curve_start = last_move
            if true_move:
                last_true_move = last_move

        if self.n_points_per_curve == 4:

            def add_cubic(start, cp1, cp2, end):
                nonlocal all_points
                assert len(all_points) % 4 == 0, len(all_points)
                all_points += [start, cp1, cp2, end]
                move_pen(end)

            def add_quad(start, cp, end):
                add_cubic(start, (start + cp + cp) / 3, (cp + cp + end) / 3, end)
                move_pen(end)

            def add_line(start, end):
                add_cubic(
                    start, (start + start + end) / 3, (start + end + end) / 3, end
                )
                move_pen(end)

        else:

            def add_cubic(start, cp1, cp2, end):
                nonlocal all_points
                assert len(all_points) % 3 == 0, len(all_points)
                two_quads = get_quadratic_approximation_of_cubic(
                    start,
                    cp1,
                    cp2,
                    end,
                )
                all_points += two_quads[:3].tolist()
                all_points += two_quads[3:].tolist()
                move_pen(end)

            def add_quad(start, cp, end):
                nonlocal all_points
                assert len(all_points) % 3 == 0, len(all_points)
                all_points += [start, cp, end]
                move_pen(end)

            def add_line(start, end):
                add_quad(start, (start + end) / 2, end)
                move_pen(end)

        for segment in self.path_obj:
            segment_class = segment.__class__
            if segment_class == se.Move:
                move_pen(_convert_point_to_3d(*segment.end), true_move=True)
            elif segment_class == se.Line:
                add_line(last_move, _convert_point_to_3d(*segment.end))
            elif segment_class == se.QuadraticBezier:
                add_quad(
                    last_move,
                    _convert_point_to_3d(*segment.control),
                    _convert_point_to_3d(*segment.end),
                )
            elif segment_class == se.CubicBezier:
                add_cubic(
                    last_move,
                    _convert_point_to_3d(*segment.control1),
                    _convert_point_to_3d(*segment.control2),
                    _convert_point_to_3d(*segment.end),
                )
            elif segment_class == se.Close:
                # If the SVG path naturally ends at the beginning of the curve,
                # we do *not* need to draw a closing line. To account for floating
                # point precision, we use a small value to compare the two points.
                if abs(np.linalg.norm(last_move - last_true_move)) > 0.0001:
                    add_line(last_move, last_true_move)
                curve_start = None
            else:
                raise AssertionError(f"Not implemented: {segment_class}")

        self.points = np.array(all_points, ndmin=2, dtype="float64")
        # If we have no points, make sure the array is shaped properly
        # (0 rows tall by 3 columns wide) so future operations can
        # add or remove points correctly.
        if len(all_points) == 0:
            self.points = np.reshape(self.points, (0, 3))
