from __future__ import annotations

import hashlib
import os
from xml.etree import ElementTree as ET

import numpy as np
import svgelements as se

from manim import config, logger

from ...constants import RIGHT
from ...utils.images import get_full_vector_image_path
from ...utils.iterables import hash_obj
from ..geometry.arc import Circle
from ..geometry.line import Line
from ..geometry.polygram import Polygon, Polyline, Rectangle, RoundedRectangle
from ..opengl.opengl_compatibility import ConvertToOpenGL
from ..types.vectorized_mobject import VMobject

__all__ = ["SVGMobject", "VMobjectFromSVGPath"]


SVG_HASH_TO_MOB_MAP: dict[int, VMobject] = {}


def _convert_point_to_3d(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 0.0])


class SVGMobject(VMobject, metaclass=ConvertToOpenGL):
    """TODO: docstring"""

    def __init__(
        self,
        file_name: str | None = None,
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
        **kwargs,
    ):
        super().__init__(color=None, stroke_color=None, fill_color=None, **kwargs)

        # process keyword arguments
        self.file_name = file_name

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

        self.init_svg_mobject()

        self.set_style(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_color=stroke_color,
            stroke_opacity=stroke_opacity,
            stroke_width=stroke_width,
        )
        self.move_into_position()

    def init_svg_mobject(self) -> None:
        hash_val = hash_obj(self.hash_seed)
        if hash_val in SVG_HASH_TO_MOB_MAP:
            mob = SVG_HASH_TO_MOB_MAP[hash_val].copy()
            self.add(*mob)
            return

        self.generate_mobject()
        SVG_HASH_TO_MOB_MAP[hash_val] = self.copy()

    @property
    def hash_seed(self) -> tuple:
        # Returns data which can uniquely represent the result of `init_points`.
        # The hashed value of it is stored as a key in `SVG_HASH_TO_MOB_MAP`.
        return (
            self.__class__.__name__,
            self.svg_default,
            self.path_string_config,
            self.file_name,
            config.renderer,
        )

    def generate_mobject(self) -> None:
        file_path = self.get_file_path()
        element_tree = ET.parse(file_path)
        new_tree = self.modify_xml_tree(element_tree)
        # Create a temporary svg file to dump modified svg to be parsed
        root, ext = os.path.splitext(file_path)
        modified_file_path = root + "_" + ext
        new_tree.write(modified_file_path)

        svg = se.SVG.parse(modified_file_path)
        os.remove(modified_file_path)

        mobjects = self.get_mobjects_from(svg)
        self.add(*mobjects)
        self.flip(RIGHT)  # Flip y

    def get_file_path(self) -> str:
        if self.file_name is None:
            raise Exception("Must specify file for SVGMobject")
        return get_full_vector_image_path(self.file_name)

    def modify_xml_tree(self, element_tree: ET.ElementTree) -> ET.ElementTree:
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
        result = []
        for shape in svg.elements():
            if isinstance(shape, se.Group):
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
            # elif isinstance(shape, se.Text):
            #     mob = self.text_to_mobject(shape)
            elif type(shape) == se.SVGElement:
                continue
            else:
                logger.warning(f"Unsupported element type: {type(shape)}")
                continue
            if not mob.has_points():
                continue
            self.apply_style_to_mobject(mob, shape)
            if isinstance(shape, se.Transformable) and shape.apply:
                self.handle_transform(mob, shape.transform)
            result.append(mob)
        return result

    @staticmethod
    def handle_transform(mob: VMobject, matrix: se.Matrix) -> VMobject:
        mat = np.array([[matrix.a, matrix.c], [matrix.b, matrix.d]])
        vec = np.array([matrix.e, matrix.f, 0.0])
        mob.apply_matrix(mat)
        mob.shift(vec)
        return mob

    @staticmethod
    def apply_style_to_mobject(mob: VMobject, shape: se.GraphicObject) -> VMobject:
        mob.set_style(
            stroke_width=shape.stroke_width,
            stroke_color=shape.stroke.hexrgb,
            stroke_opacity=shape.stroke.opacity,
            fill_color=shape.fill.hexrgb,
            fill_opacity=shape.fill.opacity,
        )
        return mob

    @staticmethod
    def handle_transform(mob, matrix):
        mat = np.array([[matrix.a, matrix.c], [matrix.b, matrix.d]])
        vec = np.array([matrix.e, matrix.f, 0.0])
        mob.apply_matrix(mat)
        mob.shift(vec)
        return mob

    def path_to_mobject(self, path: se.Path) -> VMobjectFromSVGPath:
        return VMobjectFromSVGPath(path, **self.path_string_config)

    @staticmethod
    def line_to_mobject(line: se.Line) -> Line:
        return Line(
            start=_convert_point_to_3d(line.x1, line.y1),
            end=_convert_point_to_3d(line.x2, line.y2),
        )

    @staticmethod
    def rect_to_mobject(rect: se.Rect) -> Rectangle:
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
        mob = Circle(radius=ellipse.rx)
        if ellipse.rx != ellipse.ry:
            mob.stretch_to_fit_height(2 * ellipse.ry)
        mob.shift(_convert_point_to_3d(ellipse.cx, ellipse.cy))
        return mob

    @staticmethod
    def polygon_to_mobject(polygon: se.Polygon) -> Polygon:
        points = [_convert_point_to_3d(*point) for point in polygon]
        return Polygon(*points)

    @staticmethod
    def polyline_to_mobject(polyline: se.Polyline) -> Polyline:
        points = [_convert_point_to_3d(*point) for point in polyline]
        return Polyline(*points)

    @staticmethod
    def text_to_mobject(text: se.Text):
        pass

    def move_into_position(self) -> None:
        if self.should_center:
            self.center()
        if self.svg_height is not None:
            self.set_height(self.svg_height)
        if self.svg_width is not None:
            self.set_width(self.svg_width)


class VMobjectFromSVGPath(VMobject, metaclass=ConvertToOpenGL):
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
        segment_class_to_func_map = {
            se.Move: (self.start_new_path, ("end",)),
            se.Close: (self.close_path, ()),
            se.Line: (self.add_line_to, ("end",)),
            se.QuadraticBezier: (
                self.add_quadratic_bezier_curve_to,
                ("control", "end"),
            ),
            se.CubicBezier: (
                self.add_cubic_bezier_curve_to,
                ("control1", "control2", "end"),
            ),
        }
        for segment in self.path_obj:
            segment_class = segment.__class__
            func, attr_names = segment_class_to_func_map[segment_class]
            points = [
                _convert_point_to_3d(*segment.__getattribute__(attr_name))
                for attr_name in attr_names
            ]
            func(*points)

        # Get rid of the side effect of trailing "Z M" commands.
        if self.has_new_path_started():
            self.resize_points(self.get_num_points() - 1)
