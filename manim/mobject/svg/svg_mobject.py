"""Mobjects generated from an SVG file."""

from __future__ import annotations

__all__ = ["SVGMobject"]

import itertools as it
import os
import re
import string
import warnings
from pathlib import Path
from xml.dom.minidom import Element as MinidomElement
from xml.dom.minidom import parse as minidom_parse

import numpy as np
from colour import Color

from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.line import Line
from manim.mobject.geometry.polygram import Rectangle, RoundedRectangle
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL

from ... import config, logger
from ...constants import *
from ...mobject.types.vectorized_mobject import VMobject
from .style_utils import cascade_element_style, parse_style
from .svg_path import SVGPathMobject


class SVGMobject(VMobject, metaclass=ConvertToOpenGL):
    """A SVGMobject is a Vector Mobject constructed from an SVG (or XDV) file.

    SVGMobjects are constructed from the XML data within the SVG file
    structure. As such, subcomponents from the XML data can be accessed via
    the submobjects attribute. There is varying amounts of support for SVG
    elements, experiment with SVG files at your own peril.

    Examples
    --------

    .. code-block:: python

        class Sample(Scene):
            def construct(self):
                self.play(FadeIn(SVGMobject("manim-logo-sidebar.svg")))

    Parameters
    --------
    file_name : :class:`str` or :class:`pathlib.Path`
        The file's path name. When possible, the full path is preferred but a
        relative path may be used as well. Relative paths are relative to the
        directory specified by the `--assets_dir` command line argument.

    Other Parameters
    --------
    should_center : :class:`bool`
        Whether the SVGMobject should be centered to the origin. Defaults to `True`.
    height : :class:`float`
        Specify the final height of the SVG file. Defaults to 2 units.
    width : :class:`float`
        Specify the width the SVG file should occupy. Defaults to `None`.
    unpack_groups : :class:`bool`
        Whether the hierarchies of VGroups generated should be flattened. Defaults to `True`.
    stroke_width : :class:`float`
        The stroke width of the outer edge of an SVG path element. Defaults to `4`.
    fill_opacity : :class:`float`
        Specifies the opacity of the image. `1` is opaque, `0` is transparent. Defaults to `1`.
    """

    def __init__(
        self,
        file_name=None,
        should_center=True,
        height=2,
        width=None,
        unpack_groups=True,  # if False, creates a hierarchy of VGroups
        stroke_width=DEFAULT_STROKE_WIDTH,
        fill_opacity=1.0,
        should_subdivide_sharp_curves=False,
        should_remove_null_curves=False,
        color=None,
        *,
        fill_color=None,
        stroke_color=None,
        stroke_opacity=1.0,
        **kwargs,
    ):

        self.def_map = {}
        self.file_name = Path(file_name)
        self._ensure_valid_file()
        self.should_center = should_center
        self.unpack_groups = unpack_groups
        self.path_string_config = (
            {
                "should_subdivide_sharp_curves": should_subdivide_sharp_curves,
                "should_remove_null_curves": should_remove_null_curves,
            }
            if config.renderer == "opengl"
            else {}
        )
        self._initial_svg_style = self.generate_style(
            Color(color) if color else None,
            Color(fill_color) if fill_color else None,
            Color(stroke_color) if stroke_color else None,
            fill_opacity,
            stroke_opacity,
        )
        super().__init__(
            color=color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            fill_color=fill_color,
            stroke_opacity=stroke_opacity,
            stroke_color=stroke_color,
            **kwargs,
        )
        self._move_into_position(width, height)

    def _ensure_valid_file(self):
        """Reads self.file_name and determines whether the given input file_name
        is valid.
        """
        if self.file_name is None:
            raise Exception("Must specify file for SVGMobject")

        if self.file_name.exists():
            self.file_path = self.file_name
            return

        relative = Path.cwd() / self.file_name
        if relative.exists():
            self.file_path = relative
            return

        possible_paths = [
            config.get_dir("assets_dir") / self.file_name,
            config.get_dir("assets_dir") / self.file_name.with_suffix(".svg"),
            config.get_dir("assets_dir") / self.file_name.with_suffix(".xdv"),
            self.file_path,
            self.file_path.with_suffix(".svg"),
            self.file_path.with_suffix(".xdv"),
        ]
        for path in possible_paths:
            if path.exists():
                self.file_path = path
                return
        error = f"From: {Path.cwd()}, could not find {self.file_name} at either of these locations: {possible_paths}"
        raise OSError(error)

    def generate_points(self):
        """Called by the Mobject abstract base class. Responsible for generating
        the SVGMobject's points from XML tags, populating self.mobjects, and
        any submobjects within self.mobjects.
        """
        doc = minidom_parse(str(self.file_path))
        for node in doc.childNodes:
            if not isinstance(node, MinidomElement) or node.tagName != "svg":
                continue
            mobjects = self._get_mobjects_from(node, self._initial_svg_style)
            if self.unpack_groups:
                self.add(*mobjects)
            else:
                self.add(*mobjects[0].submobjects)
        doc.unlink()

    init_points = generate_points

    def _get_mobjects_from(
        self,
        element: MinidomElement,
        inherited_style: dict[str, str],
        within_defs: bool = False,
    ) -> list[VMobject]:
        """Parses a given SVG element into a Mobject.

        Parameters
        ----------
        element : :class:`Element`
            The SVG data in the XML to be parsed.

        inherited_style : :class:`dict`
            Dictionary of the SVG attributes for children to inherit.

        within_defs : :class:`bool`
            Whether ``element`` is within a ``defs`` element, which indicates
            whether elements with `id` attributes should be added to the
            definitions list.

        Returns
        -------
        List[VMobject]
            A VMobject representing the associated SVG element.
        """

        result = []
        # First, let all non-elements pass (like text entries)
        if not isinstance(element, MinidomElement):
            return result

        style = cascade_element_style(element, inherited_style)
        is_defs = element.tagName == "defs"

        if element.tagName == "style":
            pass  # TODO, handle style
        elif element.tagName in ["g", "svg", "symbol", "defs"]:
            result += it.chain(
                *(
                    self._get_mobjects_from(
                        child,
                        style,
                        within_defs=within_defs or is_defs,
                    )
                    for child in element.childNodes
                )
            )
        elif element.tagName == "path":
            temp = element.getAttribute("d")
            if temp != "":
                result.append(self._path_string_to_mobject(temp, style))
        elif element.tagName == "use":
            # note, style is calcuated in a different way for `use` elements.
            result += self._use_to_mobjects(element, style)
        elif element.tagName in ["line"]:
            result.append(self._line_to_mobject(element, style))
        elif element.tagName == "rect":
            result.append(self._rect_to_mobject(element, style))
        elif element.tagName == "circle":
            result.append(self._circle_to_mobject(element, style))
        elif element.tagName == "ellipse":
            result.append(self._ellipse_to_mobject(element, style))
        elif element.tagName in ["polygon", "polyline"]:
            result.append(self._polygon_to_mobject(element, style))
        else:
            pass  # TODO

        result = [m for m in result if m is not None]
        group_cls = self.get_group_class()

        self._handle_transforms(element, group_cls(*result))
        if len(result) > 1 and not self.unpack_groups:
            result = [group_cls(*result)]

        if within_defs and element.hasAttribute("id"):
            # it seems wasteful to throw away the actual element,
            # but I'd like the parsing to be as similar as possible
            self.def_map[element.getAttribute("id")] = (style, element)
        if is_defs:
            # defs shouldn't be part of the result tree, only the id dictionary.
            return []

        return result

    def generate_style(
        self,
        color: Color | None,
        fill_color: Color | None,
        stroke_color: Color | None,
        fill_opacity: float,
        stroke_opacity: float,
    ):
        style = {
            "fill-opacity": fill_opacity,
            "stroke-opacity": stroke_opacity,
        }
        if color:
            style["fill"] = style["stroke"] = color.get_hex_l()
        if fill_color:
            style["fill"] = fill_color.hex_l
        if stroke_color:
            style["stroke"] = stroke_color.hex_l

        return style

    def _path_string_to_mobject(self, path_string: str, style: dict):
        """Converts a SVG path element's ``d`` attribute to a mobject.

        Parameters
        ----------
        path_string : :class:`str`
            A path with potentially multiple path commands to create a shape.

        style : :class:`dict`
            Style specification, using the SVG names for properties.

        Returns
        -------
        SVGPathMobject
            A VMobject from the given path string, or d attribute.
        """
        return SVGPathMobject(
            path_string, **self.path_string_config, **parse_style(style)
        )

    def _attribute_to_float(self, attr):
        """A helper method which converts the attribute to float.

        Parameters
        ----------
        attr : str
            An SVG path attribute.

        Returns
        -------
        float
            A float representing the attribute string value.
        """
        stripped_attr = "".join(
            [char for char in attr if char in string.digits + ".-e"],
        )
        return float(stripped_attr)

    def _use_to_mobjects(
        self,
        use_element: MinidomElement,
        local_style: dict,
    ) -> list[VMobject]:
        """Converts a SVG <use> element to a collection of VMobjects.

        Parameters
        ----------
        use_element : :class:`MinidomElement`
            An SVG <use> element which represents nodes that should be
            duplicated elsewhere.

        local_style : :class:`Dict`
            The styling using SVG property names at the point the element is `<use>`d.
            Not all values are applied; styles defined when the element is specified in
            the `<def>` tag cannot be overridden here.

        Returns
        -------
        List[VMobject]
            A collection of VMobjects that are a copy of the defined object
        """

        # Remove initial "#" character
        ref = use_element.getAttribute("xlink:href")[1:]

        try:
            def_style, def_element = self.def_map[ref]
        except KeyError:
            warning_text = f"{self.file_name} contains a reference to id #{ref}, which is not recognized"
            warnings.warn(warning_text)
            return []

        # In short, the def-ed style overrides the new style,
        # in cases when the def-ed styled is defined.
        style = local_style.copy()
        style.update(def_style)

        return self._get_mobjects_from(def_element, style)

    def _line_to_mobject(self, line_element: MinidomElement, style: dict):
        """Creates a Line VMobject from an SVG <line> element.

        Parameters
        ----------
        line_element : :class:`minidom.Element`
            An SVG line element.

        style : :class:`dict`
            Style specification, using the SVG names for properties.

        Returns
        -------
        Line
            A Line VMobject
        """
        x1, y1, x2, y2 = (
            self._attribute_to_float(line_element.getAttribute(key))
            if line_element.hasAttribute(key)
            else 0.0
            for key in ("x1", "y1", "x2", "y2")
        )
        return Line([x1, -y1, 0], [x2, -y2, 0], **parse_style(style))

    def _rect_to_mobject(self, rect_element: MinidomElement, style: dict):
        """Converts a SVG <rect> command to a VMobject.

        Parameters
        ----------
        rect_element : minidom.Element
            A SVG rect path command.

        style : dict
            Style specification, using the SVG names for properties.

        Returns
        -------
        Rectangle
            Creates either a Rectangle, or RoundRectangle, VMobject from a
            rect element.
        """

        stroke_width = rect_element.getAttribute("stroke-width")
        corner_radius = rect_element.getAttribute("rx")

        if stroke_width in ["", "none", "0"]:
            stroke_width = 0

        if corner_radius in ["", "0", "none"]:
            corner_radius = 0

        corner_radius = float(corner_radius)

        parsed_style = parse_style(style)
        parsed_style["stroke_width"] = stroke_width

        if corner_radius == 0:
            mob = Rectangle(
                width=self._attribute_to_float(rect_element.getAttribute("width")),
                height=self._attribute_to_float(rect_element.getAttribute("height")),
                **parsed_style,
            )
        else:
            mob = RoundedRectangle(
                width=self._attribute_to_float(rect_element.getAttribute("width")),
                height=self._attribute_to_float(rect_element.getAttribute("height")),
                corner_radius=corner_radius,
                **parsed_style,
            )

        mob.shift(mob.get_center() - mob.get_corner(UP + LEFT))
        return mob

    def _circle_to_mobject(self, circle_element: MinidomElement, style: dict):
        """Creates a Circle VMobject from a SVG <circle> command.

        Parameters
        ----------
        circle_element : :class:`minidom.Element`
            A SVG circle path command.

        style : :class:`dict`
            Style specification, using the SVG names for properties.

        Returns
        -------
        Circle
            A Circle VMobject
        """
        x, y, r = (
            self._attribute_to_float(circle_element.getAttribute(key))
            if circle_element.hasAttribute(key)
            else 0.0
            for key in ("cx", "cy", "r")
        )
        return Circle(radius=r, **parse_style(style)).shift(x * RIGHT + y * DOWN)

    def _ellipse_to_mobject(self, circle_element: MinidomElement, style: dict):
        """Creates a stretched Circle VMobject from a SVG <circle> path
        command.

        Parameters
        ----------
        circle_element : :class:`minidom.Element`
            A SVG circle path command.

        style : :class:`dict`
            Style specification, using the SVG names for properties.

        Returns
        -------
        Circle
            A Circle VMobject
        """
        x, y, rx, ry = (
            self._attribute_to_float(circle_element.getAttribute(key))
            if circle_element.hasAttribute(key)
            else 0.0
            for key in ("cx", "cy", "rx", "ry")
        )
        return (
            Circle(**parse_style(style))
            .scale(rx * RIGHT + ry * UP)
            .shift(x * RIGHT + y * DOWN)
        )

    def _polygon_to_mobject(self, polygon_element: MinidomElement, style: dict):
        """Constructs a VMobject from a SVG <polygon> element.

        Parameters
        ----------
        polygon_element : :class:`minidom.Element`
            An SVG polygon element.

        style : :class:`dict`
            Style specification, using the SVG names for properties.

        Returns
        -------
        SVGPathMobject
            A VMobject representing the polygon.
        """
        # This seems hacky... yes it is.
        path_string = polygon_element.getAttribute("points").lstrip()
        for digit in string.digits:
            path_string = path_string.replace(" " + digit, " L" + digit)
        path_string = "M" + path_string
        if polygon_element.tagName == "polygon":
            path_string = path_string + "Z"
        return self._path_string_to_mobject(path_string, style)

    def _handle_transforms(self, element, mobject):
        """Applies the SVG transform to the specified mobject. Transforms include:
        ``matrix``, ``translate``, and ``scale``.

        Parameters
        ----------
        element : :class:`minidom.Element`
            The transform command to perform

        mobject : :class:`Mobject`
            The Mobject to transform.
        """

        x, y = (
            self._attribute_to_float(element.getAttribute(key))
            if element.hasAttribute(key)
            else 0.0
            for key in ("x", "y")
        )
        mobject.shift(x * RIGHT + y * DOWN)

        transform_attr_value = element.getAttribute("transform")

        # parse the various transforms in the attribute value
        transform_names = ["matrix", "translate", "scale", "rotate", "skewX", "skewY"]

        # Borrowed/Inspired from:
        # https://github.com/cjlano/svg/blob/3ea3384457c9780fa7d67837c9c5fd4ebc42cb3b/svg/svg.py#L75

        # match any SVG transformation with its parameter (until final parenthesis)
        # [^)]*    == anything but a closing parenthesis
        # '|'.join == OR-list of SVG transformations
        transform_regex = "|".join([x + r"[^)]*\)" for x in transform_names])
        transforms = re.findall(transform_regex, transform_attr_value)[::-1]

        number_regex = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

        for t in transforms:
            op_name, op_args = t.split("(")
            op_name = op_name.strip()
            op_args = [float(x) for x in re.findall(number_regex, op_args)]

            if op_name == "matrix":
                transform_args = np.array(op_args).reshape([3, 2])
                x = transform_args[2][0]
                y = -transform_args[2][1]
                matrix = np.identity(self.dim)
                matrix[:2, :2] = transform_args[:2, :]
                matrix[1] *= -1
                matrix[:, 1] *= -1

                for mob in mobject.family_members_with_points():
                    if config["renderer"] == "opengl":
                        mob.points = np.dot(mob.points, matrix)
                    else:
                        mob.points = np.dot(mob.points, matrix)
                mobject.shift(x * RIGHT + y * UP)

            elif op_name == "scale":
                scale_values = op_args
                if len(scale_values) == 2:
                    scale_x, scale_y = scale_values
                    mobject.scale(np.array([scale_x, scale_y, 1]), about_point=ORIGIN)
                elif len(scale_values) == 1:
                    scale = scale_values[0]
                    mobject.scale(np.array([scale, scale, 1]), about_point=ORIGIN)

            elif op_name == "translate":
                if len(op_args) == 2:
                    x, y = op_args
                else:
                    x = op_args
                    y = 0
                mobject.shift(x * RIGHT + y * DOWN)

            else:
                # TODO: handle rotate, skewX and skewY
                # for now adding a warning message
                logger.warning(
                    "Handling of %s transform is not supported yet!",
                    op_name,
                )

    def _flatten(self, input_list):
        """A helper method to flatten the ``input_list`` into an 1D array."""
        output_list = []
        for i in input_list:
            if isinstance(i, list):
                output_list.extend(self._flatten(i))
            else:
                output_list.append(i)
        return output_list

    def _move_into_position(self, width, height):
        """Uses the SVGMobject's config dictionary to set the Mobject's
        width, height, and/or center it. Use ``width``, ``height``, and
        ``should_center`` respectively to modify this.
        """
        if self.should_center:
            self.center()
        if height is not None:
            self.height = height
        if width is not None:
            self.width = width

    def init_colors(self, propagate_colors=False):
        if config.renderer == "opengl":
            self.set_style(
                fill_color=self.fill_color or self.color,
                fill_opacity=self.fill_opacity,
                stroke_color=self.stroke_color or self.color,
                stroke_width=self.stroke_width,
                stroke_opacity=self.stroke_opacity,
                recurse=propagate_colors,
            )
        else:
            super().init_colors(propagate_colors=propagate_colors)
