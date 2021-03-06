"""Mobjects generated from an SVG file."""


__all__ = ["SVGMobject", "string_to_numbers"]


import itertools as it
import re
import os
import string
import warnings

from xml.dom.minidom import Element as MinidomElement, parse as minidom_parse

from typing import Dict, List

from .style_utils import cascade_element_style, parse_style
from .svg_path import SVGPathMobject, string_to_numbers
from ... import config
from ...constants import *
from ...mobject.geometry import Circle
from ...mobject.geometry import Rectangle
from ...mobject.geometry import RoundedRectangle
from ...mobject.types.vectorized_mobject import VGroup
from ...mobject.types.vectorized_mobject import VMobject


class SVGMobject(VMobject):
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
                self.play(
                    FadeIn(SVGMobject("manim-logo-sidebar.svg"))
                )
    Parameters
    --------
    file_name : :class:`str`
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
        **kwargs,
    ):
        self.def_map = {}
        self.file_name = file_name or self.file_name
        self.ensure_valid_file()
        self.should_center = should_center
        self.unpack_groups = unpack_groups
        VMobject.__init__(
            self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )
        self.move_into_position(width, height)

    def ensure_valid_file(self):
        """Reads self.file_name and determines whether the given input file_name
        is valid.
        """
        if self.file_name is None:
            raise Exception("Must specify file for SVGMobject")

        if os.path.exists(self.file_name):
            self.file_path = self.file_name
            return

        relative = os.path.join(os.getcwd(), self.file_name)
        if os.path.exists(relative):
            self.file_path = relative
            return

        possible_paths = [
            os.path.join(config.get_dir("assets_dir"), self.file_name),
            os.path.join(config.get_dir("assets_dir"), self.file_name + ".svg"),
            os.path.join(config.get_dir("assets_dir"), self.file_name + ".xdv"),
            self.file_name,
            self.file_name + ".svg",
            self.file_name + ".xdv",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                self.file_path = path
                return
        error = f"From: {os.getcwd()}, could not find {self.file_name} at either of these locations: {possible_paths}"
        raise IOError(error)

    def generate_points(self):
        """Called by the Mobject abstract base class. Responsible for generating
        the SVGMobject's points from XML tags, populating self.mobjects, and
        any submobjects within self.mobjects.
        """
        doc = minidom_parse(self.file_path)
        for svg in doc.getElementsByTagName("svg"):
            mobjects = self.get_mobjects_from(svg, {})
            if self.unpack_groups:
                self.add(*mobjects)
            else:
                self.add(*mobjects[0].submobjects)
        doc.unlink()

    def get_mobjects_from(
        self,
        element: MinidomElement,
        inherited_style: Dict[str, str],
        within_defs: bool = False,
    ) -> List[VMobject]:
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
                *[
                    self.get_mobjects_from(
                        child, style, within_defs=within_defs or is_defs
                    )
                    for child in element.childNodes
                ]
            )
        elif element.tagName == "path":
            temp = element.getAttribute("d")
            if temp != "":
                result.append(self.path_string_to_mobject(temp, style))
        elif element.tagName == "use":
            # note, style is calcuated in a different way for `use` elements.
            result += self.use_to_mobjects(element, style)
        elif element.tagName == "rect":
            result.append(self.rect_to_mobject(element, style))
        elif element.tagName == "circle":
            result.append(self.circle_to_mobject(element, style))
        elif element.tagName == "ellipse":
            result.append(self.ellipse_to_mobject(element, style))
        elif element.tagName in ["polygon", "polyline"]:
            result.append(self.polygon_to_mobject(element, style))
        else:
            pass  # TODO

        result = [m for m in result if m is not None]
        self.handle_transforms(element, VGroup(*result))
        if len(result) > 1 and not self.unpack_groups:
            result = [VGroup(*result)]

        if within_defs and element.hasAttribute("id"):
            # it seems wasteful to throw away the actual element,
            # but I'd like the parsing to be as similar as possible
            self.def_map[element.getAttribute("id")] = (style, element)
        if is_defs:
            # defs shouldn't be part of the result tree, only the id dictionary.
            return []

        return result

    def path_string_to_mobject(self, path_string: str, style: dict):
        """Converts a SVG path element's ``d`` attribute to a mobject.

        Parameters
        ----------
        path_string : :class:`str`
            A path with potentially multiple path commands to create a shape.

        style : :class:`dict`
            Style specification, using the SVG names for properties.

        Returns
        -------
        VMobjectFromSVGPathstring
            A VMobject from the given path string, or d attribute.
        """
        return SVGPathMobject(path_string, **parse_style(style))

    def use_to_mobjects(
        self, use_element: MinidomElement, local_style: Dict
    ) -> List[VMobject]:
        """Converts a SVG <use> element to a collection of VMobjects.

        Parameters
        ----------
        use_element : :class:`MinidomElement`
            An SVG <use> element which represents nodes that should be
            duplicated elsewhere.

        local_style : :class:`Dict`
            The styling using SVG property names at the point the element is `<use>`d.
            Not all values are applied; styles defined when the element is specified in
            the `<def>` tag cannot be overriden here.

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

        return self.get_mobjects_from(def_element, style)

    def attribute_to_float(self, attr):
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
            [char for char in attr if char in string.digits + "." + "-"]
        )
        return float(stripped_attr)

    def polygon_to_mobject(self, polygon_element: MinidomElement, style: dict):
        """Constructs a VMobject from a SVG <polygon> element.

        Parameters
        ----------
        polygon_element : :class:`minidom.Element`
            An SVG polygon element.

        style : :class:`dict`
            Style specification, using the SVG names for properties.

        Returns
        -------
        VMobjectFromSVGPathstring
            A VMobject representing the polygon.
        """
        # This seems hacky... yes it is.
        path_string = polygon_element.getAttribute("points").lstrip()
        for digit in string.digits:
            path_string = path_string.replace(" " + digit, " L" + digit)
        path_string = "M" + path_string
        if polygon_element.tagName == "polygon":
            path_string = path_string + "Z"
        return self.path_string_to_mobject(path_string, style)

    def circle_to_mobject(self, circle_element: MinidomElement, style: dict):
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
        x, y, r = [
            self.attribute_to_float(circle_element.getAttribute(key))
            if circle_element.hasAttribute(key)
            else 0.0
            for key in ("cx", "cy", "r")
        ]
        return Circle(radius=r, **parse_style(style)).shift(x * RIGHT + y * DOWN)

    def ellipse_to_mobject(self, circle_element: MinidomElement, style: dict):
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
        x, y, rx, ry = [
            self.attribute_to_float(circle_element.getAttribute(key))
            if circle_element.hasAttribute(key)
            else 0.0
            for key in ("cx", "cy", "rx", "ry")
        ]
        return (
            Circle(**parse_style(style))
            .scale(rx * RIGHT + ry * UP)
            .shift(x * RIGHT + y * DOWN)
        )

    def rect_to_mobject(self, rect_element: MinidomElement, style: dict):
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
                width=self.attribute_to_float(rect_element.getAttribute("width")),
                height=self.attribute_to_float(rect_element.getAttribute("height")),
                **parsed_style,
            )
        else:
            mob = RoundedRectangle(
                width=self.attribute_to_float(rect_element.getAttribute("width")),
                height=self.attribute_to_float(rect_element.getAttribute("height")),
                corner_radius=corner_radius,
                **parsed_style,
            )

        mob.shift(mob.get_center() - mob.get_corner(UP + LEFT))
        return mob

    def handle_transforms(self, element, mobject):
        """Applies the SVG transform to the specified mobject. Transforms include:
        ``rotate``, ``translate``, ``scale``, and ``skew``.

        Parameters
        ----------
        element : :class:`minidom.Element`
            The transform command to perform

        mobject : :class:`Mobject`
            The Mobject to transform.
        """

        if element.hasAttribute("x") and element.hasAttribute("y"):
            x = self.attribute_to_float(element.getAttribute("x"))
            # Flip y
            y = -self.attribute_to_float(element.getAttribute("y"))
            mobject.shift(x * RIGHT + y * UP)

        transform = element.getAttribute("transform")
        suffix = ")"

        # Transform matrix
        if transform.startswith("matrix(") and transform.endswith(suffix):
            transform = transform[len("matrix(") : -len(suffix)]
            transform = string_to_numbers(transform)
            transform = np.array(transform).reshape([3, 2])
            x = transform[2][0]
            y = -transform[2][1]
            matrix = np.identity(self.dim)
            matrix[:2, :2] = transform[:2, :]
            matrix[1] *= -1
            matrix[:, 1] *= -1

            for mob in mobject.family_members_with_points():
                mob.points = np.dot(mob.points, matrix)
            mobject.shift(x * RIGHT + y * UP)

        elif transform.startswith("scale(") and transform.endswith(suffix):
            transform = transform[len("scale(") : -len(suffix)]
            scale_values = string_to_numbers(transform)
            if len(scale_values) == 2:
                scale_x, scale_y = scale_values
                mobject.scale(np.array([scale_x, scale_y, 1]), about_point=ORIGIN)
            elif len(scale_values) == 1:
                scale = scale_values[0]
                mobject.scale(np.array([scale, scale, 1]), about_point=ORIGIN)

        elif transform.startswith("translate(") and transform.endswith(suffix):
            transform = transform[len("translate(") : -len(suffix)]
            x, y = string_to_numbers(transform)
            mobject.shift(x * RIGHT + y * DOWN)

    def flatten(self, input_list):
        """A helper method to flatten the ``input_list`` into an 1D array."""
        output_list = []
        for i in input_list:
            if isinstance(i, list):
                output_list.extend(self.flatten(i))
            else:
                output_list.append(i)
        return output_list

    def move_into_position(self, width, height):
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
        VMobject.init_colors(self, propagate_colors=propagate_colors)
