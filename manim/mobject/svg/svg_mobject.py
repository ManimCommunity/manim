"""Mobjects generated from an SVG file."""


__all__ = ["SVGMobject", "VMobjectFromSVGPathstring", "string_to_numbers"]


import itertools as it
import re
import os
import string
import warnings

from xml.dom import minidom
from colour import web2hex

from ... import config
from ...constants import *
from ...mobject.geometry import Circle
from ...mobject.geometry import Rectangle
from ...mobject.geometry import RoundedRectangle
from ...mobject.types.vectorized_mobject import VGroup
from ...mobject.types.vectorized_mobject import VMobject
from ...utils.color import *


def string_to_numbers(num_string):
    num_string = num_string.replace("-", ",-")
    num_string = num_string.replace("e,-", "e-")
    return [float(s) for s in re.split("[ ,]", num_string) if s != ""]


def cascade_element_style(element, inherited):
    """Collect the element's style attributes based upon both its inheritance and its own attributes.

    SVG uses cascading element styles. A closer ancestor's style takes precedence over a more distant ancestor's
    style. In order to correctly calculate the styles, the attributes must be passed down through the inheritance tree,
    updating where necessary.

    Note that this method only copies, see :meth:`parse_color_string` for converting from SVG
    attributes to manim keyword arguments."""

    style = dict(inherited)

    styling_attributes = ["fill", "stroke", "style", "fill-opacity", "stroke-opacity"]
    for attr in styling_attributes:
        entry = element.getAttribute(attr)
        if entry:
            style[attr] = entry

    return style


def parse_color_string(color_spec):
    """Handle the SVG-specific color strings and convert them to HTML #rrggbb format."""

    if color_spec[0:3] == "rgb":
        # these are only in integer form, but the Colour module wants them in floats.
        parsed_rgbs = [int(i) / 255.0 for i in color_spec[4:-1].split(",")]
        hex_color = rgb_to_hex(parsed_rgbs)

    elif color_spec[0] == "#":
        # its OK, parse as hex color standard.
        hex_color = color_spec

    else:
        # attempt to convert color names like "red" to hex color
        hex_color = web2hex(color_spec, force_long=True)

    return hex_color


def parse_style(svg_style):
    """Convert a dictionary of SVG attributes to Manim VMobject keyword arguments."""

    manim_style = dict()

    # style attributes trump other element-level attributes,
    # see https://www.w3.org/TR/SVG11/styling.html section 6.4, search "priority"
    # so overwrite the other attribute dictionary values.
    if "style" in svg_style:
        for style_spec in svg_style["style"].split(";"):
            try:
                key, value = style_spec.split(":")
            except ValueError as e:
                if not style_spec:
                    # there was just a stray semicolon at the end, producing an emptystring
                    pass
                else:
                    raise e
            else:
                svg_style[key] = value

    if "fill-opacity" in svg_style:
        manim_style["fill_opacity"] = float(svg_style["fill-opacity"])

    if "stroke-opacity" in svg_style:
        manim_style["stroke_opacity"] = float(svg_style["stroke-opacity"])

    # nones need to be handled specially
    if "fill" in svg_style:
        if svg_style["fill"] == "none":
            manim_style["fill_opacity"] = 0
        else:
            manim_style["fill_color"] = parse_color_string(svg_style["fill"])

    if "stroke" in svg_style:
        if svg_style["stroke"] == "none":
            manim_style["stroke_opacity"] = 0
        else:
            manim_style["stroke_color"] = parse_color_string(svg_style["stroke"])

    return manim_style


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

    # these are the default styling specifications for SVG images,
    # according to https://www.w3.org/TR/SVG/painting.html, ctrl-F for "initial"
    # This value can be overridden in more specific classes
    DEFAULT_SVG_STYLE = {
        "fill": "black",
        "fill-opacity": "1",
        "stroke": "none",
        "stroke-opacity": "1",
    }

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
        self.def_id_to_mobject = {}
        self.file_name = file_name or self.file_name
        self.ensure_valid_file()
        self.should_center = should_center
        self.height = height
        self.width = width
        self.unpack_groups = unpack_groups
        VMobject.__init__(
            self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )
        self.move_into_position()

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
        doc = minidom.parse(self.file_path)
        for svg in doc.getElementsByTagName("svg"):
            mobjects = self.get_mobjects_from(svg, self.DEFAULT_SVG_STYLE)
            if self.unpack_groups:
                self.add(*mobjects)
            else:
                self.add(*mobjects[0].submobjects)
        doc.unlink()

    def get_mobjects_from(self, element, inherited_style):
        """Parses a given SVG element into a Mobject.

        Parameters
        ----------
        element : :class:`minidom.Element`
            The SVG data in the XML to be parsed.

        inherited_style : `dict`
            Dictionary of the SVG attributes for children to inherit.

        Returns
        -------
        List[VMobject]
            A VMobject representing the associated SVG element.
        """

        result = []
        # First, let all non-elements pass (like text entries)
        if not isinstance(element, minidom.Element):
            return result

        style = cascade_element_style(element, inherited_style)
        if element.tagName == "defs":
            self.update_defs(element, style)
        elif element.tagName == "style":
            pass  # TODO, handle style
        elif element.tagName in ["g", "svg", "symbol"]:
            result += it.chain(
                *[self.get_mobjects_from(child, style) for child in element.childNodes]
            )
        elif element.tagName == "path":
            temp = element.getAttribute("d")
            if temp != "":
                result.append(self.path_string_to_mobject(temp, style))
        elif element.tagName == "use":
            # note, style is not passed down to "use" elements
            result += self.use_to_mobjects(element)
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
            # warnings.warn("Unknown element type: " + element.tagName)

        result = [m for m in result if m is not None]
        self.handle_transforms(element, VGroup(*result))
        if len(result) > 1 and not self.unpack_groups:
            result = [VGroup(*result)]

        return result

    def path_string_to_mobject(self, path_string, style):
        """Converts a SVG path element's ``d`` attribute to a mobject.

        Parameters
        ----------
        path_string : str
            A path with potentially multiple path commands to create a shape.

        style : dict
            Style specification, using the SVG names for properties.

        Returns
        -------
        VMobjectFromSVGPathstring
            A VMobject from the given path string, or d attribute.
        """
        return VMobjectFromSVGPathstring(path_string, **parse_style(style))

    def use_to_mobjects(self, use_element):
        """Converts a SVG <use> element to VMobject.

        Parameters
        ----------
        use_element : minidom.Element
            An SVG <use> element which represents nodes that should be
            duplicated elsewhere.

        Returns
        -------
        List[VMobject]
            A collection of VMobjects that are copies of the defined objects
        """

        # Remove initial "#" character
        ref = use_element.getAttribute("xlink:href")[1:]

        try:
            return [i.copy() for i in self.def_id_to_mobject[ref]]
        except KeyError:
            warnings.warn(
                "svg file contains a reference to id #%s, which is not recognized" % ref
            )
            return []

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

    def polygon_to_mobject(self, polygon_element, style):
        """Constructs a VMobject from a SVG <polygon> element.

        Parameters
        ----------
        polygon_element : minidom.Element
            An SVG polygon element.

        style : dict
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

    # <circle class="st1" cx="143.8" cy="268" r="22.6"/>

    def circle_to_mobject(self, circle_element, style):
        """Creates a Circle VMobject from a SVG <circle> command.

        Parameters
        ----------
        circle_element : minidom.Element
            A SVG circle path command.

        style : dict
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

    def ellipse_to_mobject(self, circle_element, style):
        """Creates a stretched Circle VMobject from a SVG <circle> path
        command.

        Parameters
        ----------
        circle_element : minidom.Element
            A SVG circle path command.

        style : dict
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

    def rect_to_mobject(self, rect_element, style):
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

        if corner_radius == 0:
            mob = Rectangle(
                width=self.attribute_to_float(rect_element.getAttribute("width")),
                height=self.attribute_to_float(rect_element.getAttribute("height")),
                stroke_width=stroke_width,
                **parse_style(style),
            )
        else:
            mob = RoundedRectangle(
                width=self.attribute_to_float(rect_element.getAttribute("width")),
                height=self.attribute_to_float(rect_element.getAttribute("height")),
                stroke_width=stroke_width,
                corner_radius=corner_radius,
                **parse_style(style),
            )

        mob.shift(mob.get_center() - mob.get_corner(UP + LEFT))
        return mob

    def handle_transforms(self, element, mobject):
        """Applies the SVG transform to the specified mobject. Transforms include:
        ``rotate``, ``translate``, ``scale``, and ``skew``.

        Parameters
        ----------
        element : minidom.Element
            The transform command to perform

        mobject : Mobject
            The Mobject to transform.
        """

        if element.hasAttribute("x") and element.hasAttribute("y"):
            x = self.attribute_to_float(element.getAttribute("x"))
            # Flip y
            y = -self.attribute_to_float(element.getAttribute("y"))
            mobject.shift(x * RIGHT + y * UP)

        transform = element.getAttribute("transform")
        suffix = ")"

        # igve me a syntax error.

        # Transform matrix
        prefix = "matrix("
        if transform.startswith(prefix) and transform.endswith(suffix):
            transform = transform[len(prefix) : -len(suffix)]
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

        # transform scale
        prefix = "scale("
        if transform.startswith(prefix) and transform.endswith(suffix):
            transform = transform[len(prefix) : -len(suffix)]
            scale_values = string_to_numbers(transform)
            if len(scale_values) == 2:
                scale_x, scale_y = scale_values
                mobject.scale(np.array([scale_x, scale_y, 1]), about_point=ORIGIN)
            elif len(scale_values) == 1:
                scale = scale_values[0]
                mobject.scale(np.array([scale, scale, 1]), about_point=ORIGIN)

        # transform translate
        prefix = "translate("
        if transform.startswith(prefix) and transform.endswith(suffix):
            transform = transform[len(prefix) : -len(suffix)]
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

    def update_defs(self, defs, style):
        """Update the definitions other <use> tags may reference

        Parameters
        -------
        defs : minidom.Element
            A `defs` element from an SVG file.

        style : dict
            The `defs` element's inherited style, which will be cascaded down to every child.
        """

        for child in defs.childNodes:
            if isinstance(child, minidom.Element) and child.hasAttribute("id"):
                child_as_mobject = self.get_mobjects_from(child, style)
                self.def_id_to_mobject[child.getAttribute("id")] = child_as_mobject

    def move_into_position(self):
        """Uses the SVGMobject's config dictionary to set the Mobject's
        width, height, and/or center it. Use ``width``, ``height``, and
        ``should_center`` respectively to modify this.
        """
        if self.should_center:
            self.center()
        if self.height is not None:
            self.set_height(self.height)
        if self.width is not None:
            self.set_width(self.width)

    def init_colors(self, propagate_colors=False):
        VMobject.init_colors(self, propagate_colors=propagate_colors)


class VMobjectFromSVGPathstring(VMobject):
    def __init__(self, path_string, **kwargs):
        self.path_string = path_string
        VMobject.__init__(self, **kwargs)
        self.current_path_start = np.zeros((1, self.dim))

    def get_path_commands(self):
        """Returns a list of possible path commands used within an SVG ``d``
        attribute.

        See: https://svgwg.org/svg2-draft/paths.html#DProperty for further
        details on what each path command does.

        Returns
        -------
        List[:class:`str`]
            The various upper and lower cased path commands.
        """
        result = [
            "M",  # moveto
            "L",  # lineto
            "H",  # horizontal lineto
            "V",  # vertical lineto
            "C",  # curveto
            "S",  # smooth curveto
            "Q",  # quadratic Bezier curve
            "T",  # smooth quadratic Bezier curveto
            "A",  # elliptical Arc
            "Z",  # closepath
        ]
        result += [s.lower() for s in result]
        return result

    def generate_points(self):
        """Generates points from a given an SVG ``d`` attribute."""
        pattern = "[%s]" % ("".join(self.get_path_commands()))
        pairs = list(
            zip(
                re.findall(pattern, self.path_string),
                re.split(pattern, self.path_string)[1:],
            )
        )
        # Which mobject should new points be added to
        prev_command = None
        for command, coord_string in pairs:
            self.handle_command(command, coord_string, prev_command)
            prev_command = command
        # people treat y-coordinate differently
        self.rotate(np.pi, RIGHT, about_point=ORIGIN)

    def handle_command(self, command, coord_string, prev_command):
        """Core logic for handling each of the various path commands."""
        # Relative SVG commands are specified as lowercase letters
        is_relative = command.islower()
        command = command.upper()

        # Keep track of the most recently completed point
        start_point = (
            self.points[-1] if self.points.shape[0] else np.zeros((1, self.dim))
        )

        # Produce the (absolute) coordinates of the controls and handles
        new_points = self.string_to_points(
            command, is_relative, coord_string, start_point
        )

        if command == "M":  # moveto
            self.start_new_path(new_points[0])
            for p in new_points[1:]:
                self.add_line_to(p)
            return

        elif command in ["H", "V", "L"]:  # lineto of any kind
            for p in new_points:
                self.add_line_to(p)
            return

        elif command == "C":  # Cubic
            # points must be added in groups of 3.
            for i in range(0, len(new_points), 3):
                self.add_cubic_bezier_curve_to(*new_points[i : i + 3])
            return

        elif command == "S":  # Smooth cubic
            prev_handle = start_point
            if prev_command.upper() in ["C", "S"]:
                prev_handle = self.points[-2]
            for i in range(0, len(new_points), 2):
                new_handle = 2 * start_point - prev_handle
                self.add_cubic_bezier_curve_to(
                    new_handle, new_points[i], new_points[i + 1]
                )
                start_point = new_points[i + 1]
                prev_handle = new_points[i]
            return

        elif command == "Q":  # quadratic Bezier curve
            for i in range(0, len(new_points), 2):
                self.add_quadratic_bezier_curve_to(new_points[i], new_points[i + 1])
            return

        elif command == "T":  # smooth quadratic
            prev_quad_handle = start_point
            if prev_command.upper() in ["Q", "T"]:
                # because of the conversion from quadratic to cubic,
                # our actual previous handle was 3/2 in the direction of p[-2] from p[-1]
                prev_quad_handle = 1.5 * self.points[-2] - 0.5 * self.points[-1]
            for p in new_points:
                new_quad_handle = 2 * start_point - prev_quad_handle
                self.add_quadratic_bezier_curve_to(new_quad_handle, p)
                start_point = p
                prev_quad_handle = new_quad_handle

        elif command == "A":  # elliptical Arc
            raise NotImplementedError()

        elif command == "Z":  # closepath
            self.add_line_to(self.current_path_start)
            return

    def string_to_points(self, command, is_relative, coord_string, start_point):
        """Convert an SVG command string into a sequence of absolute-positioned control points.

        Parameters
        -----
        command : `str`
            A string containing a single uppercase letter representing the SVG command.

        is_relative : `bool`
            Whether the command is relative to the end of the previous command

        coord_string : `str`
            A string that contains many comma- or space-separated numbers that defined the control points. Different
            commands require different numbers of numbers as arguments.

        start_point : `ndarray`
            If the command is relative, the position to begin the relations from.
        """

        # this call to "string to numbers" where problems like parsing 0.5.6 lie
        numbers = string_to_numbers(coord_string)

        # H and V expect a sequence of single coords, not coord pairs like the rest of the commands.
        if command == "H":
            result = np.zeros((len(numbers), self.dim))
            result[:, 0] = numbers
            if not is_relative:
                result[:, 1] = start_point[1]

        elif command == "V":
            result = np.zeros((len(numbers), self.dim))
            result[:, 1] = numbers
            if not is_relative:
                result[:, 0] = start_point[0]

        elif command == "A":
            raise NotImplementedError("Arcs are not implemented.")

        # This is where the A command must be included.
        # It has special numbers (angles?) that don't translate to points.
        else:
            num_points = len(numbers) // 2
            result = np.zeros((num_points, self.dim))
            result[:, :2] = np.array(numbers).reshape((num_points, 2))

        # If it's not relative, we don't have any more work!
        if not is_relative:
            return result

        # Each control / target point is calculated relative to the ending position of the previous curve.
        # Curves consist of multiple point listings depending on the command.
        entries = 1
        # Quadratic curves expect pairs, S expects 3 (cubic) but one is implied by smoothness
        if command in ["Q", "S"]:
            entries = 2
        # Only cubic curves expect three points.
        elif command == "C":
            entries = 3

        offset = start_point
        for i in range(result.shape[0]):
            result[i, :] = result[i, :] + offset
            if (i + 1) % entries == 0:
                offset = result[i, :]

        return result

    def get_original_path_string(self):
        """A simple getter for the path's ``d`` attribute."""
        return self.path_string

    def start_new_path(self, point):
        self.current_path_start = point
        super().start_new_path(point)
        return self
