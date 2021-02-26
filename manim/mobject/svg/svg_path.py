"""Mobjects generated from an SVG pathstring."""


__all__ = ["SVGPathMobject", "string_to_numbers", "VMobjectFromSVGPathstring"]


import re

from typing import List

from manim import logger

from ...constants import *
from ...mobject.types.vectorized_mobject import VMobject


def string_to_numbers(num_string: str) -> List[float]:
    """Parse the SVG string representing a sequence of numbers into an array of floats.

    Parameters
    ----------
    num_string : :class:`str`
        String representing a sequence of numbers, separated by commas, spaces, etc.

    Returns
    -------
    list(float)
        List of float values parsed out of the string.
    """
    num_string = num_string.replace("-", ",-")
    num_string = num_string.replace("e,-", "e-")
    float_results = []
    for s in re.split("[ ,]", num_string):
        if s != "":
            try:
                float_results.append(float(s))
            except ValueError:
                # in this case, it's something like "2.4.3.14 which should be parsed as "2.4 0.3 0.14"
                undotted_parts = s.split(".")
                float_results.append(undotted_parts[0] + "." + undotted_parts[1])
                float_results += ["." + u for u in undotted_parts[2:]]
    return float_results


class SVGPathMobject(VMobject):
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


class VMobjectFromSVGPathstring(SVGPathMobject):
    """Pure alias of SVGPathMobject, retained for backwards compatibility"""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "VMobjectFromSVGPathstring has been deprecated in favour "
            "of SVGPathMobject. Please use SVGPathMobject instead."
        )
        SVGPathMobject.__init__(self, *args, **kwargs)
