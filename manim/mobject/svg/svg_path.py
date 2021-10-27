"""Mobjects generated from an SVG pathstring."""


__all__ = ["SVGPathMobject", "string_to_numbers"]


import re
from math import *
from typing import List

import numpy as np

from ... import config
from ...constants import *
from ..opengl_compatibility import ConvertToOpenGL
from ..types.vectorized_mobject import VMobject


def correct_out_of_range_radii(rx, ry, x1p, y1p):
    """Correction of out-of-range radii.

    See: https://www.w3.org/TR/SVG11/implnote.html#ArcCorrectionOutOfRangeRadii
    """
    # Step 1: Ensure radii are non-zero (taken care of in elliptical_arc_to_cubic_bezier).
    # Step 2: Ensure radii are positive. If rx or ry have negative signs, these are dropped;
    # the absolute value is used instead.
    rx = abs(rx)
    ry = abs(ry)
    # Step 3: Ensure radii are large enough.
    Lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if Lambda > 1:
        rx = sqrt(Lambda) * rx
        ry = sqrt(Lambda) * ry

    # Step 4: Proceed with computations.
    return rx, ry


def vector_angle(ux, uy, vx, vy):
    """Calculate the dot product angle between two vectors.

    This clamps the argument to the arc cosine due to roundoff errors
    from some SVG files.
    """
    sign = -1 if ux * vy - uy * vx < 0 else 1
    ua = sqrt(ux * ux + uy * uy)
    va = sqrt(vx * vx + vy * vy)
    dot = ux * vx + uy * vy

    # Clamp argument between [-1,1].
    return sign * acos(max(min(dot / (ua * va), 1), -1))


def get_elliptical_arc_center_parameters(x1, y1, rx, ry, phi, fA, fS, x2, y2):
    """Conversion from endpoint to center parameterization.

    See: https://www.w3.org/TR/SVG11/implnote.html#ArcConversionEndpointToCenter
    """
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    # Step 1: Compute (x1p,y1p).
    x = (x1 - x2) / 2
    y = (y1 - y2) / 2
    x1p = x * cos_phi + y * sin_phi
    y1p = -x * sin_phi + y * cos_phi

    # Correct out of range radii
    rx, ry = correct_out_of_range_radii(rx, ry, x1p, y1p)

    # Step 2: Compute (cxp,cyp).
    rx2 = rx * rx
    ry2 = ry * ry
    x1p2 = x1p * x1p
    y1p2 = y1p * y1p
    k = sqrt(max((rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2) / (rx2 * y1p2 + ry2 * x1p2), 0))
    sign = -1 if fA == fS else 1
    cxp = sign * k * (rx * y1p) / ry
    cyp = sign * k * (-ry * x1p) / rx

    # Step 3: Compute (cx,cy) from (cxp,cyp).
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    cx = cxp * cos_phi - cyp * sin_phi + x
    cy = cxp * sin_phi + cyp * cos_phi + y

    # Step 4: Compute theta1 and dtheta.
    x = (x1p - cxp) / rx
    y = (y1p - cyp) / ry
    theta1 = vector_angle(1, 0, x, y)

    x_ = (-x1p - cxp) / rx
    y_ = (-y1p - cyp) / ry
    dtheta = degrees(vector_angle(x, y, x_, y_)) % 360

    if fS == 0 and dtheta > 0:
        dtheta -= 360
    elif fS == 1 and dtheta < 0:
        dtheta += 360

    return cx, cy, theta1, radians(dtheta)


def elliptical_arc_to_cubic_bezier(x1, y1, rx, ry, phi, fA, fS, x2, y2):
    """Generate cubic bezier points to approximate SVG elliptical arc.

    See: http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
    """
    # Out of range parameters
    # See: https://www.w3.org/TR/SVG11/implnote.html#ArcOutOfRangeParameters
    # If rx or ry are 0 then this arc is treated as a
    # straight line segment (a "lineto") joining the endpoints.
    if not rx or not ry:
        return [x1, y1, x2, y2, x2, y2]

    # phi is taken mod 360 degrees and set to radians for subsequent calculations.
    phi = radians(phi % 360)

    # Any nonzero value for either of the flags fA or fS is taken to mean the value 1.
    fA = 1 if fA else 0
    fS = 1 if fS else 0

    # Convert from endpoint to center parameterization.
    cx, cy, theta1, dtheta = get_elliptical_arc_center_parameters(
        x1,
        y1,
        rx,
        ry,
        phi,
        fA,
        fS,
        x2,
        y2,
    )

    # For a given arc we should "chop" it up into segments if it is too big
    # to help miminze cubic bezier curve approximation errors.
    # If dtheta is a multiple of 90 degrees, set the limit to 90 degrees,
    # otherwise 360/10=36 degrees is a decent sweep limit.
    if degrees(dtheta) % 90 == 0:
        sweep_limit = 90
    else:
        sweep_limit = 36

    segments = int(ceil(abs(degrees(dtheta)) / sweep_limit))
    segment = dtheta / float(segments)
    current_angle = theta1
    start_x = x1
    start_y = y1
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    alpha = sin(segment) * (sqrt(4 + 3 * pow(tan(segment / 2.0), 2)) - 1) / 3.0
    bezier_points = []

    # Calculate the cubic bezier points from elliptical arc parametric equations.
    # See: (the box on page 18) http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf
    for idx in range(segments):
        next_angle = current_angle + segment

        cos_start = cos(current_angle)
        sin_start = sin(current_angle)

        e1x = -rx * cos_phi * sin_start - ry * sin_phi * cos_start
        e1y = -rx * sin_phi * sin_start + ry * cos_phi * cos_start
        q1_x = start_x + alpha * e1x
        q1_y = start_y + alpha * e1y

        cos_end = cos(next_angle)
        sin_end = sin(next_angle)

        p2x = cx + rx * cos_phi * cos_end - ry * sin_phi * sin_end
        p2y = cy + rx * sin_phi * cos_end + ry * cos_phi * sin_end

        end_x = p2x
        end_y = p2y

        if idx == segments - 1:
            end_x = x2
            end_y = y2

        e2x = -rx * cos_phi * sin_end - ry * sin_phi * cos_end
        e2y = -rx * sin_phi * sin_end + ry * cos_phi * cos_end
        q2_x = end_x - alpha * e2x
        q2_y = end_y - alpha * e2y

        bezier_points += [[q1_x, q1_y, 0], [q2_x, q2_y, 0], [end_x, end_y, 0]]
        start_x = end_x
        start_y = end_y
        current_angle = next_angle

    return bezier_points


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
                float_results.append(float(undotted_parts[0] + "." + undotted_parts[1]))
                float_results += [float("." + u) for u in undotted_parts[2:]]
    return float_results


def grouped(iterable, n):
    """Group iterable into arrays of n items."""
    return (np.array(v) for v in zip(*[iter(iterable)] * n))


class SVGPathMobject(VMobject, metaclass=ConvertToOpenGL):
    def __init__(self, path_string, **kwargs):
        self.path_string = path_string
        if config.renderer == "opengl":
            kwargs["long_lines"] = True
        super().__init__(**kwargs)
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
            ),
        )
        # Which mobject should new points be added to
        prev_command = None
        for command, coord_string in pairs:
            self.handle_command(command, coord_string, prev_command)
            prev_command = command
        if config["renderer"] == "opengl":
            if self.should_subdivide_sharp_curves:
                # For a healthy triangulation later
                self.subdivide_sharp_curves()
            if self.should_remove_null_curves:
                # Get rid of any null curves
                self.set_points(self.get_points_without_null_curves())
        # people treat y-coordinate differently
        self.rotate(np.pi, RIGHT, about_point=ORIGIN)

    init_points = generate_points

    def handle_command(self, command, coord_string, prev_command):
        """Core logic for handling each of the various path commands."""
        # Relative SVG commands are specified as lowercase letters
        is_relative = command.islower()
        command = command.upper()

        # Keep track of the most recently completed point
        if config["renderer"] == "opengl":
            points = self.points
        else:
            points = self.points
        start_point = points[-1] if points.shape[0] else np.zeros((1, self.dim))

        # Produce the (absolute) coordinates of the controls and handles
        new_points = self.string_to_points(
            command,
            is_relative,
            coord_string,
            start_point,
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
            if config["renderer"] == "opengl":
                points = self.points
            else:
                points = self.points
            prev_handle = start_point
            if prev_command.upper() in ["C", "S"]:
                prev_handle = points[-2]
            for i in range(0, len(new_points), 2):
                new_handle = 2 * start_point - prev_handle
                self.add_cubic_bezier_curve_to(
                    new_handle,
                    new_points[i],
                    new_points[i + 1],
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
                prev_quad_handle = 1.5 * points[-2] - 0.5 * points[-1]
            for p in new_points:
                new_quad_handle = 2 * start_point - prev_quad_handle
                self.add_quadratic_bezier_curve_to(new_quad_handle, p)
                start_point = p
                prev_quad_handle = new_quad_handle

        elif command == "A":  # elliptical Arc
            # points must be added in groups of 3. See `string_to_points` for
            # case that new_points can be None.
            if new_points is not None:
                for i in range(0, len(new_points), 3):
                    self.add_cubic_bezier_curve_to(*new_points[i : i + 3])
                return

        elif command == "Z":  # closepath
            if config["renderer"] == "opengl":
                self.close_path()
            else:
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

        # arcs are weirdest, handle them first.
        if command == "A":
            result = np.zeros((0, self.dim))
            last_end_point = None
            for elliptic_numbers in grouped(numbers, 7):
                # The startpoint changes with each iteration.
                if last_end_point is not None:
                    start_point = last_end_point

                # We have to handle offsets here because ellipses are complicated.
                if is_relative:
                    elliptic_numbers[5] += start_point[0]
                    elliptic_numbers[6] += start_point[1]

                # If the endpoints (x1, y1) and (x2, y2) are identical, then this
                # is equivalent to omitting the elliptical arc segment entirely.
                # for more information of where this math came from visit:
                #  http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
                if (
                    start_point[0] == elliptic_numbers[5]
                    and start_point[1] == elliptic_numbers[6]
                ):
                    continue

                result = np.append(
                    result,
                    elliptical_arc_to_cubic_bezier(*start_point[:2], *elliptic_numbers),
                    axis=0,
                )

                # We store the endpoint so that it can be the startpoint for the
                # next iteration.
                last_end_point = elliptic_numbers[5:]

            return result

        # H and V expect a sequence of single coords, not coord pairs like the rest of the commands.
        elif command == "H":
            result = np.zeros((len(numbers), self.dim))
            result[:, 0] = numbers
            if not is_relative:
                result[:, 1] = start_point[1]

        elif command == "V":
            result = np.zeros((len(numbers), self.dim))
            result[:, 1] = numbers
            if not is_relative:
                result[:, 0] = start_point[0]

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
