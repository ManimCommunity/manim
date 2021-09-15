"""Mobjects representing function graphs."""

__all__ = ["ParametricFunction", "FunctionGraph", "ImplicitFunction"]


from typing import Callable, Optional

import numpy as np

from .. import config
from ..constants import *
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.color import YELLOW
from .opengl_compatibility import ConvertToOpenGL


class ParametricFunction(VMobject, metaclass=ConvertToOpenGL):
    """A parametric curve.

    Examples
    --------

    .. manim:: PlotParametricFunction
        :save_last_frame:

        class PlotParametricFunction(Scene):
            def func(self, t):
                return np.array((np.sin(2 * t), np.sin(3 * t), 0))

            def construct(self):
                func = ParametricFunction(self.func, t_range = np.array([0, TAU]), fill_opacity=0).set_color(RED)
                self.add(func.scale(3))

    .. manim:: ThreeDParametricSpring
        :save_last_frame:

        class ThreeDParametricSpring(ThreeDScene):
            def construct(self):
                curve1 = ParametricFunction(
                    lambda u: np.array([
                        1.2 * np.cos(u),
                        1.2 * np.sin(u),
                        u * 0.05
                    ]), color=RED, t_range = np.array([-3*TAU, 5*TAU, 0.01])
                ).set_shade_in_3d(True)
                axes = ThreeDAxes()
                self.add(axes, curve1)
                self.set_camera_orientation(phi=80 * DEGREES, theta=-60 * DEGREES)
                self.wait()
    """

    def __init__(
        self,
        function=None,
        t_range=None,
        dt=1e-8,
        discontinuities=None,
        use_smoothing=True,
        **kwargs
    ):
        self.function = function
        t_range = [0, 1, 0.01] if t_range is None else t_range
        if len(t_range) == 2:
            t_range = np.array([*t_range, 0.01])

        self.dt = dt
        self.discontinuities = [] if discontinuities is None else discontinuities
        self.use_smoothing = use_smoothing
        self.t_min, self.t_max, self.t_step = t_range

        super().__init__(**kwargs)

    def get_function(self):
        return self.function

    def get_point_from_function(self, t):
        return self.function(t)

    def generate_points(self):

        discontinuities = filter(
            lambda t: self.t_min <= t <= self.t_max,
            self.discontinuities,
        )
        discontinuities = np.array(list(discontinuities))
        boundary_times = np.array(
            [
                self.t_min,
                self.t_max,
                *(discontinuities - self.dt),
                *(discontinuities + self.dt),
            ],
        )
        boundary_times.sort()
        for t1, t2 in zip(boundary_times[0::2], boundary_times[1::2]):
            t_range = np.array([*np.arange(t1, t2, self.t_step), t2])
            points = np.array([self.function(t) for t in t_range])
            self.start_new_path(points[0])
            self.add_points_as_corners(points[1:])
        if self.use_smoothing:
            # TODO: not in line with upstream, approx_smooth does not exist
            self.make_smooth()
        return self

    init_points = generate_points


class FunctionGraph(ParametricFunction):
    """A :class:`ParametricFunction` that spans the length of the scene by default.

    Examples
    --------
    .. manim:: ExampleFunctionGraph
        :save_last_frame:

        class ExampleFunctionGraph(Scene):
            def construct(self):
                cos_func = FunctionGraph(
                    lambda t: np.cos(t) + 0.5 * np.cos(7 * t) + (1 / 7) * np.cos(14 * t),
                    color=RED,
                )

                sin_func_1 = FunctionGraph(
                    lambda t: np.sin(t) + 0.5 * np.sin(7 * t) + (1 / 7) * np.sin(14 * t),
                    color=BLUE,
                )

                sin_func_2 = FunctionGraph(
                    lambda t: np.sin(t) + 0.5 * np.sin(7 * t) + (1 / 7) * np.sin(14 * t),
                    x_range=[-4, 4],
                    color=GREEN,
                ).move_to([0, 1, 0])

                self.add(cos_func, sin_func_1, sin_func_2)
    """

    def __init__(self, function, x_range=None, color=YELLOW, **kwargs):

        if x_range is None:
            x_range = np.array([-config["frame_x_radius"], config["frame_x_radius"]])

        self.x_range = x_range
        self.parametric_function = lambda t: np.array([t, function(t), 0])
        self.function = function
        super().__init__(self.parametric_function, self.x_range, color=color, **kwargs)

    def get_function(self):
        return self.function

    def get_point_from_function(self, x):
        return self.parametric_function(x)


def _symmetrize(dic: dict):
    symm = {}
    for key, value in dic.items():
        symm[value] = key

    symm.update(dic)
    return symm


from .opengl_compatibility import ConvertToOpenGL


class ImplicitFunction(VMobject, metaclass=ConvertToOpenGL):
    def __init__(
        self,
        function: Callable = None,
        ax: Optional["CoordinateSystem"] = None,
        res: int = None,
        **kwargs
    ):
        """An implicit function, relative to the scene or a :class:`CoordinateSystem`.

        Parameters
        ----------
        function
            The implicit function in the form of f(x, y) = 0
        ax
            Optional, the coordinate system to place the implicit graph
        res
            The resolution of the implicit graph
        kwargs
            Additional parameters to be passed to :class:`VMobject`
        """
        self.res = res
        self.x_min, self.x_max = -config.frame_width / 2, config.frame_width / 2
        self.y_min, self.y_max = -config.frame_height / 2, config.frame_height / 2
        self.function = function
        if not ax:
            from .coordinate_systems import NumberPlane

            ax = NumberPlane()
        if not ax.x_length:
            ax.x_length = config.frame_width
            ax.y_length = config.frame_height

        super().__init__(**kwargs)
        self.stretch(ax.x_length / config.frame_width, 0)
        self.stretch(ax.y_length / config.frame_height, 1)
        self.shift(ax.get_center())

    def get_function(self):
        return self.function

    def get_function_val_at_point(self, x, y):
        return self.function(x, y)

    def sample_function_mask(self) -> list:
        """A mask over the plane at the specified resolution capturing function
        values at each point.
        """
        delta_x = self.x_max - self.x_min
        delta_y = self.y_max - self.y_min
        mask = []  # format: [point, val]
        for yi in range(0, self.res):
            dy = delta_y * (yi / self.res)
            y = self.y_min + dy
            vals = []
            for xi in range(0, self.res):
                dx = delta_x * (xi / self.res)
                x = self.x_min + dx
                point = np.array([x, y, 0])
                val = self.get_function_val_at_point(x, y)
                if val > 0:
                    vals.append([point, 1])
                elif val <= 0:
                    vals.append([point, 0])
            mask.append(vals)
        return mask

    def get_contours(self) -> dict:
        """A dictionary consisting of start -> list(end) points to generate contours."""
        mask = self.sample_function_mask()
        contours = {}
        for yi in range(0, len(mask) - 1):
            yarr = mask[yi]
            nyarr = mask[yi + 1]
            for xi in range(0, len(yarr) - 1):
                tl = yarr[xi]
                tr = yarr[xi + 1]
                bl = nyarr[xi]
                br = nyarr[xi + 1]

                tlp = tl[0]
                tlv = tl[1]

                trp = tr[0]
                trv = tr[1]

                blp = bl[0]
                blv = bl[1]

                brp = br[0]
                brv = br[1]

                vals = [
                    tlv,
                    trv,
                    brv,
                    blv,
                ]  # change order to match marching squares order.
                vals_strs = list(map(lambda i: str(i), vals))
                vals_str = "".join(vals_strs)

                def calc_lin_interp(fc, fe, cv, ev):
                    """
                    Parameters
                    ----------
                    fc
                        Function value at 'center' vertex
                    fe
                        Function value at 'edge' vertex
                    cv
                        The 'location' of the 'center' vertex (x or y depending)
                    ev
                        Similar to above for 'edge' vertex

                    Returns
                    -------
                    float
                        The x or y coordinate of the linear interpolation
                    """
                    return -(fc / (fe - fc)) * (ev - cv) + cv

                def calc_lin_interp_diag(cent, side, vert):
                    """
                    Parameters
                    ----------
                    cent
                        'Center' point
                    side
                        'Side' point w.r.t. cent
                    vert
                        'Vertical' point w.r.t. cent

                    Returns
                    -------
                    dict
                        Dict detailing path to follow of linear interpolation.
                    """
                    centx, centy = cent[:2]

                    sidex, sidey = side[:2]

                    vertx, verty = vert[:2]

                    qx = vertx
                    py = sidey

                    f_cent = self.get_function_val_at_point(centx, centy)
                    f_vert = self.get_function_val_at_point(vertx, verty)
                    f_side = self.get_function_val_at_point(sidex, sidey)

                    qy = calc_lin_interp(f_cent, f_vert, centy, verty)
                    px = calc_lin_interp(f_cent, f_side, centx, sidex)

                    p = (px, py, 0)
                    q = (qx, qy, 0)
                    return _symmetrize({p: q})

                def calc_lin_interp_sides():
                    """
                    Returns
                    -------
                    dict
                        Horizontal linear interpolation
                    """
                    tlx, tly = tlp[:2]
                    trx, tr_y = trp[:2]
                    blx, bly = blp[:2]
                    brx, bry = brp[:2]

                    ftl = self.get_function_val_at_point(tlx, tly)
                    ftr = self.get_function_val_at_point(trx, tr_y)
                    fbl = self.get_function_val_at_point(blx, bly)
                    fbr = self.get_function_val_at_point(brx, bry)

                    px = tlx
                    qx = trx

                    py = calc_lin_interp(fbl, ftl, bly, tly)
                    qy = calc_lin_interp(fbr, ftr, bry, tr_y)

                    p = (px, py, 0)
                    q = (qx, qy, 0)

                    return _symmetrize({p: q})

                def calc_lin_interp_vert():
                    """
                    Returns
                    -------
                    dict
                        Vertical linear interpolation
                    """
                    tlx, tly = tlp[:2]
                    trx, tr_y = trp[:2]
                    blx, bly = blp[:2]
                    brx, bry = brp[:2]

                    ftl = self.get_function_val_at_point(tlx, tly)
                    ftr = self.get_function_val_at_point(trx, tr_y)
                    fbl = self.get_function_val_at_point(blx, bly)
                    fbr = self.get_function_val_at_point(brx, bry)

                    py = bly
                    qy = tly

                    px = calc_lin_interp(fbl, fbr, blx, brx)
                    qx = calc_lin_interp(ftl, ftr, tlx, trx)

                    p = (px, py, 0)
                    q = (qx, qy, 0)

                    return _symmetrize({p: q})

                m_sqrs_dict = {
                    "0000": {},
                    "0001": calc_lin_interp_diag(blp, brp, tlp),
                    "0010": calc_lin_interp_diag(brp, blp, trp),
                    "0011": calc_lin_interp_sides(),
                    "0100": calc_lin_interp_diag(trp, tlp, brp),
                    "0101": {
                        **calc_lin_interp_diag(tlp, trp, blp),
                        **calc_lin_interp_diag(brp, blp, trp),
                    },
                    "0110": calc_lin_interp_vert(),
                    "0111": calc_lin_interp_diag(tlp, trp, blp),
                    "1000": calc_lin_interp_diag(tlp, trp, blp),
                    "1001": calc_lin_interp_vert(),
                    "1010": {
                        **calc_lin_interp_diag(trp, tlp, brp),
                        **calc_lin_interp_diag(blp, brp, tlp),
                    },
                    "1011": calc_lin_interp_diag(trp, tlp, brp),
                    "1100": calc_lin_interp_sides(),
                    "1101": calc_lin_interp_diag(brp, blp, trp),
                    "1110": calc_lin_interp_diag(blp, brp, tlp),
                    "1111": {},
                }
                # Dictionary describing how to form path given the binary signature.
                dic = m_sqrs_dict[vals_str]
                for k, v in dic.items():
                    if k in contours.keys() and v not in contours[k]:
                        contours[k].append(v)
                    elif k not in contours.keys():
                        contours[k] = [v]

        return contours

    def init_points(self):
        return self.generate_points()

    def generate_points(self):
        """This generates path basically in a follow-the-points sort of manner.

        It starts at the 'first points' in the dictionary, starts a path at the
        'start' point and iteratively follows the path from the current point
        to the first point in the current point's list of adjacent points.
        It does this until there is nowhere else to go for that curve and then
        proceeds to the next curve. At every point, current points are removed from
        the contours just to ensure no vertices are visited more than once.
        """
        contours = self.get_contours()

        def try_rem(arr, val):
            if val in arr:
                arr.remove(val)
            return arr

        def len_filter(dic):
            return {k: arr for k, arr in dic.items() if len(arr) > 0}

        while len(len_filter(contours).keys()) > 0:
            sptt, eptts = next(iter(len_filter(contours).items()))
            eptt = eptts[0]
            spta = np.array(sptt)
            epta = np.array(eptt)
            contours[sptt] = try_rem(contours[sptt], eptt)
            contours = {k: try_rem(arr, sptt) for k, arr in contours.items()}
            self.start_new_path(spta)
            cur_pt = epta
            pts = []
            while cur_pt is not None:
                pts.append(cur_pt)
                cur_ptt = tuple(cur_pt)
                if len(contours[cur_ptt]) > 0:
                    next_ptt = contours[cur_ptt][0]
                    next_pt = np.array(next_ptt)
                    cur_pt = next_pt
                    contours[cur_ptt] = try_rem(contours[cur_ptt], next_ptt)
                    contours = {k: try_rem(arr, cur_ptt) for k, arr in contours.items()}
                else:
                    cur_pt = None
            self.add_points_as_corners(pts)
        self.make_smooth()
        return self
