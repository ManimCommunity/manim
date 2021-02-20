from ..utils.space_ops import angle_of_vector
from ..utils.space_ops import angle_between_vectors
from ..utils.space_ops import compass_directions
from ..utils.space_ops import find_intersection
from ..utils.space_ops import get_norm
from ..utils.space_ops import normalize
from ..utils.space_ops import rotate_vector
from ..utils.space_ops import rotation_matrix_transpose
from ..utils.iterables import adjacent_n_tuples
from ..constants import *
from ..mobject.geometry import TipableVMobject
from ..mobject.types.vectorized_mobject import VMobject


class OpenGLArc(TipableVMobject):
    """A circular arc."""

    opengl_compatible = True

    def __init__(
        self,
        start_angle=0,
        angle=TAU / 4,
        radius=1.0,
        num_components=8,
        anchors_span_full_range=True,
        arc_center=ORIGIN,
        **kwargs
    ):
        if radius is None:  # apparently None is passed by ArcBetweenPoints
            radius = 1.0
        self.radius = radius
        self.num_components = num_components
        self.anchors_span_full_range = anchors_span_full_range
        self.arc_center = arc_center
        self.start_angle = start_angle
        self.angle = angle
        self._failed_to_get_center = False
        TipableVMobject.__init__(self, **kwargs)

    def init_gl_points(self):
        self.set_points(
            OpenGLArc.create_quadratic_bezier_points(
                angle=self.angle,
                start_angle=self.start_angle,
                n_components=self.num_components,
            )
        )
        self.scale(self.radius, about_point=ORIGIN)
        self.shift(self.arc_center)

    @staticmethod
    def create_quadratic_bezier_points(angle, start_angle=0, n_components=8):
        samples = np.array(
            [
                [np.cos(a), np.sin(a), 0]
                for a in np.linspace(
                    start_angle,
                    start_angle + angle,
                    2 * n_components + 1,
                )
            ]
        )
        theta = angle / n_components
        samples[1::2] /= np.cos(theta / 2)

        points = np.zeros((3 * n_components, 3))
        points[0::3] = samples[0:-1:2]
        points[1::3] = samples[1::2]
        points[2::3] = samples[2::2]
        return points

    def get_arc_center(self):
        """
        Looks at the normals to the first two
        anchors, and finds their intersection points
        """
        # First two anchors and handles
        a1, h, a2 = self.get_points()[:3]
        # Tangent vectors
        t1 = h - a1
        t2 = h - a2
        # Normals
        n1 = rotate_vector(t1, TAU / 4)
        n2 = rotate_vector(t2, TAU / 4)
        return find_intersection(a1, n1, a2, n2)

    def get_start_angle(self):
        angle = angle_of_vector(self.get_start() - self.get_arc_center())
        return angle % TAU

    def get_stop_angle(self):
        angle = angle_of_vector(self.get_end() - self.get_arc_center())
        return angle % TAU

    def move_arc_center_to(self, point):
        self.shift(point - self.get_arc_center())
        return self


class OpenGLArcBetweenPoints(OpenGLArc):
    def __init__(self, start, end, angle=TAU / 4, **kwargs):
        super().__init__(angle=angle, **kwargs)
        if angle == 0:
            self.set_points_as_corners([LEFT, RIGHT])
        self.put_start_and_end_on(start, end)


class OpenGLPolygon(VMobject):
    def __init__(self, *vertices, **kwargs):
        self.vertices = vertices
        super().__init__(**kwargs)

    def init_points(self):
        verts = self.vertices
        self.set_points_as_corners([*verts, verts[0]])

    def get_vertices(self):
        return self.get_start_anchors()

    def round_corners(self, radius=0.5):
        vertices = self.get_vertices()
        arcs = []
        for v1, v2, v3 in adjacent_n_tuples(vertices, 3):
            vect1 = v2 - v1
            vect2 = v3 - v2
            unit_vect1 = normalize(vect1)
            unit_vect2 = normalize(vect2)
            angle = angle_between_vectors(vect1, vect2)
            # Negative radius gives concave curves
            angle *= np.sign(radius)
            # Distance between vertex and start of the arc
            cut_off_length = radius * np.tan(angle / 2)
            # Determines counterclockwise vs. clockwise
            sign = np.sign(np.cross(vect1, vect2)[2])
            arc = ArcBetweenPoints(
                v2 - unit_vect1 * cut_off_length,
                v2 + unit_vect2 * cut_off_length,
                angle=sign * angle,
                n_components=2,
            )
            arcs.append(arc)

        self.clear_points()
        # To ensure that we loop through starting with last
        arcs = [arcs[-1], *arcs[:-1]]
        for arc1, arc2 in adjacent_pairs(arcs):
            self.append_points(arc1.get_points())
            line = Line(arc1.get_end(), arc2.get_start())
            # Make sure anchors are evenly distributed
            len_ratio = line.get_length() / arc1.get_arc_length()
            line.insert_n_curves(int(arc1.get_num_curves() * len_ratio))
            self.append_points(line.get_points())
        return self
