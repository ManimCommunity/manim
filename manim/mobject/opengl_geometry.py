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
