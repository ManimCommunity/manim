import OpenGL.GLU as GLU
import numpy as np
import numpy.linalg as linalg
from .. import config

depth = 20


def orthographic_projection_matrix(width=None, height=None, near=1, far=depth + 1):
    if width is None:
        width = config["frame_width"]
    if height is None:
        height = config["frame_height"]
    return tuple(
        np.array(
            [
                [2 / width, 0, 0, 0],
                [0, 2 / height, 0, 0],
                [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                [0, 0, 0, 1],
            ]
        ).T.ravel()
    )


def view_matrix(camera_position=None):
    if camera_position is None:
        camera_position = np.array([0, 0, depth / 2 + 1])
    return tuple(
        linalg.inv(
            np.array(
                [
                    [1, 0, 0, camera_position[0]],
                    [0, 1, 0, camera_position[1]],
                    [0, 0, 1, camera_position[2]],
                    [0, 0, 0, 1],
                ]
            )
        ).T.ravel()
    )


def triangulate(vertices, holes=[]):
    """
    Returns a list of triangles.
    Uses the GLU Tesselator functions!
    """
    triangle_vertices = []

    def edgeFlagCallback(param1, param2):
        pass

    def beginCallback(param=None):
        nonlocal triangle_vertices
        triangle_vertices = []

    def vertexCallback(vertex, otherData=None):
        triangle_vertices.append(vertex)

    def combineCallback(vertex, neighbors, neighborWeights, out=None):
        out = vertex
        return out

    def endCallback(data=None):
        pass

    tess = GLU.gluNewTess()
    GLU.gluTessProperty(tess, GLU.GLU_TESS_WINDING_RULE, GLU.GLU_TESS_WINDING_ODD)
    GLU.gluTessCallback(
        tess, GLU.GLU_TESS_EDGE_FLAG_DATA, edgeFlagCallback
    )  # forces triangulation of polygons (i.e. GL_TRIANGLES) rather than returning triangle fans or strips
    GLU.gluTessCallback(tess, GLU.GLU_TESS_BEGIN, beginCallback)
    GLU.gluTessCallback(tess, GLU.GLU_TESS_VERTEX, vertexCallback)
    GLU.gluTessCallback(tess, GLU.GLU_TESS_COMBINE, combineCallback)
    GLU.gluTessCallback(tess, GLU.GLU_TESS_END, endCallback)
    GLU.gluTessCallback(
        tess, GLU.GLU_TESS_ERROR, lambda x: print("Error", GLU.gluErrorString(x))
    )
    GLU.gluTessBeginPolygon(tess, 0)

    # first handle the main polygon
    GLU.gluTessBeginContour(tess)
    for point in vertices:
        GLU.gluTessVertex(tess, point, point)
    GLU.gluTessEndContour(tess)

    # then handle each of the holes, if applicable
    if holes != []:
        for hole in holes:
            GLU.gluTessBeginContour(tess)
            for point in hole:
                point3d = (point[0], point[1], 0)
                GLU.gluTessVertex(tess, point3d, point3d)
            GLU.gluTessEndContour(tess)

    GLU.gluTessEndPolygon(tess)
    GLU.gluDeleteTess(tess)
    return np.array(triangle_vertices)
