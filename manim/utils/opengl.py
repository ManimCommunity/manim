import OpenGL.GLU as GLU
import numpy as np
import numpy.linalg as linalg
from .. import config

depth = 20


def matrix_to_shader_input(matrix):
    return tuple(matrix.T.ravel())


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


def perspective_projection_matrix(width=None, height=None, near=4, far=18):
    if width is None:
        width = config["frame_width"] / 3
    if height is None:
        height = config["frame_height"] / 3

    return tuple(
        np.array(
            [
                [2 * near / width, 0, 0, 0],
                [0, 2 * near / height, 0, 0],
                [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0, 0, -1, 0],
            ]
        ).T.ravel()
    )


def translation_matrix(x=0, y=0, z=0):
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


def x_rotation_matrix(x=0):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x), -np.sin(x), 0],
            [0, np.sin(x), np.cos(x), 0],
            [0, 0, 0, 1],
        ]
    )


def y_rotation_matrix(y=0):
    return np.array(
        [
            [np.cos(y), 0, np.sin(y), 0],
            [0, 1, 0, 0],
            [-np.sin(y), 0, np.cos(y), 0],
            [0, 0, 0, 1],
        ]
    )


def z_rotation_matrix(z=0):
    return np.array(
        [
            [np.cos(z), -np.sin(z), 0, 0],
            [np.sin(z), np.cos(z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


# TODO: When rotating around the x axis, rotation eventually stops.
def rotate_in_place_matrix(initial_position, x=0, y=0, z=0):
    return np.matmul(
        translation_matrix(*-initial_position),
        np.matmul(
            rotation_matrix(x, y, z),
            translation_matrix(*initial_position),
        ),
    )


def rotation_matrix(x=0, y=0, z=0):
    return np.matmul(
        np.matmul(x_rotation_matrix(x), y_rotation_matrix(y)), z_rotation_matrix(z)
    )


def scale_matrix(scale_factor=None):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def view_matrix(
    camera_position=None,
    translation=None,
    x_rotation=0,
    y_rotation=0,
    z_rotation=0,
    scale=0,
):
    if translation is None:
        translation = np.array([0, 0, depth / 2 + 1])
    model_matrix = np.matmul(
        np.matmul(
            translation_matrix(*translation),
            rotation_matrix(x=x_rotation, y=y_rotation, z=z_rotation),
        ),
        scale_matrix(),
    )
    return tuple(linalg.inv(model_matrix).T.ravel())


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
