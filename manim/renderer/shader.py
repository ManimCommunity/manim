from pathlib import Path
import numpy as np
import moderngl
from ..utils import opengl
from .. import config

SHADER_FOLDER = Path(__file__).parent / "shaders"
shader_program_cache = {}


class Shader:
    def __init__(self, context, name):
        self.context = context
        self.name = name
        self.shader_program = self.get_shader_program()

    def get_shader_program(self):
        if self.name in shader_program_cache:
            self.shader_program = shader_program_cache[self.name]
        else:
            with open(SHADER_FOLDER / f"{self.name}.vert") as vertex_shader, open(
                SHADER_FOLDER / f"{self.name}.frag"
            ) as fragment_shader:
                return self.context.program(
                    vertex_shader=vertex_shader.read(),
                    fragment_shader=fragment_shader.read(),
                )

    def set_uniform(self, name, value):
        self.shader_program[name] = value


# Return projection of a onto b.
def project_vector(a, b):
    direction = b / np.linalg.norm(b)
    length = np.dot(a, b / np.linalg.norm(b))
    return direction * length


z = 0.01


class QuadShader(Shader):
    def __init__(self, context):
        super().__init__(context, "quad")

    def render(self):
        self.shader_program["uModelViewMatrix"] = opengl.view_matrix()
        self.shader_program[
            "uProjectionMatrix"
        ] = opengl.orthographic_projection_matrix()

        attributes = np.zeros(
            3,
            dtype=[
                ("in_vert", np.float32, (4,)),
                ("in_color", np.float32, (4,)),
                ("in_bezier_p0", np.float32, (4,)),
                ("in_bezier_p1", np.float32, (4,)),
                ("in_bezier_p2", np.float32, (4,)),
            ],
        )
        attributes["in_color"] = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )
        global z
        bezier_control_points = np.array(
            [
                [-1, 1, 0],
                [-1, 0, 0],
                [-2 + z, -1, 0],
            ]
        )
        z += 0.005
        thickness = 0.1
        self.shader_program["u_thickness"] = thickness

        # TODO: Compute this in the vertex shader.
        bezier_midpoint = np.average(bezier_control_points, axis=0)
        midpoint_vectors = bezier_midpoint - bezier_control_points
        scale = np.max(thickness / np.linalg.norm(midpoint_vectors, axis=1)) + 1
        scale_threshold = 5
        if scale < scale_threshold:
            attributes["in_vert"] = np.hstack(
                (bezier_midpoint - scale * midpoint_vectors, np.ones((3, 1)))
            )
        else:
            line_almost = np.array(bezier_control_points[2] - bezier_control_points[0])
            raise NotImplementedError("Points are too close to co-linear")

        attributes["in_bezier_p0"] = np.hstack(
            (np.tile(bezier_control_points[0], (3, 1)), np.ones((3, 1)))
        )
        attributes["in_bezier_p1"] = np.hstack(
            (np.tile(bezier_control_points[1], (3, 1)), np.ones((3, 1)))
        )
        attributes["in_bezier_p2"] = np.hstack(
            (np.tile(bezier_control_points[2], (3, 1)), np.ones((3, 1)))
        )

        vertex_buffer_object = self.context.buffer(attributes.tobytes())
        vertex_array_object = self.context.simple_vertex_array(
            self.shader_program,
            vertex_buffer_object,
            "in_vert",
            "in_color",
            "in_bezier_p0",
            "in_bezier_p1",
            "in_bezier_p2",
        )

        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()


class CurveShader(Shader):
    def __init__(self, context):
        super().__init__(context, "curve")

    def render(self):
        num_vertices = 6
        attributes = np.zeros(
            num_vertices,
            dtype=[
                ("in_vert", np.float32, (2,)),
                ("in_color", np.float32, (4,)),
                ("in_texture_coord", np.float32, (3,)),
            ],
        )
        attributes["in_vert"] = np.array(
            [
                [-1, -1 + 1],
                [0, 0 + 1],
                [1, -1 + 1],
                [-1, -1],
                [0, 0],
                [1, -1],
            ]
        )
        attributes["in_texture_coord"] = np.array(
            [
                [0, 0, 1],
                [0.5, 0, 1],
                [1, 1, 1],
                [0, 0, 0],
                [0.5, 0, 0],
                [1, 1, 0],
            ]
        )
        attributes["in_color"] = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )

        vertex_buffer_object = self.context.buffer(attributes.tobytes())
        vertex_array_object = self.context.simple_vertex_array(
            self.shader_program,
            vertex_buffer_object,
            "in_vert",
            "in_color",
            "in_texture_coord",
        )

        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()


class ManimCoordsShader(Shader):
    def __init__(self, context):
        super().__init__(context, "manim_coords")

    def render(self):
        self.shader_program["u_model_view_matrix"] = opengl.view_matrix()
        self.shader_program[
            "u_projection_matrix"
        ] = opengl.orthographic_projection_matrix()

        attributes = np.zeros(
            3,
            dtype=[
                ("in_vert", np.float32, (4,)),
            ],
        )
        attributes["in_vert"] = np.array(
            [
                [-1, -1, 0, 1],
                [0, 0, 0, 1],
                [1, -1, 0, 1],
            ]
        )
        self.shader_program["u_color"] = (1.0, 0.0, 1.0, 1.0)

        vertex_buffer_object = self.context.buffer(attributes.tobytes())
        vertex_array_object = self.context.simple_vertex_array(
            self.shader_program,
            vertex_buffer_object,
            "in_vert",
        )

        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()


class TestShader(Shader):
    def __init__(self, context):
        super().__init__(context, "test")

    def render(self):
        num_vertices = 3
        attributes = np.zeros(
            3, dtype=[("in_vert", np.float32, (2,)), ("in_color", np.float32, (4,))]
        )
        attributes["in_vert"] = np.array(
            [
                [-1, -1],
                [0, 0],
                [1, -1],
            ]
        )
        attributes["in_color"] = np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )

        vertex_buffer_object = self.context.buffer(attributes.tobytes())
        vertex_array_object = self.context.simple_vertex_array(
            self.shader_program, vertex_buffer_object, "in_vert", "in_color"
        )

        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()


class LineShader(Shader):
    # All LineShader instances share a single lookup table.
    quadratic_coefficient_table = np.array([])

    def __init__(self, context, curve_detail=20):
        if self.quadratic_coefficient_table.size == 0:
            t = np.linspace(1, 0, curve_detail + 1)
            self.quadratic_coefficient_table = np.stack(
                (t ** 2, 2 * t * (1 - t), (1 - t) ** 2)
            ).T
        super().__init__(context, "test")

    def render(self, mobject):
        self.vertices = self.get_polyline_points(mobject)
        self.process_vertices()
        # self.draw_stroke(mobject)
        # self.draw_stroke_2(mobject)
        # self.draw_stroke_3(mobject)
        self.draw_stroke_4()
        self.draw_fill(mobject)

    def get_polyline_points(self, mobject):
        points = mobject.data["points"]
        return np.concatenate(
            (
                np.atleast_2d(points[0]),
                np.vstack(
                    np.matmul(
                        self.quadratic_coefficient_table,
                        np.vsplit(points, points.shape[0] / 3),
                    )
                ),
            )
        )

    def process_vertices(self):
        self.edges = self.calculate_edges(self.vertices)
        self.line_normals, self.line_vertices = self.edges_to_vertices(
            self.vertices, self.edges
        )
        self.triangle_vertices = opengl.triangulate(self.vertices)

    def edges_to_vertices(self, vertices, edges):
        begin = vertices[edges[:, 0]]
        end = vertices[edges[:, 1]]
        direction = end - begin

        # Normalize.
        l2_norm = np.linalg.norm(direction, axis=1)
        l2_norm[l2_norm == 0] = 1
        direction /= np.atleast_2d(l2_norm).T

        direction_add = np.hstack((direction, np.ones((direction.shape[0], 1))))
        direction_sub = np.hstack((direction, -np.ones((direction.shape[0], 1))))
        # Creates a 3d stack of 2D matrices of the form:
        # [[direction_add, direction_sub], ...]
        # where direction_add and direction_sub are 1D arrays.
        direction_arrays = np.transpose(
            np.stack((direction_add, direction_sub)), axes=(1, 0, 2)
        )
        direction_selection_matrix = np.array(
            [[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]]
        )
        line_normals = np.vstack(
            np.matmul(direction_selection_matrix, direction_arrays)
        )

        # Creates a stack of matrices of the form:
        # [[begin, end], ...]
        # where being and end are 1D arrays.
        endpoint_arrays = np.transpose(np.stack((begin, end)), axes=(1, 0, 2))
        endpoint_selection_matrix = np.array(
            [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]
        )
        line_vertices = np.vstack(np.matmul(endpoint_selection_matrix, endpoint_arrays))

        return line_normals, line_vertices

    def calculate_edges(self, vertices):
        start_indices = np.arange(0, vertices.shape[0] - 1)
        return np.vstack((start_indices, start_indices + 1)).T

    def draw_stroke_3(self, mobject):
        self.shader_program["uMaterialColor"] = (0, 0, 1, 1)
        self.shader_program["uStrokeWeight"] = 6
        self.shader_program["uModelViewMatrix"] = opengl.view_matrix()
        self.shader_program[
            "uProjectionMatrix"
        ] = opengl.orthographic_projection_matrix()
        self.shader_program["uViewport"] = tuple(
            np.array([0, 0, 1420, 800], dtype=np.float32)
        )

        line_attributes = np.zeros(
            self.line_vertices.shape[0],
            dtype=[
                ("aPosition", np.float32, (4,)),
                ("aDirection", np.float32, (4,)),
            ],
        )
        line_attributes["aPosition"] = np.hstack(
            (self.line_vertices, np.ones((252, 1)))
        )
        line_attributes["aDirection"] = self.line_normals
        vertex_buffer_object = self.context.buffer(line_attributes.tobytes())
        vertex_attribute_format = moderngl.detect_format(
            self.shader_program, ("aPosition", "aDirection")
        )
        vertex_array_object = self.context.vertex_array(
            program=self.shader_program,
            content=[
                (
                    vertex_buffer_object,
                    vertex_attribute_format,
                    "aPosition",
                    "aDirection",
                )
            ],
            index_buffer=None,
        )
        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()

    def draw_stroke_4(self):
        num_vertices = 3
        attributes = np.zeros(
            3, dtype=[("in_vert", np.float32, (2,)), ("in_color", np.float32, (4,))]
        )
        attributes["in_vert"] = np.array(
            [
                [-1, -1],
                [0, 0],
                [1, -1],
            ]
        )
        attributes["in_color"] = np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )

        vertex_buffer_object = self.context.buffer(attributes.tobytes())
        vertex_array_object = self.context.simple_vertex_array(
            self.shader_program, vertex_buffer_object, "in_vert", "in_color"
        )

        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()

    def draw_stroke_2(self, mobject):
        # Draws a triangle using a proper view matrix and projection matrix.
        self.shader_program["uMaterialColor"] = (0, 0, 1, 1)
        self.shader_program["uModelViewMatrix"] = opengl.view_matrix()
        self.shader_program[
            "uProjectionMatrix"
        ] = opengl.orthographic_projection_matrix()
        num_vertices = 3
        line_attributes = np.zeros(
            num_vertices, dtype=[("aPosition", np.float32, (4,))]
        )
        line_attributes["aPosition"] = np.array(
            [
                [-1, -1, 0, 1],
                [0, 0, 0, 1],
                [1, -1, 0, 1],
            ]
        )
        # self.line_vertices line_attributes["aDirection"] = self.line_normals
        vertex_buffer_object = self.context.buffer(line_attributes.tobytes())
        vertex_attribute_format = moderngl.detect_format(
            self.shader_program, ("aPosition",)
        )
        vertex_array_object = self.context.vertex_array(
            program=self.shader_program,
            content=[
                (
                    vertex_buffer_object,
                    vertex_attribute_format,
                    "aPosition",
                )
            ],
            index_buffer=None,
        )
        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()

    def draw_stroke(self, mobject):
        # Draws a random glitchy shape using p5 shaders.
        # Set uniforms
        self.shader_program["uMaterialColor"] = tuple(mobject.data["stroke_rgba"][0])
        self.shader_program["uStrokeWeight"] = mobject.data["stroke_width"][0]
        self.shader_program["uModelViewMatrix"] = tuple(
            np.array(
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -346.41015625, 1],
                dtype=np.float32,
            )
        )
        self.shader_program["uProjectionMatrix"] = tuple(
            np.array(
                [
                    0.9758032560348511,
                    0,
                    0,
                    0,
                    0,
                    -1.7320507764816284,
                    0,
                    0,
                    0,
                    0,
                    -1.0202020406723022,
                    -1,
                    0,
                    0,
                    -69.98184967041016,
                    0,
                ],
                dtype=np.float32,
            )
        )
        self.shader_program["uViewport"] = tuple(
            np.array([0, 0, 1420, 800], dtype=np.float32)
        )
        self.shader_program["uPerspective"] = 1

        line_attributes = np.zeros(
            self.line_vertices.shape[0],
            dtype=[
                ("aPosition", np.float32, (3,)),
                ("aDirection", np.float32, (4,)),
            ],
        )
        line_attributes["aPosition"] = self.line_vertices
        line_attributes["aDirection"] = self.line_normals
        vertex_buffer_object = self.context.buffer(line_attributes.tobytes())
        vertex_attribute_format = moderngl.detect_format(
            self.shader_program, ("aPosition", "aDirection")
        )
        vertex_array_object = self.context.vertex_array(
            program=self.shader_program,
            content=[
                (
                    vertex_buffer_object,
                    vertex_attribute_format,
                    "aPosition",
                    "aDirection",
                )
            ],
            index_buffer=None,
        )
        vertex_array_object.render(moderngl.TRIANGLES)
        vertex_buffer_object.release()
        vertex_array_object.release()

    def draw_fill(self, mobject):
        pass

    # def get_num_segments(self, mobject, error=0.01):
    #     """
    #     For each quadratic bezier curve in mobject, computes the minimum m such that the
    #     maximum distance between the curve and a a polyline approximation of the curve
    #     with m segments is less than error. Taken from
    #     https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1000&context=facpub#section.10.6
    #     """
    #     points = mobject.data["points"]
    #     bezier_curve_points = np.vsplit(points, points.shape[0] / 3)
    #     hodographs = 2 * np.absolute(
    #         np.sum(bezier_curve_points * np.array([[1], [-2], [1]]), axis=1)
    #     )
    #     segments_squared = np.sqrt(np.sum(hodographs ** 2, axis=1)) / (8 * error)
    #     return np.ceil(np.sqrt(segments_squared))
