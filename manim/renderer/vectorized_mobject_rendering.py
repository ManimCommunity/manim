import numpy as np

from ..constants import *
from ..mobject.types.opengl_vectorized_mobject import OpenGLVMobject
from ..utils.space_ops import cross2d, earclip_triangulation, z_to_vector
from .shader import Shader


def render_opengl_vectorized_mobject_fill(renderer, mobject):
    attributes = np.empty(
        0,
        dtype=[
            ("in_vert", np.float32, (3,)),
            ("in_color", np.float32, (4,)),
            ("texture_coords", np.float32, (2,)),
            ("texture_mode", np.int32),
        ],
    )
    color = np.empty((0, 4), dtype=np.float32)
    for submob in mobject.family_members_with_points():
        mobject_triangulation = triangulate_mobject(submob)
        mobject_color = np.repeat(
            submob.data["fill_rgba"], mobject_triangulation.shape[0], axis=0
        )
        attributes = np.append(attributes, mobject_triangulation)
        color = np.append(color, mobject_color, axis=0)
    attributes["in_color"] = color

    fill_shader = Shader(
        renderer.context,
        name="vectorized_mobject_fill",
    )
    fill_shader.set_uniform(
        "u_view_matrix",
        renderer.camera.get_view_matrix(),
    )
    fill_shader.set_uniform(
        "u_projection_matrix",
        renderer.scene.camera.projection_matrix,
    )

    vbo = renderer.context.buffer(attributes.tobytes())
    vao = renderer.context.simple_vertex_array(
        fill_shader.shader_program,
        vbo,
        *attributes.dtype.names,
    )
    vao.render()
    vao.release()
    vbo.release()


def triangulate_mobject(mob):
    if not mob.needs_new_triangulation:
        return mob.triangulation

    # Figure out how to triangulate the interior to know
    # how to send the points as to the vertex shader.
    # First triangles come directly from the points
    # normal_vector = mob.get_unit_normal()
    points = mob.get_points()

    b0s = points[0::3]
    b1s = points[1::3]
    b2s = points[2::3]
    v01s = b1s - b0s
    v12s = b2s - b1s

    crosses = cross2d(v01s, v12s)
    convexities = np.sign(crosses)
    if mob.orientation == 1:
        concave_parts = convexities > 0
        convex_parts = convexities <= 0
    else:
        concave_parts = convexities < 0
        convex_parts = convexities >= 0

    # These are the vertices to which we'll apply a polygon triangulation
    atol = mob.tolerance_for_point_equality
    end_of_loop = np.zeros(len(b0s), dtype=bool)
    end_of_loop[:-1] = (np.abs(b2s[:-1] - b0s[1:]) > atol).any(1)
    end_of_loop[-1] = True

    indices = np.arange(len(points), dtype=int)
    inner_vert_indices = np.hstack(
        [
            indices[0::3],
            indices[1::3][concave_parts],
            indices[2::3][end_of_loop],
        ]
    )
    inner_vert_indices.sort()
    rings = np.arange(1, len(inner_vert_indices) + 1)[inner_vert_indices % 3 == 2]

    # Triangulate
    inner_verts = points[inner_vert_indices]
    inner_tri_indices = inner_vert_indices[earclip_triangulation(inner_verts, rings)]

    bezier_triangle_indices = np.reshape(indices, (-1, 3))
    concave_triangle_indices = np.reshape(bezier_triangle_indices[concave_parts], (-1))
    convex_triangle_indices = np.reshape(bezier_triangle_indices[convex_parts], (-1))

    points = points[
        np.hstack(
            [
                concave_triangle_indices,
                convex_triangle_indices,
                inner_tri_indices,
            ]
        )
    ]
    texture_coords = np.tile(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 1.0],
        ],
        (points.shape[0] // 3, 1),
    )
    texture_mode = np.hstack(
        (
            np.ones((concave_triangle_indices.shape[0])),
            -1 * np.ones((convex_triangle_indices.shape[0])),
            np.zeros((inner_tri_indices.shape[0])),
        ),
    )

    attributes = np.zeros(
        points.shape[0],
        dtype=[
            ("in_vert", np.float32, (3,)),
            ("in_color", np.float32, (4,)),
            ("texture_coords", np.float32, (2,)),
            ("texture_mode", np.int32),
        ],
    )
    attributes["in_vert"] = points
    attributes["texture_coords"] = texture_coords
    attributes["texture_mode"] = texture_mode

    mob.triangulation = attributes
    mob.needs_new_triangulation = False

    return attributes


def render_opengl_vectorized_mobject_stroke(renderer, mobject):
    shader = Shader(renderer.context, "vectorized_mobject_stroke")

    shader.set_uniform("u_model_view_matrix", renderer.camera.get_view_matrix())
    shader.set_uniform(
        "u_projection_matrix",
        renderer.scene.camera.projection_matrix,
    )

    points = np.empty((0, 3))
    colors = np.empty((0, 4))
    widths = np.empty((0))
    for submob in mobject.family_members_with_points():
        points = np.append(points, submob.data["points"], axis=0)
        colors = np.append(
            colors,
            np.repeat(
                submob.data["stroke_rgba"], submob.data["points"].shape[0], axis=0
            ),
            axis=0,
        )
        widths = np.append(
            widths,
            np.repeat(submob.data["stroke_width"], submob.data["points"].shape[0]),
        )

    stroke_data = np.zeros(
        len(points),
        dtype=[
            # ("previous_curve", np.float32, (3, 3)),
            ("current_curve", np.float32, (3, 3)),
            # ("next_curve", np.float32, (3, 3)),
            ("tile_coordinate", np.float32, (2,)),
            ("in_color", np.float32, (4,)),
            ("in_width", np.float32),
        ],
    )

    stroke_data["in_color"] = colors
    stroke_data["in_width"] = widths
    curves = np.reshape(points, (-1, 3, 3))
    # stroke_data["previous_curve"] = np.repeat(np.roll(curves, 1, axis=0), 3, axis=0)
    stroke_data["current_curve"] = np.repeat(curves, 3, axis=0)
    # stroke_data["next_curve"] = np.repeat(np.roll(curves, -1, axis=0), 3, axis=0)

    # Repeat each vertex in order to make a tile.
    stroke_data = np.tile(stroke_data, 2)
    stroke_data["tile_coordinate"] = np.concatenate(
        (
            np.tile(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                (len(points) // 3, 1),
            ),
            np.tile(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ],
                (len(points) // 3, 1),
            ),
        ),
        axis=0,
    )

    shader.set_uniform("color", tuple(mobject.data["stroke_rgba"][0]))
    shader.set_uniform("manim_unit_normal", tuple(-mobject.data["unit_normal"][0]))

    vbo = renderer.context.buffer(stroke_data.tobytes())
    vao = renderer.context.simple_vertex_array(
        shader.shader_program, vbo, *stroke_data.dtype.names
    )
    renderer.frame_buffer_object.use()
    vao.render()
    vao.release()
    vbo.release()
