import collections

import numpy as np

from ..constants import *
from ..utils import opengl
from ..utils.space_ops import cross2d, earclip_triangulation
from .shader import Shader


def build_matrix_lists(mob):
    root_hierarchical_matrix = mob.hierarchical_model_matrix()
    matrix_to_mobject_list = collections.defaultdict(list)
    if mob.has_points():
        matrix_to_mobject_list[tuple(root_hierarchical_matrix.ravel())].append(mob)
    mobject_to_hierarchical_matrix = {mob: root_hierarchical_matrix}
    dfs = [mob]
    while dfs:
        parent = dfs.pop()
        for child in parent.submobjects:
            child_hierarchical_matrix = (
                mobject_to_hierarchical_matrix[parent] @ child.model_matrix
            )
            mobject_to_hierarchical_matrix[child] = child_hierarchical_matrix
            if child.has_points():
                matrix_to_mobject_list[tuple(child_hierarchical_matrix.ravel())].append(
                    child,
                )
            dfs.append(child)
    return matrix_to_mobject_list


def render_opengl_vectorized_mobject_fill(renderer, mobject):
    matrix_to_mobject_list = build_matrix_lists(mobject)

    for matrix_tuple, mobject_list in matrix_to_mobject_list.items():
        model_matrix = np.array(matrix_tuple).reshape((4, 4))
        render_mobject_fills_with_matrix(renderer, model_matrix, mobject_list)


def render_mobject_fills_with_matrix(renderer, model_matrix, mobjects):
    # Precompute the total number of vertices for which to reserve space.
    # Note that triangulate_mobject() will cache its results.
    total_size = 0
    for submob in mobjects:
        total_size += triangulate_mobject(submob).shape[0]

    attributes = np.empty(
        total_size,
        dtype=[
            ("in_vert", np.float32, (3,)),
            ("in_color", np.float32, (4,)),
            ("texture_coords", np.float32, (2,)),
            ("texture_mode", np.int32),
        ],
    )

    write_offset = 0
    for submob in mobjects:
        if not submob.has_points():
            continue
        mobject_triangulation = triangulate_mobject(submob)
        end_offset = write_offset + mobject_triangulation.shape[0]
        attributes[write_offset:end_offset] = mobject_triangulation
        attributes["in_color"][write_offset:end_offset] = np.repeat(
            submob.fill_rgba,
            mobject_triangulation.shape[0],
            axis=0,
        )
        write_offset = end_offset

    fill_shader = Shader(renderer.context, name="vectorized_mobject_fill")
    fill_shader.set_uniform(
        "u_model_view_matrix",
        opengl.matrix_to_shader_input(
            renderer.camera.get_view_matrix(format=False) @ model_matrix,
        ),
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
    points = mob.points

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
        ],
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
            ],
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
            np.ones(concave_triangle_indices.shape[0]),
            -1 * np.ones(convex_triangle_indices.shape[0]),
            np.zeros(inner_tri_indices.shape[0]),
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
    matrix_to_mobject_list = build_matrix_lists(mobject)
    for matrix_tuple, mobject_list in matrix_to_mobject_list.items():
        model_matrix = np.array(matrix_tuple).reshape((4, 4))
        render_mobject_strokes_with_matrix(renderer, model_matrix, mobject_list)


def render_mobject_strokes_with_matrix(renderer, model_matrix, mobjects):
    # Precompute the total number of vertices for which to reserve space.
    total_size = 0
    for submob in mobjects:
        total_size += submob.points.shape[0]

    points = np.empty((total_size, 3))
    colors = np.empty((total_size, 4))
    widths = np.empty(total_size)

    write_offset = 0
    for submob in mobjects:
        if not submob.has_points():
            continue
        end_offset = write_offset + submob.points.shape[0]

        points[write_offset:end_offset] = submob.points
        if submob.stroke_rgba.shape[0] == points[write_offset:end_offset].shape[0]:
            colors[write_offset:end_offset] = submob.stroke_rgba
        else:
            colors[write_offset:end_offset] = np.repeat(
                submob.stroke_rgba,
                submob.points.shape[0],
                axis=0,
            )
        widths[write_offset:end_offset] = np.repeat(
            submob.stroke_width,
            submob.points.shape[0],
        )
        write_offset = end_offset

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
    stroke_data["tile_coordinate"] = np.vstack(
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
    )

    shader = Shader(renderer.context, "vectorized_mobject_stroke")
    shader.set_uniform(
        "u_model_view_matrix",
        opengl.matrix_to_shader_input(
            renderer.camera.get_view_matrix(format=False) @ model_matrix,
        ),
    )
    shader.set_uniform("u_projection_matrix", renderer.scene.camera.projection_matrix)
    shader.set_uniform("manim_unit_normal", tuple(-mobjects[0].unit_normal[0]))

    vbo = renderer.context.buffer(stroke_data.tobytes())
    vao = renderer.context.simple_vertex_array(
        shader.shader_program, vbo, *stroke_data.dtype.names
    )
    renderer.frame_buffer_object.use()
    vao.render()
    vao.release()
    vbo.release()
