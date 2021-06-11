import moderngl
import numpy as np

from .shader import Shader


def render_opengl_vectorized_mobject_fill(renderer, mobject):
    # This algorithm is taken from
    # https://medium.com/@evanwallace/easy-scalable-text-rendering-on-the-gpu-c3f4d782c5ac.
    curves_list = []
    for submob in mobject.family_members_with_points():
        subpaths = submob.get_subpaths()
        for i in range(len(subpaths)):
            subpaths[i] = np.concatenate(
                (subpaths[i], np.linspace(subpaths[i][0], subpaths[i][-1], 3))
            )
        curves_list.append(np.concatenate(subpaths))

    non_overlapping_layers = find_non_overlapping_layers(curves_list)

    textures = []
    texture_fbos = []
    renderer.context.blend_func = renderer.context.ADDITIVE_BLENDING
    for layer in non_overlapping_layers:
        vao, vbo, ibo = layer_to_stencil_vao(
            renderer.context,
            layer,
            uniforms={
                "u_view_matrix": renderer.scene.camera.get_view_matrix(),
                "u_projection_matrix": renderer.scene.camera.projection_matrix,
            },
        )

        # Render the vertex array object to the texture.
        # TODO: Reuse the same texture fbos between renders.
        texture = renderer.context.texture(renderer.get_pixel_shape(), 3)
        texture_fbo = renderer.context.framebuffer(texture)
        texture_fbo.use()

        vao.render()
        vao.release()
        vbo.release()
        ibo.release()

        textures.append(texture)
        texture_fbos.append(texture_fbo)

    renderer.context.blend_func = (
        moderngl.SRC_ALPHA,
        moderngl.ONE_MINUS_SRC_ALPHA,
        moderngl.ONE,
        moderngl.ONE,
    )
    vao, vbo = textures_to_fill_vao(renderer.context, textures, mobject)
    renderer.frame_buffer_object.use()
    vao.render()
    vao.release()
    vbo.release()
    for texture in textures:
        texture.release()
    for texture_fbo in texture_fbos:
        texture_fbo.release()


def find_non_overlapping_layers(curves_list):
    def get_bounding_box(curves):
        # TODO: Convert to normalized device coordinates.
        lower_left = curves.min(0)
        upper_right = curves.max(0)
        return [lower_left, upper_right]

    # TODO: Use precomputed bounding box.
    bounding_boxes = [get_bounding_box(curves) for curves in curves_list]

    def boxes_intersect(box1, box2):
        no_intersetion = (
            (box1[1][0] < box2[0][0])
            or (box2[1][0] < box1[0][0])
            or (box1[1][1] < box2[0][1])
            or (box2[1][1] < box1[0][1])
        )
        return not no_intersetion

    def box_intersects_layer(box, lst):
        return any(boxes_intersect(box, box2) for box2 in lst)

    # This should probably be replaced with a sweep and prune algorithm.
    layers = []
    for i in range(len(bounding_boxes)):
        box = bounding_boxes[i]
        added_to_list = False
        for j in range(len(layers)):
            if not box_intersects_layer(box, layers[j]):
                layers[j].append(curves_list[i])
                added_to_list = True
        if not added_to_list:
            layers.append([curves_list[i]])
    return layers


def layer_to_stencil_vao(context, layer, uniforms=None):
    input_vertices = np.concatenate(tuple(layer))

    # Add stencil-then-fill data.
    negative_indices = (
        np.ones(input_vertices.shape[0] // 3, dtype=np.int32)
        + np.arange(input_vertices.shape[0] // 3) * 3
    )
    endpoints = np.delete(input_vertices, negative_indices, axis=0)
    stencil_fill_vertices = np.concatenate((np.array([[0.0, 0.0, 0.0]]), endpoints))

    indices = np.arange(1, stencil_fill_vertices.shape[0], dtype=np.int32)
    indices = np.insert(
        indices, np.arange(endpoints.shape[0] // 2, dtype=np.int32) * 2, 0
    )

    texture_coords = np.repeat([[0, 1]], stencil_fill_vertices.shape[0], axis=0)

    # Add Loop-Blinn data.
    vertices = np.concatenate((stencil_fill_vertices, input_vertices))
    texture_coords = np.concatenate(
        (
            texture_coords,
            np.tile(
                [
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [1.0, 1.0],
                ],
                (input_vertices.shape[0] // 3, 1),
            ),
        ),
    )
    indices = np.concatenate(
        (
            indices,
            np.arange(
                stencil_fill_vertices.shape[0],
                stencil_fill_vertices.shape[0] + input_vertices.shape[0],
            ),
        ),
        dtype=np.int32,
    )

    vertex_data = np.zeros(
        vertices.shape[0],
        dtype=[("in_vert", np.float32, (3,)), ("texture_coords", np.float32, (2,))],
    )
    vertex_data["in_vert"] = vertices
    vertex_data["texture_coords"] = texture_coords

    stencil_shader = Shader(context, name="vectorized_mobject_stencil")
    for uniform, value in uniforms.items():
        stencil_shader.set_uniform(uniform, value)
    vbo = context.buffer(vertex_data.tobytes())
    ibo = context.buffer(indices)
    vao = context.simple_vertex_array(
        stencil_shader.shader_program,
        vbo,
        *vertex_data.dtype.names,
        index_buffer=ibo,
    )
    return vao, vbo, ibo


def textures_to_fill_vao(context, textures, mobject):
    vertices = np.zeros(
        len(textures) * 6,
        dtype=[
            ("in_vert", np.float32, (2,)),
            ("in_texcoord_0", np.float32, (2,)),
            ("in_texindex", np.float32, (1,)),
        ],
    )
    vertices["in_vert"] = np.tile(
        [
            # Upper left half of the screen.
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            # Lower right half of the screen.
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        (len(textures), 1),
    )
    vertices["in_texcoord_0"] = np.tile(
        [
            # Upper left half of the texture.
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            # Lower right half of the texture.
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        (len(textures), 1),
    )
    vertices["in_texindex"] = np.repeat(
        np.atleast_2d(np.arange(len(textures))).transpose(),
        6,
        axis=0,
    )
    texture_shader = Shader(context, name="vectorized_mobject_fill")
    texture_shader.set_uniform("color", tuple(mobject.data["fill_rgba"][0]))
    for i in range(len(textures)):
        texture_shader.set_uniform(f"Texture{i}", i)
        textures[i].use(location=i)

    vbo = context.buffer(vertices.tobytes())
    vao = context.simple_vertex_array(
        texture_shader.shader_program,
        vbo,
        *vertices.dtype.names,
    )
    return vao, vbo


def render_opengl_vectorized_mobject_stroke(renderer, mobject):
    shader = Shader(renderer.context, "vectorized_mobject_stroke")
    shader.set_uniform("u_color", (0.0, 1.0, 0.0, 1.0))

    shader.set_uniform("u_model_view_matrix", renderer.camera.get_view_matrix())
    shader.set_uniform(
        "u_projection_matrix",
        renderer.scene.camera.projection_matrix,
    )

    points = np.empty((0, 3))
    for submob in mobject.family_members_with_points():
        points = np.append(points, submob.data["points"], axis=0)

    # points = mobject[0].data["points"][-15:-9]
    # points = mobject[0].data["points"]
    # points = mobject.data["points"]
    stroke_data = np.zeros(
        len(points),
        dtype=[
            # ("point", np.float32, (3,)),
            ("previous_curve", np.float32, (3, 3)),
            ("current_curve", np.float32, (3, 3)),
            ("next_curve", np.float32, (3, 3)),
            ("tile_coordinate", np.float32, (2,)),
        ],
    )

    # stroke_data["point"] = points
    curves = np.reshape(points, (-1, 3, 3))
    stroke_data["previous_curve"] = np.repeat(np.roll(curves, 1, axis=0), 3, axis=0)
    stroke_data["current_curve"] = np.repeat(curves, 3, axis=0)
    stroke_data["next_curve"] = np.repeat(np.roll(curves, -1, axis=0), 3, axis=0)

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
    # import ipdb

    # ipdb.set_trace(context=9)
    # import time

    # time.sleep(0.2)

    shader.set_uniform("color", tuple(mobject.data["stroke_rgba"][0]))
    shader.set_uniform("stroke_width", mobject.data["stroke_width"])
    shader.set_uniform("manim_unit_normal", tuple(-mobject.data["unit_normal"][0]))

    vbo = renderer.context.buffer(stroke_data.tobytes())
    vao = renderer.context.simple_vertex_array(
        shader.shader_program,
        vbo,
        *stroke_data.dtype.names,
    )

    renderer.frame_buffer_object.use()
    vao.render()
    vao.release()
    vbo.release()
