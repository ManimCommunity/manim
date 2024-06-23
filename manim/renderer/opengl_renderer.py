from __future__ import annotations

import moderngl as gl
import numpy as np

import manim.constants as const
import manim.utils.color.core as c
import manim.utils.color.manim_colors as color
from manim import config, logger
from manim.camera.camera import Camera
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.renderer.buffers.buffer import STD140BufferFormat
from manim.renderer.opengl_shader_program import load_shader_program_by_folder
from manim.renderer.renderer import ImageType, Renderer, RendererData, RendererProtocol
from manim.utils.iterables import listify
from manim.utils.space_ops import cross2d, earclip_triangulation, z_to_vector

ubo_camera = STD140BufferFormat(
    "ubo_camera",
    (
        ("vec2", "frame_shape"),
        ("vec3", "camera_center"),
        ("mat3", "camera_rotation"),
        ("float", "focal_distance"),
        ("float", "is_fixed_in_frame"),
        ("float", "is_fixed_orientation"),
        ("vec3", "fixed_orientation_center"),
    ),
)

fill_dtype = [
    ("point", np.float32, (3,)),
    ("unit_normal", np.float32, (3,)),
    ("color", np.float32, (4,)),
    ("vert_index", np.float32, (1,)),
]
stroke_dtype = [
    ("point", np.float32, (3,)),
    ("prev_point", np.float32, (3,)),
    ("next_point", np.float32, (3,)),
    ("stroke_width", np.float32, (1,)),
    ("color", np.float32, (4,)),
]
frame_dtype = [("pos", np.float32, (2,)), ("uv", np.float32, (2,))]


class GLRenderData(RendererData):
    def __init__(self) -> None:
        super().__init__()
        self.fill_rgbas = np.zeros((1, 4))
        self.stroke_rgbas = np.zeros((1, 4))
        self.stroke_widths = np.zeros((1, 1))
        self.normals = np.zeros((1, 4))
        self.orientation = np.zeros((1, 1))
        self.vert_indices = np.zeros((0, 3))
        self.bounding_box = np.zeros((3, 3))

    def __repr__(self) -> str:
        return f"""GLRenderData
fill:
{self.fill_rgbas}
stroke:
{self.stroke_rgbas}
normals:
{self.normals}
orientation:
{self.orientation}
mesh:
{self.vert_indices}
bounding_box:
{self.bounding_box}
        """


# TODO: Move into GLVMobjectManager
def get_triangulation(self: OpenGLVMobject, normal_vector=None):
    # Figure out how to triangulate the interior to know
    # how to send the points as to the vertex shader.
    # First triangles come directly from the points
    if normal_vector is None:
        normal_vector = self.get_unit_normal()

    points = self.points

    if len(points) <= 1:
        self.triangulation = np.zeros(0, dtype="i4")
        self.needs_new_triangulation = False
        return self.triangulation

    if not np.isclose(normal_vector, const.OUT).all():
        # Rotate points such that unit normal vector is OUT
        points = np.dot(points, z_to_vector(normal_vector))
    indices = np.arange(len(points), dtype=int)

    b0s = points[0::3]
    b1s = points[1::3]
    b2s = points[2::3]
    v01s = b1s - b0s
    v12s = b2s - b1s

    crosses = cross2d(v01s, v12s)
    convexities = np.sign(crosses)

    atol = self.tolerance_for_point_equality
    end_of_loop = np.zeros(len(b0s), dtype=bool)
    end_of_loop[:-1] = (np.abs(b2s[:-1] - b0s[1:]) > atol).any(1)
    end_of_loop[-1] = True

    concave_parts = convexities < 0

    # These are the vertices to which we'll apply a polygon triangulation
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

    tri_indices = np.hstack([indices, inner_tri_indices])
    self.triangulation = tri_indices
    self.needs_new_triangulation = False
    return tri_indices


def prepare_array(values: np.ndarray, desired_length: int):
    """Interpolates a given list of colors to match the desired length

    Parameters
    ----------
    values : np.ndarray
        a 2 dimensional numpy array where values are interpolated on the y axis
    desired_length : int
        the desired length for the array

    Returns
    -------
    np.ndarray
        the interpolated array of values
    """
    fill_length = len(values)
    if fill_length == 1:
        return np.repeat(values, desired_length, axis=0)
    xm = np.linspace(0, fill_length - 1, desired_length)
    rgbas = []
    for x in xm:
        minimum = int(np.floor(x))
        maximum = int(np.ceil(x))
        alpha = x - minimum
        if alpha == 0:
            rgbas.append(values[minimum])
            continue

        val_a = values[minimum]
        val_b = values[maximum]
        rgbas.append(val_a * (1 - alpha) + val_b * alpha)
    return np.array(rgbas)


# TODO: Move into GLVMobjectManager
def compute_bounding_box(mob):
    all_points = np.vstack(
        [
            mob.points,
            *(m.get_bounding_box() for m in mob.get_family()[1:] if m.has_points()),
        ],
    )
    if len(all_points) == 0:
        return np.zeros((3, mob.dim))
    else:
        # Lower left and upper right corners
        mins = all_points.min(0)
        maxs = all_points.max(0)
        mids = (mins + maxs) / 2
        return np.array([mins, mids, maxs])


class ProgramManager:
    @staticmethod
    def get_available_uniforms(prog):
        names = []
        for name in prog:
            member = prog[name]
            if isinstance(member, gl.Uniform):
                names.append(name)

    @staticmethod
    def write_uniforms(prog, uniforms):
        for name in prog:
            member = prog[name]
            if isinstance(member, gl.Uniform):
                if name in uniforms:
                    member.value = uniforms[name]

    @staticmethod
    def bind_to_uniform_block(uniform_buffer_object: gl.Buffer, idx: int = 0):
        uniform_buffer_object.bind_to_uniform_block(idx)


class OpenGLRenderer(Renderer, RendererProtocol):
    pixel_array_dtype = np.uint8

    def __init__(
        self,
        pixel_width: int = config.pixel_width,
        pixel_height: int = config.pixel_height,
        samples: int = 4,
        background_color: c.ManimColor = color.BLACK,
        background_opacity: float = 1.0,
        background_image: str | None = None,
        substitute_output_fbo: gl.Framebuffer | None = None,
    ) -> None:
        super().__init__()
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.samples = samples
        self.background_color = background_color.to_rgba()
        self.background_image = background_image
        self.rgb_max_val: float = np.iinfo(self.pixel_array_dtype).max

        # Initializing Context
        logger.debug("Initializing OpenGL context and framebuffers")
        self.ctx: gl.Context = gl.create_context()

        # Those are the actual buffers that are used for rendering
        self.stencil_texture = self.ctx.texture(
            (self.pixel_width, self.pixel_height), components=1, samples=0, dtype="f1"
        )
        self.render_target_texture = self.ctx.texture(
            (self.pixel_width, self.pixel_height), components=4, samples=0, dtype="f1"
        )
        self.stencil_buffer = self.ctx.renderbuffer(
            (self.pixel_width, self.pixel_height),
            components=1,
            samples=samples,
            dtype="f1",
        )
        self.color_buffer = self.ctx.renderbuffer(
            (self.pixel_width, self.pixel_height),
            components=4,
            samples=samples,
            dtype="f1",
        )
        self.depth_buffer = self.ctx.depth_renderbuffer(
            (self.pixel_width, self.pixel_height), samples=samples
        )

        # Here we create different fbos that can be reused which are basically just targets to use for rendering and copy
        # render_target_fbo is used for rendering it can write to color and stencil
        self.render_target_fbo = self.ctx.framebuffer(
            color_attachments=[self.color_buffer, self.stencil_buffer],
            depth_attachment=self.depth_buffer,
        )

        # this is used as source for stencil copy
        self.stencil_buffer_fbo = self.ctx.framebuffer(
            color_attachments=[self.stencil_buffer]
        )
        # this is used as destination for stencil copy
        self.stencil_texture_fbo = self.ctx.framebuffer(
            color_attachments=[self.stencil_texture]
        )
        # this is used as source for copying color to the output
        self.color_buffer_fbo = self.ctx.framebuffer(
            color_attachments=[self.color_buffer]
        )

        # this is used as destination for copying the rendered target
        # and using it as texture on the output_fbo
        self.render_target_texture_fbo = self.ctx.framebuffer(
            color_attachments=[self.render_target_texture]
        )
        self.output_fbo = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.renderbuffer(
                    (self.pixel_width, self.pixel_height), dtype="f1", components=4
                ),
            ]
        )

        # Preparing vmobject shader
        logger.debug("Initializing Shader Programs")
        self.vmobject_fill_program = load_shader_program_by_folder(
            self.ctx, "quadratic_bezier_fill"
        )
        self.vmobject_stroke_program = load_shader_program_by_folder(
            self.ctx, "quadratic_bezier_stroke"
        )
        self.render_texture_program = load_shader_program_by_folder(
            self.ctx, "render_texture"
        )

    def use_window(self):
        self.output_fbo.release()
        self.output_fbo = self.ctx.detect_framebuffer()

    # TODO this should also be done with the update decorators because if the camera doesn't change this is pretty rough
    def init_camera(self, camera: Camera):
        camera_data = {
            "frame_shape": (config.frame_width, config.frame_height),
            "camera_center": camera.get_center(),
            "camera_rotation": camera.get_inverse_camera_rotation_matrix().T,
            "focal_distance": camera.get_focal_distance(),
            "is_fixed_in_frame": 0.0,
            "is_fixed_orientation": 0.0,
            "fixed_orientation_center": np.array([0.0, 0.0, 0.0]),
        }
        ubo_camera.write(camera_data)

        uniforms = {}
        uniforms["anti_alias_width"] = 0.01977
        uniforms["light_source_position"] = (-10, 10, 10)
        uniforms["pixel_shape"] = (self.pixel_width, self.pixel_height)

        buffer = self.ctx.buffer(ubo_camera.data)
        # TODO: convert to singular 4x4 matrix after getting *something* to render
        # self.vmobject_fill_program['view'].value = camera.get_view()?
        ProgramManager.bind_to_uniform_block(buffer)
        ProgramManager.write_uniforms(self.vmobject_fill_program, uniforms)
        ProgramManager.write_uniforms(self.vmobject_stroke_program, uniforms)

    # TODO: Move into GLVMobjectManager
    def get_stroke_shader_data(self, mob: OpenGLVMobject) -> np.ndarray:
        if not isinstance(mob.renderer_data, GLRenderData):
            raise TypeError()

        points = mob.points
        stroke_data = np.zeros(len(points), dtype=stroke_dtype)

        nppc = mob.n_points_per_curve
        stroke_data["point"] = points
        stroke_data["prev_point"][:nppc] = points[-nppc:]
        stroke_data["prev_point"][nppc:] = points[:-nppc]
        stroke_data["next_point"][:-nppc] = points[nppc:]
        stroke_data["next_point"][-nppc:] = points[:nppc]
        stroke_data["color"] = mob.renderer_data.stroke_rgbas
        stroke_data["stroke_width"] = mob.renderer_data.stroke_widths.reshape((-1, 1))

        return stroke_data

    # TODO: Move into GLVMobjectManager
    def get_fill_shader_data(self, mob: OpenGLVMobject) -> np.ndarray:
        if not isinstance(mob.renderer_data, GLRenderData):
            raise TypeError()

        fill_data = np.zeros(len(mob.points), dtype=fill_dtype)
        fill_data["point"] = mob.points
        fill_data["color"] = mob.renderer_data.fill_rgbas
        # fill_data["orientation"] = mob.renderer_data.orientation
        fill_data["unit_normal"] = mob.renderer_data.normals
        fill_data["vert_index"] = np.reshape(range(len(mob.points)), (-1, 1))
        return fill_data

    def pre_render(self, camera):
        self.init_camera(camera=camera)
        self.ctx.clear()
        self.render_target_fbo.use()
        self.render_target_fbo.clear(*self.background_color)

    def post_render(self):
        frame_data = np.zeros(6, dtype=frame_dtype)
        frame_data["pos"] = np.array(
            [[-1, -1], [-1, 1], [1, -1], [1, -1], [-1, 1], [1, 1]]
        )
        frame_data["uv"] = np.array([[0, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 1]])
        vbo = self.ctx.buffer(frame_data.tobytes())
        format = gl.detect_format(self.render_texture_program, frame_data.dtype.names)
        vao = self.ctx.vertex_array(
            program=self.render_texture_program,
            content=[(vbo, format, *frame_data.dtype.names)],
        )
        self.ctx.copy_framebuffer(self.render_target_texture_fbo, self.color_buffer_fbo)
        self.render_target_texture.use(0)
        self.output_fbo.use()
        vao.render(gl.TRIANGLES)
        vbo.release()
        vao.release()
        # self.ctx.copy_framebuffer(self.output_fbo, self.color_buffer_fbo)

    def render_program(self, prog, data, indices=None):
        vbo = self.ctx.buffer(data.tobytes())
        ibo = (
            self.ctx.buffer(np.asarray(indices).astype("i4").tobytes())
            if indices is not None
            else None
        )
        # print(prog,vbo,data)
        vert_format = gl.detect_format(prog, data.dtype.names)
        # print(vert_format)
        vao = self.ctx.vertex_array(
            program=prog,
            content=[(vbo, vert_format, *data.dtype.names)],
            index_buffer=ibo,
        )

        vao.render(gl.TRIANGLES)
        # data, data_size = ibo.read(), ibo.size
        vbo.release()
        if ibo is not None:
            ibo.release()
        vao.release()
        # return data, data_size

    def render_image(self, mob):
        raise NotImplementedError  # TODO

    def render_previous(self, camera: Camera) -> None:
        raise NotImplementedError

    def render_vmobject(self, mob: OpenGLVMobject) -> None:  # type: ignore
        self.stencil_buffer_fbo.use()
        self.stencil_buffer_fbo.clear()
        self.render_target_fbo.use()
        # Setting camera uniforms

        self.ctx.enable(gl.BLEND)  # type: ignore
        # TODO: Because the Triangulation is messing up the normals this won't work
        self.ctx.blend_func = (  # type: ignore
            gl.SRC_ALPHA,
            gl.ONE_MINUS_SRC_ALPHA,
            gl.ONE,
            gl.ONE,
        )

        def enable_depth(mob):
            if sub.depth_test:
                self.ctx.enable(gl.DEPTH_TEST)  # type: ignore
            else:
                self.ctx.disable(gl.DEPTH_TEST)  # type: ignore

        for sub in mob.family_members_with_points():
            # TODO: review this renderer data optimization attempt
            if True:  # if sub.renderer_data is None:
                # Initialize
                GLVMobjectManager.init_render_data(sub)

            if not isinstance(sub.renderer_data, GLRenderData):
                return

            # if mob.colors_changed:

            #     mob.renderer_data.fill_rgbas = np.resize(mob.fill_color, (len(mob.renderer_data.mesh),4))

            # if mob.points_changed:3357
            #     if(mob.has_fill()):
            #         mob.renderer_data.mesh = ... # Triangulation todo

        family = mob.family_members_with_points()
        num_mobs = len(family)

        # Another stroke pass is needed in the beginning to deal with transparency properly
        for counter, sub in enumerate(family):
            if not isinstance(sub.renderer_data, GLRenderData):
                return
            enable_depth(sub)
            uniforms = GLVMobjectManager.read_uniforms(sub)
            uniforms["index"] = (counter + 1) / num_mobs / 2
            uniforms["disable_stencil"] = float(True)
            # uniforms['z_shift'] = counter/9 + 1/20
            self.ctx.copy_framebuffer(self.stencil_texture_fbo, self.stencil_buffer_fbo)
            self.stencil_texture.use(0)
            self.vmobject_stroke_program["stencil_texture"] = 0
            if sub.has_stroke():
                ProgramManager.write_uniforms(self.vmobject_stroke_program, uniforms)
                self.render_program(
                    self.vmobject_stroke_program,
                    self.get_stroke_shader_data(sub),
                    np.array(range(len(sub.points))),
                )

        for counter, sub in enumerate(family):
            if not isinstance(sub.renderer_data, GLRenderData):
                return
            enable_depth(sub)
            uniforms = GLVMobjectManager.read_uniforms(sub)
            # uniforms['z_shift'] = counter/9
            uniforms["index"] = (counter + 1) / num_mobs
            uniforms["disable_stencil"] = float(False)
            self.ctx.copy_framebuffer(self.stencil_texture_fbo, self.stencil_buffer_fbo)
            self.stencil_texture.use(0)
            self.vmobject_fill_program["stencil_texture"] = 0
            if sub.has_fill():
                ProgramManager.write_uniforms(self.vmobject_fill_program, uniforms)
                self.render_program(
                    self.vmobject_fill_program,
                    self.get_fill_shader_data(sub),
                    sub.renderer_data.vert_indices,
                )

        for counter, sub in enumerate(family):
            if not isinstance(sub.renderer_data, GLRenderData):
                return
            enable_depth(sub)
            uniforms = GLVMobjectManager.read_uniforms(sub)
            uniforms["index"] = (counter + 1) / num_mobs
            uniforms["disable_stencil"] = float(False)
            # uniforms['z_shift'] = counter/9 + 1/20
            self.ctx.copy_framebuffer(self.stencil_texture_fbo, self.stencil_buffer_fbo)
            self.stencil_texture.use(0)
            self.vmobject_stroke_program["stencil_texture"] = 0
            if sub.has_stroke():
                ProgramManager.write_uniforms(self.vmobject_stroke_program, uniforms)
                self.render_program(
                    self.vmobject_stroke_program,
                    self.get_stroke_shader_data(sub),
                    np.array(range(len(sub.points))),
                )

    def get_pixels(self) -> ImageType:
        raw = self.output_fbo.read(components=4, dtype="f1", clamp=True)  # RGBA, floats
        y, x = self.output_fbo.viewport[2:4]
        buf = np.frombuffer(raw, dtype=np.uint8).reshape((x, y, 4))
        # FIXME: this is slow?
        return np.flip(buf, 0)


class GLVMobjectManager:
    @staticmethod
    def init_render_data(mob: OpenGLVMobject):
        logger.debug("Initializing GLRenderData")
        mob.renderer_data = GLRenderData()

        # Generate Mesh
        mob.renderer_data.vert_indices = get_triangulation(mob)
        points_length = len(mob.points)

        # Generate Fill Color
        fill_color = np.array([c._internal_value for c in mob.fill_color])
        stroke_color = np.array([c._internal_value for c in mob.stroke_color])
        mob.renderer_data.fill_rgbas = prepare_array(fill_color, points_length)
        mob.renderer_data.stroke_rgbas = prepare_array(stroke_color, points_length)
        mob.renderer_data.stroke_widths = prepare_array(
            np.asarray(listify(mob.stroke_width)), points_length
        )
        mob.renderer_data.normals = np.repeat(
            [mob.get_unit_normal()], points_length, axis=0
        )
        mob.renderer_data.bounding_box = compute_bounding_box(mob)
        # print(mob.renderer_data)

    @staticmethod
    def read_uniforms(mob: OpenGLVMobject):
        uniforms = {}
        uniforms["reflectiveness"] = mob.reflectiveness
        uniforms["is_fixed_in_frame"] = float(mob.is_fixed_in_frame)
        uniforms["is_fixed_orientation"] = float(mob.is_fixed_orientation)
        uniforms["gloss"] = mob.gloss
        uniforms["shadow"] = mob.shadow
        uniforms["flat_stroke"] = float(mob.flat_stroke)
        uniforms["joint_type"] = float(mob.joint_type.value)
        uniforms["flat_stroke"] = float(mob.flat_stroke)
        return uniforms
