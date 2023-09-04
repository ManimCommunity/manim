import numpy as np
from renderer import Renderer

from manim.mobject.types.vectorized_mobject import VMobject
from renderer import RendererData 

class GLRenderData(RendererData):
    def __init__(self) -> None:
        super().__init__()
        self.fill_rgbas = np.zeros((1,4))
        self.stroke_rgbas = np.zeros((1,4))
        self.normals = np.zeros((1,4))
        self.mesh = np.zeros((0,3))
        self.bounding_box = np.zeros((3,3))

class OpenGLRenderer(Renderer): 
    def __init__(
        self,
        ctx: moderngl.Context | None = None,
        background_image: str | None = None,
        frame_config: dict = {},
        pixel_width: int = config.pixel_width,
        pixel_height: int = config.pixel_height,
        fps: int = config.frame_rate,
        # Note: frame height and width will be resized to match the pixel aspect rati
        background_color=BLACK,
        background_opacity: float = 1.0,
        # Points in vectorized mobjects with norm greater
        # than this value will be rescaled
        max_allowable_norm: float = 1.0,
        image_mode: str = "RGBA",
        n_channels: int = 4,
        pixel_array_dtype: type = np.uint8,
        light_source_position: np.ndarray = np.array([-10, 10, 10]),
        # Although vector graphics handle antialiasing fine
        # without multisampling, for 3d scenes one might want
        # to set samples to be greater than 0.
        samples: int = 0,
    ) -> None:
        self.background_image = background_image
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.fps = fps
        self.max_allowable_norm = max_allowable_norm
        self.image_mode = image_mode
        self.n_channels = n_channels
        self.pixel_array_dtype = pixel_array_dtype
        self.light_source_position = light_source_position
        self.samples = samples

        self.rgb_max_val: float = np.iinfo(self.pixel_array_dtype).max
        self.background_color: list[float] = list(
            color_to_rgba(background_color, background_opacity)
        )
        # self.init_frame(**frame_config)
        # self.init_context(ctx)
        # self.init_shaders()
        # self.init_textures()
        # self.init_light_source()
        # self.refresh_perspective_uniforms()
        # A cached map from mobjects to their associated list of render groups
        # so that these render groups are not regenerated unnecessarily for static
        # mobjects
        self.mob_to_render_groups: dict = {}

    def render_vmobject(self, mob: VMobject) -> None:
        # Should be in OpenGL renderer
        if not mob.renderer_data:
            # Initalization
            mob.renderer_data = GLRenderData()

        if mob.colors_changed:
            mob.renderer_data.fill_rgbas = np.resize(mob.fill_color, (len(mob.renderer_data.mesh),4))
        
        if mob.points_changed:
            if(mob.has_fill()):
                mob.renderer_data.mesh = ... # Triangulation todo
        
        # set shader
        # use vbo
        # render fill

        # set shader
        # use vbo
        # render stroke
        self.fbo ...

#     def init_frame(self, **config) -> None:
#         self.frame = OpenGLCameraFrame(**config)

#     def init_context(self, ctx: moderngl.Context | None = None) -> None:
#         if ctx is None:
#             ctx = moderngl.create_standalone_context()
#             fbo = self.get_fbo(ctx, 0)
#         else:
#             fbo = ctx.detect_framebuffer()

#         self.ctx = ctx
#         self.fbo = fbo
#         self.set_ctx_blending()

#         # For multisample antisampling
#         fbo_msaa = self.get_fbo(ctx, self.samples)
#         fbo_msaa.use()
#         self.fbo_msaa = fbo_msaa

#     def set_ctx_blending(self, enable: bool = True) -> None:
#         if enable:
#             self.ctx.enable(moderngl.BLEND)
#         else:
#             self.ctx.disable(moderngl.BLEND)

#     def set_ctx_depth_test(self, enable: bool = True) -> None:
#         if enable:
#             self.ctx.enable(moderngl.DEPTH_TEST)
#         else:
#             self.ctx.disable(moderngl.DEPTH_TEST)

#     def init_light_source(self) -> None:
#         self.light_source = OpenGLPoint(self.light_source_position)

#     # Methods associated with the frame buffer
#     def get_fbo(self, ctx: moderngl.Context, samples: int = 0) -> moderngl.Framebuffer:
#         pw = self.pixel_width
#         ph = self.pixel_height
#         return ctx.framebuffer(
#             color_attachments=ctx.texture(
#                 (pw, ph), components=self.n_channels, samples=samples
#             ),
#             depth_attachment=ctx.depth_renderbuffer((pw, ph), samples=samples),
#         )

#     def clear(self) -> None:
#         self.fbo.clear(*self.background_color)
#         self.fbo_msaa.clear(*self.background_color)

#     def reset_pixel_shape(self, new_width: int, new_height: int) -> None:
#         self.pixel_width = new_width
#         self.pixel_height = new_height
#         self.refresh_perspective_uniforms()

#     def get_raw_fbo_data(self, dtype: str = "f1") -> bytes:
#         # Copy blocks from the fbo_msaa to the drawn fbo using Blit
#         # pw, ph = (self.pixel_width, self.pixel_height)
#         # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo_msaa.glo)
#         # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.fbo.glo)
#         # gl.glBlitFramebuffer(
#         #     0, 0, pw, ph, 0, 0, pw, ph, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
#         # )

#         self.ctx.copy_framebuffer(self.fbo, self.fbo_msaa)
#         return self.fbo.read(
#             viewport=self.fbo.viewport,
#             components=self.n_channels,
#             dtype=dtype,
#         )

#     def get_image(self) -> Image.Image:
#         return Image.frombytes(
#             "RGBA",
#             self.get_pixel_shape(),
#             self.get_raw_fbo_data(),
#             "raw",
#             "RGBA",
#             0,
#             -1,
#         )

#     def get_pixel_array(self) -> np.ndarray:
#         raw = self.get_raw_fbo_data(dtype="f4")
#         flat_arr = np.frombuffer(raw, dtype="f4")
#         arr = flat_arr.reshape([*reversed(self.fbo.size), self.n_channels])
#         arr = arr[::-1]
#         # Convert from float
#         return (self.rgb_max_val * arr).astype(self.pixel_array_dtype)

#     def get_texture(self):
#         texture = self.ctx.texture(
#             size=self.fbo.size, components=4, data=self.get_raw_fbo_data(), dtype="f4"
#         )
#         return texture

#     # Getting camera attributes
#     def get_pixel_shape(self) -> tuple[int, int]:
#         return self.fbo.viewport[2:4]
#         # return (self.pixel_width, self.pixel_height)

#     def get_pixel_width(self) -> int:
#         return self.get_pixel_shape()[0]

#     def get_pixel_height(self) -> int:
#         return self.get_pixel_shape()[1]

#     def get_frame_height(self) -> float:
#         return self.frame.get_height()

#     def get_frame_width(self) -> float:
#         return self.frame.get_width()

#     def get_frame_shape(self) -> tuple[float, float]:
#         return (self.get_frame_width(), self.get_frame_height())

#     def get_frame_center(self) -> np.ndarray:
#         return self.frame.get_center()

#     def get_location(self) -> tuple[float, float, float] | np.ndarray:
#         return self.frame.get_implied_camera_location()

#     def resize_frame_shape(self, fixed_dimension: bool = False) -> None:
#         """
#         Changes frame_shape to match the aspect ratio
#         of the pixels, where fixed_dimension determines
#         whether frame_height or frame_width
#         remains fixed while the other changes accordingly.
#         """
#         pixel_height = self.get_pixel_height()
#         pixel_width = self.get_pixel_width()
#         frame_height = self.get_frame_height()
#         frame_width = self.get_frame_width()
#         aspect_ratio = fdiv(pixel_width, pixel_height)
#         if not fixed_dimension:
#             frame_height = frame_width / aspect_ratio
#         else:
#             frame_width = aspect_ratio * frame_height
#         self.frame.set_height(frame_height)
#         self.frame.set_width(frame_width)

#     # Rendering
#     def capture(self, *mobjects: OpenGLMobject) -> None:
#         self.refresh_perspective_uniforms()
#         for mobject in mobjects:
#             for render_group in self.get_render_group_list(mobject):
#                 self.render(render_group)

#     def render(self, render_group: dict[str, Any]) -> None:
#         shader_wrapper: ShaderWrapper = render_group["shader_wrapper"]
#         shader_program = render_group["prog"]
#         self.set_shader_uniforms(shader_program, shader_wrapper)
#         self.set_ctx_depth_test(shader_wrapper.depth_test)
#         render_group["vao"].render(int(shader_wrapper.render_primitive))
#         if render_group["single_use"]:
#             self.release_render_group(render_group)

#     def get_render_group_list(self, mobject: OpenGLMobject) -> Iterable[dict[str, Any]]:
#         if mobject.is_changing():
#             return self.generate_render_group_list(mobject)

#         # Otherwise, cache result for later use
#         key = id(mobject)
#         if key not in self.mob_to_render_groups:
#             self.mob_to_render_groups[key] = list(
#                 self.generate_render_group_list(mobject)
#             )
#         return self.mob_to_render_groups[key]

#     def generate_render_group_list(
#         self, mobject: OpenGLMobject
#     ) -> Iterable[dict[str, Any]]:
#         return (
#             self.get_render_group(sw, single_use=mobject.is_changing())
#             for sw in mobject.get_shader_wrapper_list()
#         )

#     def get_render_group(
#         self, shader_wrapper: ShaderWrapper, single_use: bool = True
#     ) -> dict[str, Any]:
#         # Data buffers
#         vbo = self.ctx.buffer(shader_wrapper.vert_data.tobytes())
#         if shader_wrapper.vert_indices is None:
#             ibo = None
#         else:
#             vert_index_data = shader_wrapper.vert_indices.astype("i4").tobytes()
#             if vert_index_data:
#                 ibo = self.ctx.buffer(vert_index_data)
#             else:
#                 ibo = None

#         # Program an vertex array
#         shader_program, vert_format = self.get_shader_program(shader_wrapper)  # type: ignore
#         vao = self.ctx.vertex_array(
#             program=shader_program,
#             content=[(vbo, vert_format, *shader_wrapper.vert_attributes)],
#             index_buffer=ibo,
#         )
#         return {
#             "vbo": vbo,
#             "ibo": ibo,
#             "vao": vao,
#             "prog": shader_program,
#             "shader_wrapper": shader_wrapper,
#             "single_use": single_use,
#         }

#     def release_render_group(self, render_group: dict[str, Any]) -> None:
#         for key in ["vbo", "ibo", "vao"]:
#             if render_group[key] is not None:
#                 render_group[key].release()

#     def refresh_static_mobjects(self) -> None:
#         for render_group in it.chain(*self.mob_to_render_groups.values()):
#             self.release_render_group(render_group)
#         self.mob_to_render_groups = {}

#     # Shaders
#     def init_shaders(self) -> None:
#         # Initialize with the null id going to None
#         self.id_to_shader_program: dict[int, tuple[moderngl.Program, str] | None] = {
#             hash(""): None
#         }

#     def get_shader_program(
#         self, shader_wrapper: ShaderWrapper
#     ) -> tuple[moderngl.Program, str] | None:
#         sid = shader_wrapper.get_program_id()
#         if sid not in self.id_to_shader_program:
#             # Create shader program for the first time, then cache
#             # in the id_to_shader_program dictionary
#             program = self.ctx.program(**shader_wrapper.get_program_code())
#             vert_format = moderngl.detect_format(
#                 program, shader_wrapper.vert_attributes
#             )
#             self.id_to_shader_program[sid] = (program, vert_format)

#         return self.id_to_shader_program[sid]

#     def set_shader_uniforms(
#         self,
#         shader: moderngl.Program,
#         shader_wrapper: ShaderWrapper,
#     ) -> None:
#         for name, path in shader_wrapper.texture_paths.items():
#             tid = self.get_texture_id(path)
#             shader[name].value = tid
#         for name, value in it.chain(
#             self.perspective_uniforms.items(), shader_wrapper.uniforms.items()
#         ):
#             if name in shader:
#                 if isinstance(value, np.ndarray) and value.ndim > 0:
#                     value = tuple(value)
#                 shader[name].value = value
#             else:
#                 logger.debug(f"Uniform {name} not found in shader {shader}")

#     def refresh_perspective_uniforms(self) -> None:
#         frame = self.frame
#         # Orient light
#         rotation = frame.get_inverse_camera_rotation_matrix()
#         offset = frame.get_center()
#         light_pos = np.dot(rotation, self.light_source.get_location() + offset)
#         cam_pos = self.frame.get_implied_camera_location()  # TODO

#         self.perspective_uniforms = {
#             "frame_shape": frame.get_shape(),
#             "pixel_shape": self.get_pixel_shape(),
#             "camera_offset": tuple(offset),
#             "camera_rotation": tuple(np.array(rotation).T.flatten()),
#             "camera_position": tuple(cam_pos),
#             "light_source_position": tuple(light_pos),
#             "focal_distance": frame.get_focal_distance(),
#         }

#     def init_textures(self) -> None:
#         self.n_textures: int = 0
#         self.path_to_texture: dict[str, tuple[int, moderngl.Texture]] = {}

#     def get_texture_id(self, path: str) -> int:
#         if path not in self.path_to_texture:
#             if self.n_textures == 15:  # I have no clue why this is needed
#                 self.n_textures += 1
#             tid = self.n_textures
#             self.n_textures += 1
#             im = Image.open(path).convert("RGBA")
#             texture = self.ctx.texture(
#                 size=im.size,
#                 components=len(im.getbands()),
#                 data=im.tobytes(),
#             )
#             texture.use(location=tid)
#             self.path_to_texture[path] = (tid, texture)
#         return self.path_to_texture[path][0]

#     def release_texture(self, path: str):
#         tid_and_texture = self.path_to_texture.pop(path, None)
#         if tid_and_texture:
#             tid_and_texture[1].release()
#         return self