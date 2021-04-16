import os
from pathlib import Path

from manim import *
from manim.opengl import *
import manim.utils.opengl as opengl
import manim.utils.space_ops as space_ops

# Copied from https://3b1b.github.io/manim/getting_started/example_scenes.html#surfaceexample.
# Lines that do not yet work with the Community Version are commented.


class CubeTest(Scene):
    def construct(self):
        config["background_color"] = "#333333"

        shader = Shader(self.renderer.context, name="vertex_colors")

        # shader.set_uniform("u_color", (1.0, 0.0, 0.0, 1.0))
        attributes = np.zeros(
            18,
            dtype=[
                ("in_vert", np.float32, (4,)),
                ("in_color", np.float32, (4,)),
            ],
        )
        attributes["in_vert"] = np.array(
            [
                # xy plane
                [-1, -1, 0, 1],
                [-1, 1, 0, 1],
                [1, 1, 0, 1],
                [-1, -1, 0, 1],
                [1, -1, 0, 1],
                [1, 1, 0, 1],
                # yz plane
                [0, -1, -1, 1],
                [0, -1, 1, 1],
                [0, 1, 1, 1],
                [0, -1, -1, 1],
                [0, 1, -1, 1],
                [0, 1, 1, 1],
                # xz plane
                [-1, 0, -1, 1],
                [-1, 0, 1, 1],
                [1, 0, 1, 1],
                [-1, 0, -1, 1],
                [1, 0, -1, 1],
                [1, 0, 1, 1],
            ]
        )
        attributes["in_color"] = np.array(
            [
                # xy plane
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                # yz plane
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                # xz plane
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        )
        mesh = Mesh(shader, attributes)
        self.add(mesh)

        # def update_mesh(mesh, dt):
        #     mesh.model_matrix = np.matmul(
        #         opengl.rotation_matrix(y=dt), mesh.model_matrix
        #     )

        # mesh.add_updater(update_mesh)

        total_time = 0

        def update_camera(camera, dt):
            nonlocal total_time
            # Rotate the camera in place.
            # camera_translation = camera.get_position()
            # camera.model_matrix = np.matmul(
            #     translation_matrix(*-camera_translation), camera.model_matrix
            # )
            # camera.model_matrix = np.matmul(
            #     rotation_matrix(x=dt / (TAU / 8)), camera.model_matrix
            # )
            # camera.model_matrix = np.matmul(
            #     translation_matrix(*camera_translation), camera.model_matrix
            # )

            # if total_time <= 2.5:
            #     # Rotate the camera around the z axis.
            #     self.camera.model_matrix = np.matmul(
            #         rotation_matrix(z=dt / 8),
            #         self.camera.model_matrix,
            #     )
            # else:
            #     # Increase the angle off the z axis.
            #     origin_to_camera = self.camera.get_position()
            #     axis_of_rotation = np.cross(OUT, origin_to_camera)
            #     rot_matrix = space_ops.rotation_matrix(dt / 4, axis_of_rotation)
            #     print(axis_of_rotation)

            #     # Convert to homogeneous coordinates.
            #     rot_matrix = np.hstack((rot_matrix, np.array([[0], [0], [0]])))
            #     rot_matrix = np.vstack((rot_matrix, np.array([0, 0, 0, 1])))

            #     self.camera.model_matrix = np.matmul(
            #         rot_matrix,
            #         self.camera.model_matrix,
            #     )

            total_time += dt

        camera_target = ORIGIN

        def on_mouse_drag(point, d_point, buttons, modifiers):
            nonlocal camera_target
            # Left click drag.
            if buttons == 1:
                # Translate to target the origin and rotate around the z axis.
                self.camera.model_matrix = (
                    opengl.rotation_matrix(z=-d_point[0])
                    @ opengl.translation_matrix(*-camera_target)
                    @ self.camera.model_matrix
                )

                # Rotation off of the z axis.
                camera_position = self.camera.get_position()
                camera_y_axis = self.camera.model_matrix[:3, 1]
                axis_of_rotation = space_ops.normalize(
                    np.cross(camera_y_axis, camera_position)
                )
                rotation_matrix = space_ops.rotation_matrix(
                    d_point[1], axis_of_rotation, homogeneous=True
                )

                maximum_polar_angle = PI / 2
                minimum_polar_angle = 0

                potential_camera_model_matrix = (
                    rotation_matrix @ self.camera.model_matrix
                )
                potential_camera_location = potential_camera_model_matrix[:3, 3]
                potential_camera_y_axis = potential_camera_model_matrix[:3, 1]
                sign = (
                    np.sign(potential_camera_y_axis[2])
                    if potential_camera_y_axis[2] != 0
                    else 1
                )
                potential_polar_angle = sign * np.arccos(
                    potential_camera_location[2]
                    / np.linalg.norm(potential_camera_location)
                )
                if minimum_polar_angle <= potential_polar_angle <= maximum_polar_angle:
                    self.camera.model_matrix = potential_camera_model_matrix
                else:
                    sign = np.sign(camera_y_axis[2]) if camera_y_axis[2] != 0 else 1
                    current_polar_angle = sign * np.arccos(
                        camera_position[2] / np.linalg.norm(camera_position)
                    )
                    if potential_polar_angle > maximum_polar_angle:
                        polar_angle_delta = maximum_polar_angle - current_polar_angle
                    else:
                        polar_angle_delta = minimum_polar_angle - current_polar_angle
                    rotation_matrix = space_ops.rotation_matrix(
                        polar_angle_delta, axis_of_rotation, homogeneous=True
                    )
                    self.camera.model_matrix = (
                        rotation_matrix @ self.camera.model_matrix
                    )

                # Translate to target the original target.
                self.camera.model_matrix = (
                    opengl.translation_matrix(*camera_target) @ self.camera.model_matrix
                )
            # Right click drag.
            elif buttons == 4:
                camera_x_axis = self.camera.model_matrix[:3, 0]
                horizontal_shift_vector = -d_point[0] * camera_x_axis
                vertical_shift_vector = -d_point[1] * np.cross(OUT, camera_x_axis)
                total_shift_vector = horizontal_shift_vector + vertical_shift_vector

                self.camera.model_matrix = (
                    opengl.translation_matrix(*total_shift_vector)
                    @ self.camera.model_matrix
                )
                camera_target += total_shift_vector

        def on_mouse_scroll(point, offset):
            nonlocal camera_target
            camera_to_target = camera_target - self.camera.get_position()
            camera_to_target *= np.sign(offset[1])
            shift_vector = 0.01 * camera_to_target
            self.camera.model_matrix = (
                opengl.translation_matrix(*shift_vector) @ self.camera.model_matrix
            )

        setattr(self, "on_mouse_drag", on_mouse_drag)
        setattr(self, "on_mouse_scroll", on_mouse_scroll)

        self.camera.add_updater(update_camera)

        # self.camera.model_matrix = (
        #     opengl.rotation_matrix(x=TAU / 8) @ self.camera.model_matrix
        # )
        # self.camera.model_matrix = (
        #     opengl.rotation_matrix(z=TAU / 8) @ self.camera.model_matrix
        # )

        self.embed_2()


class FullscreenQuadTest(Scene):
    def construct(self):
        surface = FullScreenQuad(
            self.renderer.context,
            """
            #version 330


            #define TWO_PI 6.28318530718

            uniform vec2 u_resolution;
            uniform float u_time;

            //  Function from Iñigo Quiles
            //  https://www.shadertoy.com/view/MsS3Wc
            vec3 hsb2rgb( in vec3 c ){
                vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),
                                         6.0)-3.0)-1.0,
                                 0.0,
                                 1.0 );
                rgb = rgb*rgb*(3.0-2.0*rgb);
                return c.z * mix( vec3(1.0), rgb, c.y);
            }

            void main(){
                vec2 st = gl_FragCoord.xy/u_resolution;
                vec3 color = vec3(0.0);

                // Use polar coordinates instead of cartesian
                vec2 toCenter = vec2(0.5)-st;
                float angle = atan(toCenter.y,toCenter.x);
                angle += u_time;
                float radius = length(toCenter)*2.0;

                // Map the angle (-PI to PI) to the Hue (from 0 to 1)
                // and the Saturation to the radius
                color = hsb2rgb(vec3((angle/TWO_PI)+0.5,radius,1.0));

                gl_FragColor = vec4(color,1.0);
            }
            """,
            output_color_variable="gl_FragColor",
        )
        surface.shader.set_uniform("u_resolution", (854.0, 480.0))
        shader_time = 0

        def update_surface(surface):
            nonlocal shader_time
            surface.shader.set_uniform("u_time", shader_time)
            shader_time += 1 / 60.0

        surface.add_updater(update_surface)
        self.add(surface)
        # self.wait(5)
        self.embed_2()


class Test2(Scene):
    def construct(self):
        attributes = np.zeros(
            6,
            dtype=[
                ("in_red", np.float32, (1,)),
                ("in_green", np.float32, (1,)),
                ("in_blue", np.float32, (1,)),
            ],
        )
        attributes["in_red"] = np.array(
            [
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
            ]
        )
        attributes["in_green"] = np.array(
            [
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
            ]
        )
        attributes["in_blue"] = np.array(
            [
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
            ]
        )

        surface = FullScreenQuad(
            self.renderer.context,
            """
            #version 330

            in float v_red;
            in float v_green;
            in float v_blue;
            out vec4 frag_color;

            void main() {
              frag_color = vec4(v_red, v_green, v_blue, 1);
            }
            """,
            attributes,
        )

        increase = True

        def update_surface(mesh, dt):
            nonlocal increase
            if increase:
                mesh.attributes["in_red"][:, 0] += dt
                mesh.attributes["in_green"][:, 0] += dt
                mesh.attributes["in_blue"][:, 0] += dt
            else:
                mesh.attributes["in_red"][:, 0] -= dt
                mesh.attributes["in_green"][:, 0] -= dt
                mesh.attributes["in_blue"][:, 0] -= dt
            if mesh.attributes["in_red"][0][0] >= 1:
                increase = False
            elif mesh.attributes["in_red"][0][0] <= 0:
                increase = True

        surface.add_updater(update_surface)

        self.add(surface)
        self.wait(5)


class ShaderExample(Scene):
    def construct(self):
        config["background_color"] = "#333333"

        c = OpenGLCircle(fill_opacity=0.7).shift(UL)
        self.add(c)

        shader = Shader(
            self.renderer.context,
            source=dict(
                vertex_shader="""
                #version 330

                in vec4 in_vert;
                in vec4 in_color;
                out vec4 v_color;
                uniform mat4 u_model_view_matrix;
                uniform mat4 u_projection_matrix;

                void main() {
                    v_color = in_color;
                    vec4 camera_space_vertex = u_model_view_matrix * in_vert;
                    vec4 clip_space_vertex = u_projection_matrix * camera_space_vertex;
                    gl_Position = clip_space_vertex;
                }
            """,
                fragment_shader="""
            #version 330

            in vec4 v_color;
            out vec4 frag_color;

            void main() {
              frag_color = v_color;
            }
            """,
            ),
        )
        shader.set_uniform("u_model_view_matrix", opengl.view_matrix())
        shader.set_uniform(
            "u_projection_matrix", opengl.orthographic_projection_matrix()
        )

        attributes = np.zeros(
            6,
            dtype=[
                ("in_vert", np.float32, (4,)),
                ("in_color", np.float32, (4,)),
            ],
        )
        attributes["in_vert"] = np.array(
            [
                [-1, -1, 0, 1],
                [-1, 1, 0, 1],
                [1, 1, 0, 1],
                [-1, -1, 0, 1],
                [1, -1, 0, 1],
                [1, 1, 0, 1],
            ]
        )
        attributes["in_color"] = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        )
        mesh = Mesh(shader, attributes)
        self.add(mesh)

        self.wait(5)
        # self.embed_2()


class InteractiveDevelopment(Scene):
    def construct(self):
        circle = OpenGLCircle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        square = OpenGLSquare()

        self.play(Create(square))
        self.wait()

        # This opens an iPython termnial where you can keep writing
        # lines as if they were part of this construct method.
        # In particular, 'square', 'circle' and 'self' will all be
        # part of the local namespace in that terminal.
        self.embed()

        # Try copying and pasting some of the lines below into
        # the interactive shell
        self.play(ReplacementTransform(square, circle))
        self.wait()
        self.play(circle.animate.stretch(4, 0))
        self.play(Rotate(circle, 90 * DEGREES))
        self.play(circle.animate.shift(2 * RIGHT).scale(0.25))

        # text = Text("""
        #     In general, using the interactive shell
        #     is very helpful when developing new scenes
        # """)
        # self.play(Write(text))

        # # In the interactive shell, you can just type
        # # play, add, remove, clear, wait, save_state and restore,
        # # instead of self.play, self.add, self.remove, etc.

        # # To interact with the window, type touch().  You can then
        # # scroll in the window, or zoom by holding down 'z' while scrolling,
        # # and change camera perspective by holding down 'd' while moving
        # # the mouse.  Press 'r' to reset to the standard camera position.
        # # Press 'q' to stop interacting with the window and go back to
        # # typing new commands into the shell.

        # # In principle you can customize a scene to be responsive to
        # # mouse and keyboard interactions
        # always(circle.move_to, self.mouse_point)


class SurfaceExample(Scene):
    def construct(self):
        # surface_text = Text("For 3d scenes, try using surfaces")
        # surface_text.fix_in_frame()
        # surface_text.to_edge(UP)
        # self.add(surface_text)
        # self.wait(0.1)

        torus1 = OpenGLTorus(r1=1, r2=1)
        torus2 = OpenGLTorus(r1=3, r2=1)
        sphere = OpenGLSphere(radius=3, resolution=torus1.resolution)
        # You can texture a surface with up to two images, which will
        # be interpreted as the side towards the light, and away from
        # the light.  These can be either urls, or paths to a local file
        # in whatever you've set as the image directory in
        # the custom_config.yml file

        script_location = Path(os.path.realpath(__file__)).parent
        day_texture = (
            script_location / "assets" / "1280px-Whole_world_-_land_and_oceans.jpg"
        )
        night_texture = script_location / "assets" / "1280px-The_earth_at_night.jpg"

        surfaces = [
            OpenGLTexturedSurface(surface, day_texture, night_texture)
            for surface in [sphere, torus1, torus2]
        ]

        for mob in surfaces:
            mob.shift(IN)
            mob.mesh = OpenGLSurfaceMesh(mob)
            mob.mesh.set_stroke(BLUE, 1, opacity=0.5)

        # Set perspective
        frame = self.renderer.camera
        frame.set_euler_angles(
            theta=-30 * DEGREES,
            phi=70 * DEGREES,
        )

        surface = surfaces[0]

        self.play(
            FadeIn(surface),
            Create(surface.mesh, lag_ratio=0.01, run_time=3),
        )
        for mob in surfaces:
            mob.add(mob.mesh)
        surface.save_state()
        self.play(Rotate(surface, PI / 2), run_time=2)
        for mob in surfaces[1:]:
            mob.rotate(PI / 2)

        self.play(Transform(surface, surfaces[1]), run_time=3)

        self.play(
            Transform(surface, surfaces[2]),
            # Move camera frame during the transition
            frame.animate.increment_phi(-10 * DEGREES),
            frame.animate.increment_theta(-20 * DEGREES),
            run_time=3,
        )
        # Add ambient rotation
        frame.add_updater(lambda m, dt: m.increment_theta(-0.1 * dt))

        # Play around with where the light is
        # light_text = Text("You can move around the light source")
        # light_text.move_to(surface_text)
        # light_text.fix_in_frame()

        # self.play(FadeTransform(surface_text, light_text))
        light = self.camera.light_source
        self.add(light)
        light.save_state()
        self.play(light.animate.move_to(3 * IN), run_time=5)
        self.play(light.animate.shift(10 * OUT), run_time=5)

        # drag_text = Text("Try moving the mouse while pressing d or s")
        # drag_text.move_to(light_text)
        # drag_text.fix_in_frame()
