import os
from pathlib import Path

from manim import *
from manim.opengl import *
from manim.utils.opengl import *

# Copied from https://3b1b.github.io/manim/getting_started/example_scenes.html#surfaceexample.
# Lines that do not yet work with the Community Version are commented.


time = 0


class Test(Scene):
    def construct(self):
        # c = OpenGLCircle()
        # self.play(Create(c))

        c = OpenGLSphere()
        self.add(c)
        self.embed_2()


class ShaderExample(Scene):
    def construct(self):
        global time
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
