from pathlib import Path

from manim import *

# Copied from https://3b1b.github.io/manim/getting_started/example_scenes.html#surfaceexample.
# Lines that do not yet work with the Community Version are commented.


class TextTest(Scene):
    def construct(self):
        import string

        text = Text(string.ascii_lowercase, stroke_width=4, stroke_color=BLUE).scale(2)
        text2 = (
            Text(string.ascii_uppercase, stroke_width=4, stroke_color=BLUE)
            .scale(2)
            .next_to(text, DOWN)
        )
        # self.add(text, text2)
        self.play(Write(text))
        self.play(Write(text2))
        self.interactive_embed()


class InteractiveDevelopment(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        square = Square()

        self.play(Create(square))
        self.wait()

        # This opens an iPython terminal where you can keep writing
        # lines as if they were part of this construct method.
        # In particular, 'square', 'circle' and 'self' will all be
        # part of the local namespace in that terminal.
        # self.embed()

        # Try copying and pasting some of the lines below into
        # the interactive shell
        self.play(ReplacementTransform(square, circle))
        self.wait()
        self.play(circle.animate.stretch(4, 0))
        self.play(Rotate(circle, 90 * DEGREES))
        self.play(circle.animate.shift(2 * RIGHT).scale(0.25))

        # text = Text(
        #     """
        #     In general, using the interactive shell
        #     is very helpful when developing new scenes
        # """
        # )
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


class SurfaceExample(ThreeDScene):
    def construct(self):
        surface_text = Text("For 3d scenes, try using surfaces")
        surface_text.to_edge(UP)
        self.add_fixed_in_frame_mobjects(surface_text)
        self.wait(0.1)

        def sphere(u, v, radius=3):
            return radius * np.array(
                [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)]
            )

        def torus(u, v, major_radius, minor_radius):
            return np.array(
                [
                    (major_radius + minor_radius * np.cos(v)) * np.cos(u),
                    (major_radius + minor_radius * np.cos(v)) * np.sin(u),
                    minor_radius * np.sin(v),
                ]
            )

        # The texture is drawn onto the parameter domain: u runs left to right
        # across the image, v from its bottom to its top.
        texture = (
            Path(__file__).resolve().parent
            / "assets"
            / "1280px-Whole_world_-_land_and_oceans.jpg"
        )
        # Surfaces with the same resolution can be transformed into each other.
        surfaces = [
            ParametricSurface(
                func,
                u_range=[0, TAU],
                v_range=v_range,
                resolution=(64, 32),
                texture=texture,
            )
            for func, v_range in [
                (sphere, [0, PI]),
                (lambda u, v: torus(u, v, 1, 1), [0, TAU]),
                (lambda u, v: torus(u, v, 3, 1), [0, TAU]),
            ]
        ]
        for mob in surfaces:
            mob.shift(IN)

        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        surface = surfaces[0]
        wireframe = surface.get_wireframe(color=BLUE, stroke_width=0.5)
        self.play(
            FadeIn(surface),
            Create(wireframe, lag_ratio=0.01, run_time=3),
        )
        self.play(FadeOut(wireframe))
        self.play(Rotate(surface, PI / 2), run_time=2)
        for mob in surfaces[1:]:
            mob.rotate(PI / 2)

        self.play(Transform(surface, surfaces[1]), run_time=3)

        # Move the camera during the transition.
        self.move_camera(
            phi=60 * DEGREES,
            theta=-50 * DEGREES,
            added_anims=[Transform(surface, surfaces[2])],
            run_time=3,
        )
        self.begin_ambient_camera_rotation(rate=-0.1)

        # Play around with where the light is.
        light_text = Text("You can move around the light source")
        light_text.move_to(surface_text)
        self.add_fixed_in_frame_mobjects(light_text)
        self.play(FadeTransform(surface_text, light_text))
        light = self.camera.light_source
        self.play(light.animate.move_to(3 * IN), run_time=5)
        self.play(light.animate.shift(10 * OUT), run_time=5)
