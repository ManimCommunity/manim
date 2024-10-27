"""A scene whose camera can be moved around.

.. SEEALSO::

    :mod:`.moving_camera`


Examples
--------

.. manim:: ChangingCameraWidthAndRestore

    class ChangingCameraWidthAndRestore(MovingCameraScene):
        def construct(self):
            text = Text("Hello World").set_color(BLUE)
            self.add(text)
            self.camera.frame.save_state()
            self.play(self.camera.frame.animate.set(width=text.width * 1.2))
            self.wait(0.3)
            self.play(Restore(self.camera.frame))


.. manim:: MovingCameraCenter

    class MovingCameraCenter(MovingCameraScene):
        def construct(self):
            s = Square(color=RED, fill_opacity=0.5).move_to(2 * LEFT)
            t = Triangle(color=GREEN, fill_opacity=0.5).move_to(2 * RIGHT)
            self.wait(0.3)
            self.add(s, t)
            self.play(self.camera.frame.animate.move_to(s))
            self.wait(0.3)
            self.play(self.camera.frame.animate.move_to(t))


.. manim:: MovingAndZoomingCamera

    class MovingAndZoomingCamera(MovingCameraScene):
        def construct(self):
            s = Square(color=BLUE, fill_opacity=0.5).move_to(2 * LEFT)
            t = Triangle(color=YELLOW, fill_opacity=0.5).move_to(2 * RIGHT)
            self.add(s, t)
            self.play(self.camera.frame.animate.move_to(s).set(width=s.width*2))
            self.wait(0.3)
            self.play(self.camera.frame.animate.move_to(t).set(width=t.width*2))

            self.play(self.camera.frame.animate.move_to(ORIGIN).set(width=14))

.. manim:: MovingCameraOnGraph

    class MovingCameraOnGraph(MovingCameraScene):
        def construct(self):
            self.camera.frame.save_state()

            ax = Axes(x_range=[-1, 10], y_range=[-1, 10])
            graph = ax.plot(lambda x: np.sin(x), color=WHITE, x_range=[0, 3 * PI])

            dot_1 = Dot(ax.i2gp(graph.t_min, graph))
            dot_2 = Dot(ax.i2gp(graph.t_max, graph))
            self.add(ax, graph, dot_1, dot_2)

            self.play(self.camera.frame.animate.scale(0.5).move_to(dot_1))
            self.play(self.camera.frame.animate.move_to(dot_2))
            self.play(Restore(self.camera.frame))
            self.wait()

.. manim:: SlidingMultipleScenes

    class SlidingMultipleScenes(MovingCameraScene):
        def construct(self):
            def create_scene(number):
                frame = Rectangle(width=16,height=9)
                circ = Circle().shift(LEFT)
                text = Tex(f"This is Scene {str(number)}").next_to(circ, RIGHT)
                frame.add(circ,text)
                return frame

            group = VGroup(*(create_scene(i) for i in range(4))).arrange_in_grid(buff=4)
            self.add(group)
            self.camera.auto_zoom(group[0], animate=False)
            for scene in group:
                self.play(self.camera.auto_zoom(scene))
                self.wait()

            self.play(self.camera.auto_zoom(group, margin=2))
"""

from __future__ import annotations

__all__ = ["MovingCameraScene"]

from manim.animation.animation import Animation

from ..camera.moving_camera import MovingCamera
from ..scene.scene import Scene
from ..utils.family import extract_mobject_family_members
from ..utils.iterables import list_update


class MovingCameraScene(Scene):
    """
    This is a Scene, with special configurations and properties that
    make it suitable for cases where the camera must be moved around.

    Note: Examples are included in the moving_camera_scene module
    documentation, see below in the 'see also' section.

    .. SEEALSO::

        :mod:`.moving_camera_scene`
        :class:`.MovingCamera`
    """

    def __init__(self, camera_class=MovingCamera, **kwargs):
        super().__init__(camera_class=camera_class, **kwargs)

    def get_moving_mobjects(self, *animations: Animation):
        """
        This method returns a list of all of the Mobjects in the Scene that
        are moving, that are also in the animations passed.

        Parameters
        ----------
        *animations
            The Animations whose mobjects will be checked.
        """
        moving_mobjects = super().get_moving_mobjects(*animations)
        all_moving_mobjects = extract_mobject_family_members(moving_mobjects)
        movement_indicators = self.renderer.camera.get_mobjects_indicating_movement()
        for movement_indicator in movement_indicators:
            if movement_indicator in all_moving_mobjects:
                # When one of these is moving, the camera should
                # consider all mobjects to be moving
                return list_update(self.mobjects, moving_mobjects)
        return moving_mobjects
