"""Scene variant configured with :class:`MovingCamera`.

.. SEEALSO::

    :mod:`.moving_camera`


Examples
--------

.. manim:: ChangingCameraWidthAndRestore

    class ChangingCameraWidthAndRestore(Scene):
        def construct(self):
            text = Text("Hello World").set_color(BLUE)
            self.add(text)
            self.camera.frame.save_state()
            self.play(self.camera.frame.animate.set(width=text.width * 1.2))
            self.wait(0.3)
            self.play(Restore(self.camera.frame))


.. manim:: MovingCameraCenter

    class MovingCameraCenter(Scene):
        def construct(self):
            s = Square(color=RED, fill_opacity=0.5).move_to(2 * LEFT)
            t = Triangle(color=GREEN, fill_opacity=0.5).move_to(2 * RIGHT)
            self.wait(0.3)
            self.add(s, t)
            self.play(self.camera.frame.animate.move_to(s))
            self.wait(0.3)
            self.play(self.camera.frame.animate.move_to(t))


.. manim:: MovingAndZoomingCamera

    class MovingAndZoomingCamera(Scene):
        def construct(self):
            s = Square(color=BLUE, fill_opacity=0.5).move_to(2 * LEFT)
            t = Triangle(color=YELLOW, fill_opacity=0.5).move_to(2 * RIGHT)
            self.add(s, t)
            self.play(self.camera.frame.animate.move_to(s).set(width=s.width*2))
            self.wait(0.3)
            self.play(self.camera.frame.animate.move_to(t).set(width=t.width*2))

            self.play(self.camera.frame.animate.move_to(ORIGIN).set(width=14))

.. manim:: MovingCameraOnGraph

    class MovingCameraOnGraph(Scene):
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

.. manim:: SlidingMultipleFrames

    class SlidingMultipleFrames(Scene):
        def construct(self):
            def create_frame(number):
                frame = Rectangle(width=16, height=9)
                circ = Circle().shift(LEFT)
                text = Tex(f"This is Frame {str(number)}").next_to(circ, RIGHT)
                frame.add(circ,text)
                return frame

            group = VGroup(*(create_frame(i) for i in range(4))).arrange_in_grid(buff=4)
            self.add(group)
            self.camera.auto_zoom(group[0], animate=False)
            for frame in group:
                self.play(self.camera.auto_zoom(frame))
                self.wait()

            self.play(self.camera.auto_zoom(group, margin=2))
"""

from __future__ import annotations

__all__ = ["MovingCameraScene"]

from typing import Any

from ..camera.camera import Camera
from ..camera.moving_camera import MovingCamera
from ..scene.scene import Scene


class MovingCameraScene(Scene):
    """A scene whose default camera class is :class:`MovingCamera`.

    Its camera exposes the standard animatable frame and auto-zoom behavior.
    """

    def __init__(
        self, camera_class: type[Camera] = MovingCamera, **kwargs: Any
    ) -> None:
        super().__init__(camera_class=camera_class, **kwargs)
