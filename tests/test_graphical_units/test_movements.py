from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "movements"


@frames_comparison(last_frame=False)
def test_Homotopy(scene):
    def func(x, y, z, t):
        norm = np.linalg.norm([x, y])
        tau = interpolate(5, -5, t) + norm / config["frame_x_radius"]
        alpha = sigmoid(tau)
        return [x, y + 0.5 * np.sin(2 * np.pi * alpha) - t * SMALL_BUFF / 2, z]

    square = Square()
    scene.play(Homotopy(func, square))


@frames_comparison(last_frame=False)
def test_PhaseFlow(scene):
    square = Square()

    def func(t):
        return t * 0.5 * UP

    scene.play(PhaseFlow(func, square))


@frames_comparison(last_frame=False)
def test_MoveAlongPath(scene):
    square = Square()
    dot = Dot()
    scene.play(MoveAlongPath(dot, square))


@frames_comparison(last_frame=False)
def test_Rotate(scene):
    square = Square()
    scene.play(Rotate(square, PI))


@frames_comparison(last_frame=False)
def test_MoveTo(scene):
    square = Square()
    scene.play(square.animate.move_to(np.array([1.0, 1.0, 0.0])))


@frames_comparison(last_frame=False)
def test_Shift(scene):
    square = Square()
    scene.play(square.animate.shift(UP))
