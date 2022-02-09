from manim import Circle, Square, ThreeDScene


def test_fixed_mobjects():
    scene = ThreeDScene()
    s = Square()
    c = Circle()
    scene.add_fixed_in_frame_mobjects(s, c)
    assert set(scene.mobjects) == {s, c}
    assert set(scene.camera.fixed_in_frame_mobjects) == {s, c}
    scene.remove_fixed_in_frame_mobjects(s)
    assert set(scene.mobjects) == {s, c}
    assert set(scene.camera.fixed_in_frame_mobjects) == {c}
    scene.add_fixed_orientation_mobjects(s)
    assert set(scene.camera.fixed_orientation_mobjects) == {s}
    scene.remove_fixed_orientation_mobjects(s)
    assert len(scene.camera.fixed_orientation_mobjects) == 0
