from manim import DEGREES, Circle, Square, ThreeDScene


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


def test_set_to_default_angled_camera_orientation():
    scene = ThreeDScene()

    scene.set_to_default_angled_camera_orientation(phi=45 * DEGREES)

    assert scene.camera.get_phi() == 45 * DEGREES
    assert scene.camera.get_theta() == -135 * DEGREES
