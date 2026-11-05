import pytest

from manim import DEGREES, Camera, Circle, Square, ThreeDScene, tempconfig


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


def test_3d_scenes_need_a_camera_with_an_orientation():
    with pytest.raises(TypeError, match="Expected a ThreeDCamera"):
        ThreeDScene(camera_class=Camera)


@pytest.mark.parametrize("shown", [False, True])
def test_move_camera_keeps_the_frame_only_if_it_was_shown(shown):
    with tempconfig({"dry_run": True}):
        scene = ThreeDScene()
        if shown:
            scene.add(scene.camera.frame)
        scene.move_camera(frame_center=(1, 0, 0), run_time=0.1)
        assert (scene.camera.frame in scene.mobjects) == shown
