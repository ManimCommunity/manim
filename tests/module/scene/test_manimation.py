from manim._config import tempconfig
from manim.mobject.geometry.arc import Circle
from manim.scene.moving_camera_scene import MovingCameraScene
from manim.scene.scene import REGISTERED_MANIMATIONS, Scene, manimation


def test_cli_registry_manimation():
    @manimation
    def MyAnimation(self: Scene):
        self.add(Circle())

    assert MyAnimation.__class__ in REGISTERED_MANIMATIONS


def test_manimation_is_instance():
    @manimation
    def MyAnimation(self: Scene):
        self.add(Circle())

    assert isinstance(MyAnimation, Scene)
    assert issubclass(MyAnimation.__class__, Scene)


def test_manimation_render():
    @manimation
    def MyAnimation(self: Scene):
        self.add(Circle())

    with tempconfig({"write_to_movie": False}):
        MyAnimation.render()


def test_manimation_parameters():
    @manimation(scene_class=MovingCameraScene)
    def MyAnimation(self: MovingCameraScene):
        self.add(Circle())

    assert isinstance(MyAnimation, MovingCameraScene)
    assert issubclass(MyAnimation.__class__, Scene)
