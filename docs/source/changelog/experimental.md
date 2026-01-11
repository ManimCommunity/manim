# Migrating from v0.19.0 to v0.20.0

This constitutes a list of all the changes needed to migrate your code
to work with the latest version of Manim

## Manager
If you ever used `Scene.render`, you must replace it with {class}`.Manager`.

Original code:
```py
scene = SceneClass()
scene.render()
```
should be changed to:
```py
manager = Manager(SceneClass)
manager.render()
```

If you are a plugin author that subclasses `Scene` and changed `Scene.render`, you should migrate
your code to use the specific public methods on {class}`.Manager` instead.

## ThreeDScene and Camera
`ThreeDScene` has been completely removed, and all of its functionality has been replaced
with methods on {class}`.Camera`, which can be accessed via {attr}`.Scene.camera`.

For example, the following code
```py
class MyScene(ThreeDScene):
    def construct(self):
        t = Text("Hello")
        self.add_fixed_in_frame_mobjects(t)
        self.begin_ambient_camera_rotation()
        self.wait(3)
```
should be changed to
```py
# change ThreeDScene -> Scene
class MyScene(Scene):
    def construct(self):
        t = Text("Hello")
        # add_fixed_in_frame_mobjects() no longer exists.
        # Now you must use Mobject.fix_in_frame() manually for each Mobject.
        t.fix_in_frame()
        self.add(t)

        # access the method on the camera
        self.camera.begin_ambient_rotation()
        self.add(self.camera)
        self.wait(3)
```

## Animation
`Animation.interpolate_mobject` has been combined into `Animation.interpolate`.

Methods `Animation._setup_scene` and `Animation.clean_up_from_scene` have been removed
in favor of `Animation.begin` and `Animation.finish`. If you need to access the scene,
you can use a simple buffer to communicate. Note that this buffer cannot access
methods on the {class}`.Scene`, but can only do basic actions like {meth}`.Scene.add`,
{meth}`.Scene.remove`, and {meth}`.Scene.replace`.

For example, the following code:
```py
class MyAnimation(Animation):
    def begin(self) -> None:
        self._sqrs = VGroup(Square())

    def _setup_scene(self, scene: Scene) -> None:
        scene.add(self._sqrs)
        self.scene = scene

    def interpolate_mobject(self, alpha: float) -> None:
        sqr = Square().move_to((alpha, 0, 0))
        self._sqrs.add(sqr)
        self.scene.add(sqr)

    def clean_up_from_scene(self, scene: Scene) -> None:
        scene.remove(self._sqrs)
```

should be changed to
```py
class MyAnimation(Animation):
    def begin(self) -> None:
        self._sqrs = VGroup(Square())
        self.buffer.add(self._sqrs)

    def interpolate(self, alpha: float) -> None:
        sqr = Square().move_to((alpha, 0, 0))
        self._sqrs.add(sqr)
        self.buffer.add(sqr)
        # tell the scene to empty the buffer
        self.apply_buffer = True

    def finish(self) -> None:
        self.buffer.remove(self._sqrs)
```
