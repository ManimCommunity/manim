from __future__ import annotations

import inspect
import random
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self, assert_never

import numpy as np
from pyglet.window import key

from manim import config, logger
from manim.animation.animation import prepare_animation
from manim.animation.animation import Wait
from manim.animation.scene_buffer import SceneBuffer, SceneOperation
from manim.camera.camera import Camera
from manim.constants import DEFAULT_WAIT_TIME
from manim.event_handler import EVENT_DISPATCHER
from manim.event_handler.event_type import EventType
from manim.mobject.mobject import Group, Point
from manim.mobject.opengl.opengl_mobject import OpenGLMobject, _AnimationBuilder
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.scene.sections import group as SceneGroup
from manim.utils.iterables import list_difference_update

if TYPE_CHECKING:
    from collections.abc import Iterable, Reversible, Sequence

    from manim.animation.protocol import AnimationProtocol
    from manim.manager import Manager
    from manim.typing import Point3D, Vector3D

# TODO: these keybindings should be made configurable

PAN_3D_KEY = "d"
FRAME_SHIFT_KEY = "f"
ZOOM_KEY = "z"
RESET_FRAME_KEY = "r"
QUIT_KEY = "q"


class Scene:
    """The Canvas of Manim.

    You can use it by putting the following into a
    file ``manimation.py``

    .. manim:: SceneWithSettings

        class SceneWithSettings(Scene):
            # set configuration attributes
            random_seed = 1

            # all the action happens here
            def construct(self):
                self.play(Create(ManimBanner()))

    And then run ``manim -p manimation.py``. To write the result to a file,
    do ``manim -w manimation.py``.

    Attributes
    ----------

        random_seed : The seed for random and numpy.random
        pan_sensitivity :
    """

    random_seed: int | None = None
    pan_sensitivity: float = 3.0
    max_num_saved_states: int = 50

    always_update_mobjects: bool = False
    start_at_animation_number: int = 0
    end_at_animation_number: int | None = None
    presenter_mode: bool = False
    embed_exception_mode: str = ""
    embed_error_sound: bool = False

    groups_api: bool = False

    def __init__(self, manager: Manager[Self]):
        # Core state of the scene
        self.camera: Camera = Camera()
        self.manager = manager
        self.mobjects: list[OpenGLMobject] = []
        self.num_plays: int = 0
        # the time is updated by the manager
        self.time: float = 0
        self.undo_stack: deque[SceneState] = deque()
        self.redo_stack: list[SceneState] = []

        # Items associated with interaction
        self.mouse_point = Point()
        self.mouse_drag_point = Point()
        self.hold_on_wait = self.presenter_mode
        self.quit_interaction = False

        # Much nicer to work with deterministic scenes
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.default_rng(self.random_seed)

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_default_scene_name(self) -> str:
        name = str(self)
        saan = self.start_at_animation_number
        eaan = self.end_at_animation_number
        if saan is not None:
            name += f"_{saan}"
        if eaan is not None:
            name += f"_{eaan}"
        return name

    def process_buffer(self, buffer: SceneBuffer) -> None:
        for op, args, kwargs in buffer:
            match op:
                case SceneOperation.ADD:
                    self.add(*args, **kwargs)
                case SceneOperation.REMOVE:
                    self.remove(*args, **kwargs)
                case SceneOperation.REPLACE:
                    self.replace(*args, **kwargs)
                case _:
                    assert_never(op)
        buffer.clear()

    def setup(self) -> None:
        """
        This method is used to set up scenes to do any setup
        involved before the construct method is called.
        """

    def construct(self) -> None:
        """
        The entrypoint to animations in Manim.
        Should be overridden in the subclass to produce animations
        """
        raise RuntimeError(
            "Could not find the construct method, did you misspell the name?"
        )

    def tear_down(self) -> None:
        """This method is used to clean up scenes"""

    def find_groups(self) -> list[SceneGroup]:
        """Find all groups in a :class:`.Scene`"""
        sections: list[SceneGroup] = [
            bound
            for _, bound in inspect.getmembers(
                self, predicate=lambda x: isinstance(x, SceneGroup)
            )
        ]
        sections.sort()
        return sections

    # Only these methods should touch the camera
    # Related to updating

    def _update_mobjects(self, dt: float) -> None:
        for mobject in self.mobjects:
            mobject.update(dt)

    def should_update_mobjects(self) -> bool:
        """
        This is called to check if a wait frame should be frozen

        Returns
        -------
            bool: does it have to be rerendered or is it static
        """
        # always rerender by returning True
        # TODO: Apply caching here
        return self.always_update_mobjects or any(
            mob.has_updaters for mob in self.mobjects
        )

    def is_current_animation_frozen_frame(self) -> bool:
        if not self.animations:
            return False
        
        current = self.animations[0]
        return getattr(current, 'is_static_wait', False)

    def has_time_based_updaters(self) -> bool:
        return any(
            sm.has_time_based_updater()
            for mob in self.mobjects
            for sm in mob.get_family()
        )

    # Related to internal mobject organization

    def add(self, *new_mobjects: OpenGLMobject) -> Self:
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.
        """
        self.remove(*new_mobjects)
        self.mobjects += new_mobjects
        return self

    def remove(self, *mobjects_to_remove: OpenGLMobject) -> Self:
        """
        Removes anything in mobjects from scenes mobject list, but in the event that one
        of the items to be removed is a member of the family of an item in mobject_list,
        the other family members are added back into the list.

        For example, if the scene includes Group(m1, m2, m3), and we call scene.remove(m1),
        the desired behavior is for the scene to then include m2 and m3 (ungrouped).
        """
        for mob in mobjects_to_remove:
            # First restructure self.mobjects so that parents/grandparents/etc. are replaced
            # with their children, likewise for all ancestors in the extended family.
            for ancestor in mob.get_ancestors(extended=True):
                self.replace(ancestor, *ancestor.submobjects)
            self.mobjects = list_difference_update(self.mobjects, mob.get_family())
        return self

    def replace(self, mobject: OpenGLMobject, *replacements: OpenGLMobject):
        """Replace one Mobject in the scene with one or more other Mobjects,
        preserving draw order.

        If ``mobject`` is a submobject of some other :class:`OpenGLMobject`
        (e.g. a :class:`.Group`), the ``replacements`` will replace it inside
        the group, without otherwise changing the parent mobject.

        Parameters
        ----------
        mobject
            The mobject to be replaced. Must be present in the scene.
        replacements
            One or more Mobjects which must not already be in the scene.

        """
        if mobject in self.mobjects:
            index = self.mobjects.index(mobject)
            self.mobjects = [
                *self.mobjects[:index],
                *replacements,
                *self.mobjects[index + 1 :],
            ]
        return self

    def add_updater(self, func: Callable[[float], None]) -> None:
        """Add an update function to the scene.

        The scene updater functions are run every frame,
        and they are the last type of updaters to run.

        .. WARNING::

            When using the Cairo renderer, scene updaters that
            modify mobjects are not detected in the same way
            that mobject updaters are. To be more concrete,
            a mobject only modified via a scene updater will
            not necessarily be added to the list of *moving
            mobjects* and thus might not be updated every frame.

            TL;DR: Use mobject updaters to update mobjects.

        Parameters
        ----------
        func
            The updater function. It takes a float, which is the
            time difference since the last update (usually equal
            to the frame rate).

        See also
        --------
        :meth:`.Scene.remove_updater`
        :meth:`.Scene.update_self`
        """
        self.updaters.append(func)

    def remove_updater(self, func: Callable[[float], None]) -> None:
        """Remove an update function from the scene.

        Parameters
        ----------
        func
            The updater function to be removed.

        See also
        --------
        :meth:`.Scene.add_updater`
        :meth:`.Scene.update_self`
        """
        self.updaters = [f for f in self.updaters if f is not func]

    def bring_to_front(self, *mobjects: OpenGLMobject) -> Self:
        self.add(*mobjects)
        return self

    def bring_to_back(self, *mobjects: OpenGLMobject) -> Self:
        self.remove(*mobjects)
        self.mobjects = [*mobjects, *self.mobjects]
        return self

    def clear(self) -> Self:
        self.mobjects.clear()
        return self

    def get_mobjects(self) -> Sequence[OpenGLMobject]:
        return list(self.mobjects)

    def get_mobject_copies(self) -> Sequence[OpenGLMobject]:
        return [m.copy() for m in self.mobjects]

    def point_to_mobject(
        self,
        point: Point3D,
        search_set: Reversible[OpenGLMobject] | None = None,
        buff: float = 0.0,
    ) -> OpenGLMobject | None:
        """
        E.g. if clicking on the scene, this returns the top layer mobject
        under a given point
        """
        if search_set is None:
            search_set = self.mobjects
        for mobject in reversed(search_set):
            if mobject.is_point_touching(point, buff=buff):
                return mobject
        return None

    def get_group(self, *mobjects):
        if all(isinstance(m, VMobject) for m in mobjects):
            return VGroup(*mobjects)
        else:
            return Group(*mobjects)

    # Related to skipping

    # Methods associated with running animations
    def pre_play(self) -> None:
        """To be implemented in subclasses."""

    def post_play(self) -> None:
        self.num_plays += 1

    def begin_animations(self, animations: Iterable[AnimationProtocol]) -> None:
        for animation in animations:
            animation.begin()
            self.process_buffer(animation.buffer)

    def _update_animations(
        self, animations: Iterable[AnimationProtocol], t: float, dt: float
    ) -> None:
        for animation in animations:
            animation.update_mobjects(dt)
            alpha = t / animation.get_run_time()
            animation.interpolate(alpha)
            if animation.apply_buffer:
                self.process_buffer(animation.buffer)
                animation.apply_buffer = False

    def finish_animations(self, animations: Iterable[AnimationProtocol]) -> None:
        for animation in animations:
            animation.finish()
            self.process_buffer(animation.buffer)

    @classmethod
    def validate_run_time(
        cls,
        run_time: float,
        method: Callable[[Any], Any],
        parameter_name: str = "run_time",
    ) -> float:
        method_name = f"{cls.__name__}.{method.__name__}()"
        if run_time <= 0:
            raise ValueError(
                f"{method_name} has a {parameter_name} of "
                f"{run_time:g} <= 0 seconds which Manim cannot render. "
                f"The {parameter_name} must be a positive number."
            )

        # config.frame_rate holds the number of frames per second
        fps = config.frame_rate
        seconds_per_frame = 1 / fps
        if run_time < seconds_per_frame:
            logger.warning(
                f"The original {parameter_name} of {method_name}, "
                f"{run_time:g} seconds, is too short for the current frame "
                f"rate of {fps:g} FPS. Rendering with the shortest possible "
                f"{parameter_name} of {seconds_per_frame:g} seconds instead."
            )
            run_time = seconds_per_frame

        return run_time

    def play(
        self,
        # the OpenGLMobject is a side-effect of the return type of animate, it will
        # raise a ValueError
        *proto_animations: AnimationProtocol
        | _AnimationBuilder[OpenGLMobject]
        | OpenGLMobject,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
    ) -> None:
        if len(proto_animations) == 0:
            logger.warning("Called Scene.play with no animations")
            return

        # Build _AnimationBuilders.
        animations = [prepare_animation(x) for x in proto_animations]
        for anim in animations:
            anim.update_rate_info(run_time, rate_func, lag_ratio)

        # Validate the final run_time.
        total_run_time = max(anim.get_run_time() for anim in animations)
        new_total_run_time = self.validate_run_time(
            total_run_time, self.play, "total run_time"
        )
        if new_total_run_time != total_run_time:
            for anim in animations:
                anim.update_rate_info(new_total_run_time)
        self.animations = animations
        # NOTE: Should be changed at some point with the 2 pass rendering system 21.06.2024
        self.manager._play(*animations)

    def wait(
        self,
        duration: float = DEFAULT_WAIT_TIME,
        stop_condition: Callable[[], bool] | None = None,
        frozen_frame: bool | None = None,
        note: str | None = None,
        ignore_presenter_mode: bool = False,
    ) -> None:
        duration = self.validate_run_time(duration, self.wait, "duration")
        if frozen_frame is None:
            frozen_frame = not self.should_update_mobjects()
        self.play(Wait(duration, stop_condition=stop_condition, frozen_frame=frozen_frame))
        # if (
        #     self.presenter_mode
        #     and not self.skip_animations
        #     and not ignore_presenter_mode
        # ):
        #     if note:
        #         logger.info(note)
        #     self.hold_loop()

    def wait_until(self, stop_condition: Callable[[], bool], max_time: float = 60):
        max_time = self.validate_run_time(max_time, self.wait_until, "max_time")
        self.wait(max_time, stop_condition=stop_condition)

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        gain_to_background: float | None = None,
    ):
        raise NotImplementedError("TODO")
        time = self.time + time_offset
        self.file_writer.add_sound(sound_file, time, gain, gain_to_background)

    def get_state(self) -> SceneState:
        return SceneState(self)

    def restore_state(self, scene_state: SceneState):
        scene_state.restore_scene(self)

    def save_state(self) -> None:
        if not config.preview:
            return
        state = self.get_state()
        if self.undo_stack and state.mobjects_match(self.undo_stack[-1]):
            return
        self.redo_stack = []
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_num_saved_states:
            self.undo_stack.popleft()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.get_state())
            self.restore_state(self.undo_stack.pop())

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.get_state())
            self.restore_state(self.redo_stack.pop())

    # TODO: reimplement checkpoint feature with CE's section API
    # Event handling

    def on_mouse_motion(self, point: Point3D, d_point: Vector3D) -> None:
        self.mouse_point.move_to(point)

        event_data = {"point": point, "d_point": d_point}
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.MouseMotionEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

        # TODO
        return
        frame = self.camera.frame
        # Handle perspective changes
        if self.window.is_key_pressed(ord(PAN_3D_KEY)):
            frame.increment_theta(-self.pan_sensitivity * d_point[0])
            frame.increment_phi(self.pan_sensitivity * d_point[1])
        # Handle frame movements
        elif self.window.is_key_pressed(ord(FRAME_SHIFT_KEY)):
            shift = -d_point
            shift[0] *= frame.get_width() / 2
            shift[1] *= frame.get_height() / 2
            transform = frame.get_inverse_camera_rotation_matrix()
            shift = np.dot(np.transpose(transform), shift)
            frame.shift(shift)

    def on_mouse_drag(
        self, point: Point3D, d_point: Vector3D, buttons: int, modifiers: int
    ) -> None:
        self.mouse_drag_point.move_to(point)

        event_data = {
            "point": point,
            "d_point": d_point,
            "buttons": buttons,
            "modifiers": modifiers,
        }
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.MouseDragEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_press(self, point: Point3D, button: int, mods: int) -> None:
        self.mouse_drag_point.move_to(point)
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.MousePressEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_release(self, point: Point3D, button: int, mods: int) -> None:
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.MouseReleaseEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_scroll(self, point: Point3D, offset: Vector3D) -> None:
        event_data = {"point": point, "offset": offset}
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.MouseScrollEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

        frame = self.camera.frame
        if self.window.is_key_pressed(ord(ZOOM_KEY)):
            factor = 1 + np.arctan(10 * offset[1])
            frame.scale(1 / factor, about_point=point)
        else:
            transform = frame.get_inverse_camera_rotation_matrix()
            shift = np.dot(np.transpose(transform), offset)
            frame.shift(-20.0 * shift)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        event_data = {"symbol": symbol, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.KeyReleaseEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        try:
            char = chr(symbol)
        except OverflowError:
            logger.warning("The value of the pressed key is too large.")
            return

        event_data = {"symbol": symbol, "modifiers": modifiers}
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.KeyPressEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

        if char == RESET_FRAME_KEY:
            self.play(self.camera.frame.animate.to_default_state())
        elif char == "z" and modifiers == key.MOD_COMMAND:
            self.undo()
        elif char == "z" and modifiers == key.MOD_COMMAND | key.MOD_SHIFT:
            self.redo()
        # command + q or esc
        elif (char == QUIT_KEY and modifiers == key.MOD_COMMAND) or char == key.ESCAPE:
            self.quit_interaction = True
        # Space or right arrow
        elif char == " " or symbol == key.RIGHT:
            self.hold_on_wait = False

    def on_resize(self, width: int, height: int) -> None:
        pass

    def on_show(self) -> None:
        pass

    def on_hide(self) -> None:
        pass

    def on_close(self) -> None:
        pass


class SceneState:
    def __init__(
        self, scene: Scene, ignore: Iterable[OpenGLMobject] | None = None
    ) -> None:
        self.time = scene.time
        self.num_plays = scene.num_plays
        self.camera = scene.camera.copy()
        self.mobjects_to_copies = OrderedDict.fromkeys(scene.mobjects)
        if ignore:
            for mob in ignore:
                self.mobjects_to_copies.pop(mob, None)

        last_m2c = scene.undo_stack[-1].mobjects_to_copies if scene.undo_stack else {}
        for mob in self.mobjects_to_copies:
            # If it hasn't changed since the last state, just point to the
            # same copy as before
            if mob in last_m2c and last_m2c[mob].looks_identical(mob):
                self.mobjects_to_copies[mob] = last_m2c[mob]
            else:
                self.mobjects_to_copies[mob] = mob.copy()

    @property
    def mobjects(self) -> Sequence[OpenGLMobject]:
        return tuple(self.mobjects_to_copies.keys())

    def __eq__(self, state: Any) -> bool:
        return isinstance(state, SceneState) and all(
            (
                self.time == state.time,
                self.num_plays == state.num_plays,
                self.mobjects_to_copies == state.mobjects_to_copies,
            )
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__} of {len(self.mobjects_to_copies)} Mobjects"

    def mobjects_match(self, state: SceneState):
        return self.mobjects_to_copies == state.mobjects_to_copies

    def n_changes(self, state: SceneState):
        m2c = state.mobjects_to_copies
        return sum(
            1 - int(mob in m2c and mob.looks_identical(m2c[mob]))
            for mob in self.mobjects_to_copies
        )

    def restore_scene(self, scene: Scene):
        scene.time = self.time
        scene.num_plays = self.num_plays
        scene.mobjects = [
            mob.become(mob_copy) for mob, mob_copy in self.mobjects_to_copies.items()
        ]
