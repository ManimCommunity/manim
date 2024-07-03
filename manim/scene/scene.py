from __future__ import annotations

import random
from collections import OrderedDict, deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from pyglet.window import key

from manim import config, logger
from manim.animation.animation import prepare_animation
from manim.camera.camera import Camera
from manim.constants import DEFAULT_WAIT_TIME
from manim.event_handler import EVENT_DISPATCHER
from manim.event_handler.event_type import EventType
from manim.mobject.mobject import Group, Point, _AnimationBuilder
from manim.mobject.opengl.opengl_mobject import OpenGLMobject as Mobject
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.exceptions import EndSceneEarlyException
from manim.utils.iterables import list_difference_update

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Callable

    from manim.animation.protocol import AnimationProtocol as Animation
    from manim.animation.scene_buffer import SceneBuffer
    from manim.manager import Manager

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

    skip_animations: bool = False
    always_update_mobjects: bool = False
    start_at_animation_number: int = 0
    end_at_animation_number: int | None = None
    presenter_mode: bool = False
    embed_exception_mode: str = ""
    embed_error_sound: bool = False

    def __init__(self, manager: Manager):
        # Core state of the scene
        self.camera: Camera = Camera()
        self.manager = manager
        self.mobjects: list[Mobject] = []
        self.num_plays: int = 0
        # the time is updated by the manager
        self.time: float = 0
        self.original_skipping_status: bool = self.skip_animations
        self.undo_stack: deque[SceneState] = deque()
        self.redo_stack: list[SceneState] = []

        if self.start_at_animation_number is not None:
            self.skip_animations = True

        # Items associated with interaction
        self.mouse_point = Point()
        self.mouse_drag_point = Point()
        self.hold_on_wait = self.presenter_mode
        self.quit_interaction = False

        # Much nicer to work with deterministic scenes
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

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
        self.remove(*buffer.to_remove)
        for to_replace_pairs in buffer.to_replace:
            self.replace(*to_replace_pairs)
        self.add(*buffer.to_add)
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
        raise RuntimeError("Could not find the construct method, did you misspell it?")

    def tear_down(self) -> None:
        """
        This method is used to clean up scenes
        """

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
        return True
        # wait_animation = self.animations[0]
        # if wait_animation.is_static_wait is None:
        #     should_update = (
        #         self.always_update_mobjects
        #         or self.updaters
        #         or wait_animation.stop_condition is not None
        #         or any(
        #             mob.has_time_based_updater()
        #             for mob in self.get_mobject_family_members()
        #         )
        #     )
        #     wait_animation.is_static_wait = not should_update
        # return not wait_animation.is_static_wait

    def has_time_based_updaters(self) -> bool:
        return any(
            sm.has_time_based_updater()
            for mob in self.mobjects
            for sm in mob.get_family()
        )

    # Related to internal mobject organization

    def add(self, *new_mobjects: Mobject):
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.
        """
        self.remove(*new_mobjects)
        self.mobjects += new_mobjects
        return self

    def remove(self, *mobjects_to_remove: Mobject):
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

    def replace(self, mobject: Mobject, *replacements: Mobject):
        """Replace one mobject in the scene with another, preserving draw order.

        If ``old_mobject`` is a submobject of some other Mobject (e.g. a
        :class:`.Group`), the new_mobject will replace it inside the group,
        without otherwise changing the parent mobject.

        Parameters
        ----------
        old_mobject
            The mobject to be replaced. Must be present in the scene.
        new_mobject
            A mobject which must not already be in the scene.

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

    def bring_to_front(self, *mobjects: Mobject):
        self.add(*mobjects)
        return self

    def bring_to_back(self, *mobjects: Mobject):
        self.remove(*mobjects)
        self.mobjects = [*mobjects, *self.mobjects]
        return self

    def clear(self):
        self.mobjects.clear()
        return self

    def get_mobjects(self) -> Sequence[Mobject]:
        return list(self.mobjects)

    def get_mobject_copies(self) -> Sequence[Mobject]:
        return [m.copy() for m in self.mobjects]

    def point_to_mobject(
        self,
        point: np.ndarray,
        search_set: Iterable[Mobject] | None = None,
        buff: float = 0,
    ) -> Mobject | None:
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

    def update_skipping_status(self) -> None:
        if (
            self.start_at_animation_number is not None
            and self.num_plays == self.start_at_animation_number
            and not self.original_skipping_status
        ):
            self.skip_animations = False
        if (
            self.end_at_animation_number is not None
            and self.num_plays >= self.end_at_animation_number
        ):
            raise EndSceneEarlyException()

    # Methods associated with running animations
    def pre_play(self):
        """To be implemented in subclasses."""

    def post_play(self):
        self.num_plays += 1

    def begin_animations(self, animations: Iterable[Animation]) -> None:
        for animation in animations:
            animation.begin()
            self.process_buffer(animation.buffer)

    def _update_animations(self, animations: Iterable[Animation], t: float, dt: float):
        for animation in animations:
            animation.update_mobjects(dt)
            alpha = t / animation.get_run_time()
            animation.interpolate(alpha)
            if animation.apply_buffer:
                self.process_buffer(animation.buffer)
                animation.apply_buffer = False

    def finish_animations(self, animations: Iterable[Animation]) -> None:
        for animation in animations:
            animation.finish()
            self.process_buffer(animation.buffer)

        if self.skip_animations:
            self._update_mobjects(self.manager._calc_runtime(animations))
        else:
            self._update_mobjects(0)

    def play(
        self,
        *proto_animations: Animation | _AnimationBuilder,
        run_time: float | None = None,
        rate_func: Callable[[float], float] | None = None,
        lag_ratio: float | None = None,
    ) -> None:
        if len(proto_animations) == 0:
            logger.warning("Called Scene.play with no animations")
            return
        animations = [prepare_animation(x) for x in proto_animations]
        for anim in animations:
            anim.update_rate_info(run_time, rate_func, lag_ratio)

        # NOTE: Should be changed at some point with the 2 pass rendering system 21.06.2024
        self.manager._play(*animations, run_time=run_time)

    def wait(
        self,
        duration: float = DEFAULT_WAIT_TIME,
        stop_condition: Callable[[], bool] | None = None,
        note: str | None = None,
        ignore_presenter_mode: bool = False,
    ):
        self.manager._wait(duration, stop_condition=stop_condition)
        # if (
        #     self.presenter_mode
        #     and not self.skip_animations
        #     and not ignore_presenter_mode
        # ):
        #     if note:
        #         logger.info(note)
        #     self.hold_loop()

    def wait_until(self, stop_condition: Callable[[], bool], max_time: float = 60):
        self.wait(max_time, stop_condition=stop_condition)

    def force_skipping(self):
        self.original_skipping_status = self.skip_animations
        self.skip_animations = True
        return self

    def revert_to_original_skipping_status(self):
        if hasattr(self, "original_skipping_status"):
            self.skip_animations = self.original_skipping_status
        return self

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        gain_to_background: float | None = None,
    ):
        if self.skip_animations:
            return
        time = self.time + time_offset
        self.file_writer.add_sound(sound_file, time, gain, gain_to_background)

    # Helpers for interactive development

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

    def on_mouse_motion(self, point: np.ndarray, d_point: np.ndarray) -> None:
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
        self, point: np.ndarray, d_point: np.ndarray, buttons: int, modifiers: int
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

    def on_mouse_press(self, point: np.ndarray, button: int, mods: int) -> None:
        self.mouse_drag_point.move_to(point)
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.MousePressEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_release(self, point: np.ndarray, button: int, mods: int) -> None:
        event_data = {"point": point, "button": button, "mods": mods}
        propagate_event = EVENT_DISPATCHER.dispatch(
            EventType.MouseReleaseEvent, **event_data
        )
        if propagate_event is not None and propagate_event is False:
            return

    def on_mouse_scroll(self, point: np.ndarray, offset: np.ndarray) -> None:
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
    def __init__(self, scene: Scene, ignore: list[Mobject] | None = None) -> None:
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
    def mobjects(self) -> Sequence[Mobject]:
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
