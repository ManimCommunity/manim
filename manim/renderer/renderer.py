from abc import ABC, abstractmethod

import typing

from manim.mobject.mobject import Mobject
from manim.utils.decorators import optionalmethod


class Renderer(ABC):
    def __init__(self, camera=None):
        self.camera = camera

    @abstractmethod
    def init_scene(self):
        pass

    @abstractmethod
    def before_animation(self):
        pass

    @abstractmethod
    def after_animation(self):
        pass

    @abstractmethod
    def before_render(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def after_render(self):
        pass

    @abstractmethod
    def after_scene(self):
        pass

    @abstractmethod
    def update_frame(self, moving_mobjects, skip_animations, mobjects, foreground_mobjects, file_writer, meshes):
        pass

    @abstractmethod
    def save_static_frame_data(
            self,
            static_mobjects: typing.Iterable[Mobject],
            mobjects=None,
            foreground_mobjects=None):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def has_interaction(self):
        pass

    @abstractmethod
    def use_z_index(self):
        pass

    @abstractmethod
    def can_handle_static_wait(self):
        pass

    @abstractmethod
    def freeze_current_frame(self, duration, file_writer, skip_animations):
        pass

    @abstractmethod
    def should_save_last_frame(self, num_plays):
        pass

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def get_current_time(self):
        pass

    @optionalmethod
    def interactive_embed(self):
        pass

    @optionalmethod
    def on_mouse_motion(self, point, d_point):
        pass

    @optionalmethod
    def on_mouse_scroll(self, point, offset):
        pass

    @optionalmethod
    def on_key_press(self, symbol, modifiers):
        pass

    @optionalmethod
    def on_key_release(self, symbol, modifiers):
        pass

    @optionalmethod
    def on_mouse_drag(self, point, d_point, buttons, modifiers):
        pass

    @optionalmethod
    def mouse_scroll_orbit_controls(self, point, offset):
        pass

    @optionalmethod
    def mouse_drag_orbit_controls(self, point, d_point, buttons, modifiers):
        pass

    @optionalmethod
    def set_key_function(self, char, func):
        pass

    @optionalmethod
    def on_mouse_press(self, point, button, modifiers):
        pass