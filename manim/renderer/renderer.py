from abc import ABC, abstractmethod

import typing

from manim.mobject.mobject import Mobject
from manim.utils.decorators import optionalmethod


class Renderer(ABC):
    def __init__(self, camera=None):
        self.camera = camera
        self.interactive_mode = False

    @abstractmethod
    def init_scene(self):
        pass

    @abstractmethod
    def before_animation(self):
        pass

    @abstractmethod
    def render(self, mobjects, skip_animations=False, **kwargs):
        pass

    @abstractmethod
    def after_scene(self):
        pass

    @abstractmethod
    def update_frame(self, mobjects, **kwargs):
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
    def get_window(self):
        pass
