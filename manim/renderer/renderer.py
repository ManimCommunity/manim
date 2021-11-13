from abc import ABC, abstractmethod

from manim.utils.decorators import optionalmethod


class Renderer(ABC):
    def __init__(self):
        pass

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
    def get_frame(self):
        pass

    @abstractmethod
    def has_interaction(self):
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
