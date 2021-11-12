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

    @optionalmethod
    def interactive_embed(self):
        pass
