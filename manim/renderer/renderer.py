from abc import ABC, abstractclassmethod, abstractstaticmethod
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import TypeAlias

from manim import config
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.mobject.types.image_mobject import ImageMobject
from manim.mobject.types.vectorized_mobject import VMobject

ImageType: TypeAlias = np.ndarray


class RendererData:
    pass


class Renderer(ABC):
    def __init__(self):
        self.fbo = np.zeros((config.height, config.width))
        self.capabilities = {
            VMobject: self.render_vmobject,
            ImageMobject: self.render_image,
        }

    def render(self, camera, renderables: [VMobject]) -> ImageType:  # Image
        for mob in renderables:
            if type(mob) in self.capabilities:
                self.capabilities[type(mob)](mob)
            else:
                print("WARNING: NOT SUPPORTED")

        return self.fbo.get_pixels()

    @abstractclassmethod
    def render_vmobject(self, mob: OpenGLVMobject) -> None:
        raise NotImplementedError

    def render_mesh(self, mob) -> None:
        raise NotImplementedError

    def render_image(self, mob) -> None:
        raise NotImplementedError


# NOTE: The user should expect depth between renderers not to be handled discussed at 03.09.2023 Between jsonv and MrDiver
# NOTE: Cairo_camera overlay_PIL_image for MultiRenderer

# class Compositor:
#     def __init__(self):
#         self.renderers = []

#     def add_capability(self, renderer) -> None:
#         self.renderers.append(renderer)

#     def add(img1, img2):
#         raise NotImplementedError

#     def subtract(*images: List[Image]):
#         raise NotImplementedError

#     def mix():
#         raise NotImplementedError

#     def multiply():
#         raise NotImplementedError

#     def divide():
#         raise NotImplementedError


# class GraphScene(Scene):
#     def construct(self):
#         config.renderer =

# class VolumetricScene(Scene):
#     def construct(self):
#         pass

# compositor = Compositor()
# compositor.add_capability(GraphScene, OpenGL) # no file writing
# compositor.add_capability(VolumetricScene, Blender, ) # 3 sec
# compositor.addPostFX(CustomShader)
# compositor.render()
