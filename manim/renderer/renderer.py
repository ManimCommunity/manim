from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

import numpy as np
from typing_extensions import TypeAlias

from manim._config import logger
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.mobject.types.image_mobject import ImageMobject

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import moderngl as gl

    from manim.camera.camera import Camera

ImageType: TypeAlias = np.ndarray


class RendererData:
    pass


class Renderer(ABC):
    def __init__(self):
        self.capabilities = [
            (OpenGLVMobject, self.render_vmobject),  # type: ignore
            (ImageMobject, self.render_image),  # type: ignore
        ]

    def render(self, camera, renderables: Iterable[OpenGLMobject]) -> None:  # Image
        self.pre_render(camera)
        for mob in renderables:
            for type_, render_func in self.capabilities:
                if isinstance(mob, type_):
                    render_func(mob)
                    break
            else:
                logger.warn(
                    f"The type{type(mob)} is not supported in Renderer: {self.__class__}"
                )
        self.post_render()

    @abstractmethod
    def pre_render(self, camera: Camera):
        raise NotImplementedError

    @abstractmethod
    def post_render(self):
        raise NotImplementedError

    @abstractmethod
    def render_vmobject(self, mob: OpenGLVMobject):
        raise NotImplementedError

    @abstractmethod
    def render_image(self, mob: ImageMobject):
        raise NotImplementedError


class RendererProtocol(Protocol):
    capabilities: Sequence[
        tuple[type[OpenGLMobject], Callable[[type[OpenGLMobject]], object]]
    ]

    def render(self, camera: Camera, renderables: Iterable[OpenGLMobject]) -> None:
        ...

    def render_previous(self, camera: Camera) -> None:
        ...

    def pre_render(self, camera) -> object:
        ...

    def post_render(self) -> object:
        ...

    def use_window(self):
        ...

    def render_vmobject(self, mob: OpenGLVMobject) -> object:
        ...

    def render_mesh(self, mob) -> None:
        ...

    def render_image(self, mob: ImageMobject) -> None:
        ...

    def get_pixels(self) -> ImageType:
        ...


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
