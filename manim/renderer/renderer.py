from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from manim._config import logger
from manim.mobject.mobject import InvisibleMobject
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.mobject.types.image_mobject import ImageMobject

if TYPE_CHECKING:
    from manim.camera.camera import Camera
    from manim.scene.scene import SceneState
    from manim.typing import PixelArray


class RendererData:
    pass


class Renderer(ABC):
    """Abstract class that handles dispatching mobjects to their specialized mobjects.

    Specifically, it maps :class:`.OpenGLVMobject` to :meth:`render_vmobject`, :class:`.ImageMobject`
    to :meth:`render_image`, etc.
    """

    def __init__(self):
        self.capabilities = [
            (OpenGLVMobject, self.render_vmobject),
            (ImageMobject, self.render_image),
        ]
        self._unsupported_mob_warnings: set[str] = set()

    def render(self, state: SceneState) -> None:
        self.pre_render(state.camera)
        for mob in state.mobjects:
            self.render_mobject(mob)
        self.post_render()

    def render_mobject(self, mob: OpenGLMobject) -> None:
        for MobClass, render_func in self.capabilities:
            if isinstance(mob, MobClass):
                render_func(mob)
                break
        else:
            if not isinstance(mob, InvisibleMobject):
                warning_message = (
                    f"{type(mob).__name__} cannot be rendered by {type(self).__name__}. "
                    "Attempting to render its submobjects..."
                )
                # This prevents spamming the console.
                if warning_message not in self._unsupported_mob_warnings:
                    logger.warning(warning_message)
                    self._unsupported_mob_warnings.add(warning_message)
            for submob in mob.submobjects:
                self.render_mobject(submob)

    @abstractmethod
    def pre_render(self, camera: Camera) -> None:
        """Actions before rendering any :class:`.OpenGLMobject`"""

    @abstractmethod
    def post_render(self) -> None:
        """Actions before rendering any :class:`.OpenGLMobject`"""

    @abstractmethod
    def render_vmobject(self, mob: OpenGLVMobject) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_image(self, mob: ImageMobject) -> None:
        raise NotImplementedError


# Note: runtime checking is slow,
# but it only happens once or twice so it should be fine
@runtime_checkable
class RendererProtocol(Protocol):
    """The Protocol a renderer must implement to be used in :class:`.Manager`."""

    def render(self, state: SceneState) -> None:
        """Render a group of Mobjects"""
        ...

    def use_window(self) -> None:
        """Hook called after instantiation."""
        ...

    def get_pixels(self) -> PixelArray:
        """Get the pixels that should be written to a file."""
        ...

    def release(self) -> None:
        """Release any resources the renderer is holding."""
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

#     def subtract(*images: List[PixelArray]):
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
