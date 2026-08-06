"""Orchestration for rendering a scene."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import config, logger
from .utils.exceptions import EndSceneEarlyException, RerunSceneException
from .utils.file_ops import open_media_file

if TYPE_CHECKING:
    from .animation.animation import Animation
    from .mobject.mobject import Mobject, _AnimationBuilder
    from .scene.scene import Scene

__all__ = ["Manager"]


class Manager:
    """Coordinate the lifecycle and renderer calls for a single scene.

    This is deliberately a concrete orchestration object.  Renderer, camera, and
    output ownership remain on :class:`~manim.scene.scene.Scene` and its renderer
    for now.
    """

    def __init__(self, scene: Scene) -> None:
        if scene.manager is not None:
            raise ValueError("A manager is already attached to this scene.")
        self.scene = scene
        scene.manager = self

    def render(self, preview: bool = False) -> bool:
        """Render the manager's scene.

        Returns ``True`` when an interactive rerun was requested, matching the
        historical :meth:`Scene.render` return value.
        """
        scene = self.scene
        scene.setup()
        try:
            scene.construct()
        except EndSceneEarlyException:
            pass
        except RerunSceneException:
            scene.remove(*scene.mobjects)
            # TODO: The CairoRenderer does not have the method clear_screen().
            scene.renderer.clear_screen()  # type: ignore[union-attr]
            scene.renderer.num_plays = 0
            return True
        scene.tear_down()
        # We have to reset these settings in case of multiple renders.
        scene.renderer.scene_finished(scene)

        # Show info only if animations are rendered or to get image.
        if (
            scene.renderer.num_plays
            or config["format"] == "png"
            or config["save_last_frame"]
        ):
            logger.info(
                f"Rendered {str(scene)}\nPlayed {scene.renderer.num_plays} animations",
            )

        # If preview open up the render after rendering.
        if preview:
            config["preview"] = True

        if config["preview"] or config["show_in_file_browser"]:
            open_media_file(scene.renderer.file_writer)

        return False

    def play(
        self,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        """Forward an animation request to the scene's current renderer."""
        self.scene.renderer.play(self.scene, *args, **kwargs)
