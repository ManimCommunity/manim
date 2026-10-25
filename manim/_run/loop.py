"""Run play and wait calls frame by frame and report each step to listeners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL.Image import Image

    from ..animation.animation import Animation
    from ..manager import Manager
    from ..mobject.mobject import Mobject, _AnimationBuilder
    from ..typing import RGBAPixelArray
    from .selection import _OutputSelection

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _Play:
    """A play or wait call, as its frames start."""

    index: int
    """0 for the first play or wait call of the scene, 1 for the next, and so on."""
    first_frame: int
    """The index of the play's first frame in the scene's frame sequence."""
    run_time: float
    """The requested run time in seconds."""
    animations: tuple[Animation, ...]
    frozen: bool
    """Whether the play is a frozen wait: one frame shown for its whole duration."""
    in_output: bool
    """Whether the play is part of the rendered output."""


@dataclass(frozen=True, slots=True)
class _Section:
    name: str
    type: str
    skip_animations: bool


@dataclass(frozen=True, slots=True)
class _Caption:
    content: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class _Sound:
    file: str
    start: float
    gain: float | None
    options: dict[str, Any] = field(default_factory=dict)
    in_output: bool = True
    """Whether the play that starts at the sound's call is part of the output."""


class _Frame:
    """One frame of a play, drawn only when a listener asks for its image.

    All listeners of a frame share one drawing.
    """

    __slots__ = ("_drawn", "_loop", "_pixels", "index", "offset", "play", "repeat")

    def __init__(
        self, loop: _RunLoop, play: _Play, index: int, repeat: int, offset: float
    ) -> None:
        self._loop = loop
        self.play = play
        self.index = index
        """The index of the frame in the scene's frame sequence."""
        self.repeat = repeat
        """How many frames show this image: more than 1 only for a frozen wait."""
        self.offset = offset
        """The time of the frame in seconds, measured from the start of the play."""
        self._drawn = False
        self._pixels: RGBAPixelArray | None = None

    def draw(self) -> None:
        """Draw the frame into the renderer's target, unless it is drawn already."""
        if not self._drawn:
            self._loop.drawer.draw(self)
            self._drawn = True

    def pixels(self) -> RGBAPixelArray:
        """Return the frame as an RGBA pixel array, drawing it if needed."""
        if self._pixels is None:
            self.draw()
            self._pixels = self._loop.drawer.pixels(self)
        return self._pixels

    def image(self) -> Image:
        """Return the frame as a PIL image, drawing it if needed."""
        from PIL import Image

        return Image.fromarray(self.pixels())


class _RunListener:
    """Something that reacts to what happens during a run.

    Listeners must not change the scene's mobjects or call back into the loop.
    They run in a fixed order. The encoder runs before the preview and the
    progress bar and marks a cached play by setting ``skip_animations``, which
    the listeners after it read. Every method does nothing unless overridden.
    """

    def play_started(self, play: _Play) -> None:
        """A play's animations are prepared but have not begun yet."""

    def frame(self, frame: _Frame) -> None:
        """The scene shows the state of ``frame``; time is at the frame's start."""

    def play_finished(self, play: _Play, end_frame: int) -> None:
        """The play's animations have finished and were cleaned up."""

    def marker(self, marker: _Section | _Caption | _Sound) -> None:
        """A section, caption, or sound call was made."""


class _Drawer(Protocol):
    def draw(self, frame: _Frame) -> None: ...

    def pixels(self, frame: _Frame) -> RGBAPixelArray: ...

    def end_play(self, play: _Play) -> None: ...


class _TargetDrawer:
    """Draw frames into the renderer's own target, as a render does."""

    def __init__(self, manager: Manager) -> None:
        self._manager = manager
        self._prepared = False

    def draw(self, frame: _Frame) -> None:
        renderer = self._manager.renderer
        scene = self._manager.scene
        if not self._prepared:
            # Drawing resources and Cairo's background of still mobjects, once
            # per play and only for plays that draw.
            renderer._start_animation()
            renderer._prepare_animation(scene)
            self._prepared = True
        renderer.render(scene, frame.offset, scene.moving_mobjects)

    def pixels(self, frame: _Frame) -> RGBAPixelArray:
        return self._manager.renderer.get_frame()

    def end_play(self, play: _Play) -> None:
        if self._prepared and not play.frozen:
            # After a frozen wait, the background of still mobjects stays, so that
            # a frame drawn between plays (see Manager._render_preview_frame) shows
            # the still mobjects too. The next play draws a new background.
            self._manager.renderer.static_image = None  # type: ignore[union-attr]
        self._prepared = False


class _SceneImageDrawer:
    """Draw frames with a temporary renderer, as :meth:`.Scene.get_image` does.

    Used during evaluation, which opens no drawing resources of its own.
    """

    def __init__(self, manager: Manager) -> None:
        self._manager = manager
        self._pixels: RGBAPixelArray | None = None

    def draw(self, frame: _Frame) -> None:
        image = self._manager.renderer._get_scene_image(self._manager.scene)
        self._pixels = np.asarray(image)

    def pixels(self, frame: _Frame) -> RGBAPixelArray:
        assert self._pixels is not None
        pixels, self._pixels = self._pixels, None
        return pixels

    def end_play(self, play: _Play) -> None:
        pass


class _RunLoop:
    """Run play and wait calls frame by frame.

    Every play steps through all of its frames, also when it is left out of the
    output or reused from the cache: the scene always reaches the same state as
    in a full render. Whether a frame is drawn is up to the listeners.
    """

    def __init__(
        self,
        manager: Manager,
        *,
        listeners: Sequence[_RunListener],
        selection: _OutputSelection,
        drawer: _Drawer,
    ) -> None:
        self.manager = manager
        self.listeners = tuple(listeners)
        self.selection = selection
        self.drawer = drawer
        self._playing = False

    def play(
        self, *args: Animation | Mobject | _AnimationBuilder, **kwargs: Any
    ) -> None:
        if self._playing:
            raise RuntimeError(
                "play() and wait() cannot be called during another play or wait "
                "call, for example from an updater."
            )
        self._playing = True
        try:
            self._play(*args, **kwargs)
        finally:
            self._playing = False

    def _play(
        self, *args: Animation | Mobject | _AnimationBuilder, **kwargs: Any
    ) -> None:
        manager = self.manager
        scene = manager.scene
        state = manager._execution
        index = manager.num_plays
        in_output = self.selection.start_play(index)
        manager.skip_animations = not in_output
        scene.compile_animation_data(*args, **kwargs)
        assert scene.animations is not None
        play = _Play(
            index=index,
            first_frame=state.frame,
            run_time=scene.duration,
            animations=tuple(scene.animations),
            frozen=scene.is_current_animation_frozen_frame(),
            in_output=in_output,
        )
        for listener in self.listeners:
            listener.play_started(play)
        scene.begin_animations()
        try:
            if play.frozen:
                repeat = int(scene.duration * manager.session_spec.frame_rate)
                self._frame(play, state.frame, repeat, 0.0)
                state.frame += repeat
            else:
                self._step(play)
        finally:
            self.drawer.end_play(play)
        for listener in self.listeners:
            listener.play_finished(play, state.frame)
        manager.num_plays += 1

    def _step(self, play: _Play) -> None:
        scene = self.manager.scene
        state = self.manager._execution
        assert scene.animations is not None
        scene.duration = scene.get_run_time(scene.animations)
        # The same sample times as Scene.get_time_progression.
        step = 1 / self.manager.session_spec.frame_rate
        for offset in np.arange(0, scene.duration, step):
            scene.update_to_time(offset)
            self._frame(play, state.frame, 1, float(offset))
            # Stop conditions, finish() and cleanup see the end of the frame.
            state.frame += 1
            if scene.stop_condition is not None and scene.stop_condition():
                break
        for animation in scene.animations:
            animation.finish()
            animation.clean_up_from_scene(scene)
        scene.update_mobjects(0)

    def _frame(self, play: _Play, index: int, repeat: int, offset: float) -> None:
        frame = _Frame(self, play, index, repeat, offset)
        for listener in self.listeners:
            listener.frame(frame)

    @property
    def in_output(self) -> bool:
        """Whether the next play, which starts at the current time, is in the output."""
        return self.selection.includes(self.manager.num_plays)

    def marker(self, marker: _Section | _Caption | _Sound) -> None:
        if isinstance(marker, _Section):
            self.selection.start_section(marker.skip_animations)
        for listener in self.listeners:
            listener.marker(marker)
