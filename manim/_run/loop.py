"""Run play and wait calls frame by frame and report each step to listeners."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .frames import frame_count

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL.Image import Image

    from ..animation.animation import Animation
    from ..manager import Manager
    from ..mobject.mobject import Mobject, _AnimationBuilder
    from ..mobject.opengl.opengl_mobject import OpenGLMobject
    from ..scene.scene import Scene
    from ..typing import RGBAPixelArray
    from .cache import _SegmentCache
    from .mode import _RunMode
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
    cache_key: str | None = None
    """The key of the play's partial movie file, if the play is in the output."""
    cached: bool = False
    """Whether the play's partial movie file is cached, so its frames are reused."""

    @property
    def drawn(self) -> bool:
        """Whether the play's frames are drawn: it is in the output and not
        cached. Decided before the listeners hear of the play.
        """
        return self.in_output and not self.cached


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
        """Return the frame as an RGBA pixel array, drawing it if needed.

        A drawer that draws into a target draws the frame first; one without a
        target, such as evaluation's, draws the pixels directly.
        """
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

    Listeners must not change the scene's mobjects, call back into the loop,
    or decide anything for other listeners: what a play is, including whether it
    is drawn, is decided before they hear of it (see :class:`_Play`). Every
    method does nothing unless overridden.
    """

    def play_started(self, play: _Play) -> None:
        """A play's animations are prepared but have not begun yet."""

    def frame(self, frame: _Frame) -> None:
        """The scene shows the state of ``frame``; time is at the frame's start."""

    def play_finished(self, play: _Play) -> None:
        """The play's animations have finished and were cleaned up."""

    def play_failed(self, play: _Play) -> None:
        """The play raised an exception after it started; it does not finish."""

    def marker(self, marker: _Section | _Caption | _Sound) -> None:
        """A section, caption, or sound call was made."""


class _Drawer:
    """Draws the frames of a run for its listeners.

    Like :class:`_RunListener`, a base class: :meth:`draw` and :meth:`end_play`
    do nothing unless overridden, for drawers without a target of their own.
    """

    def draw(self, frame: _Frame) -> None:
        """Draw ``frame`` into the renderer's target, for a preview to show."""

    def pixels(self, frame: _Frame) -> RGBAPixelArray:
        """The pixels of ``frame``, which :meth:`draw` drew if it was called."""
        raise NotImplementedError

    def end_play(self, play: _Play) -> None:
        """The play's frames are drawn."""


class _TargetDrawer(_Drawer):
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


class _SceneImageDrawer(_Drawer):
    """Draw frames with a temporary renderer, as :meth:`.Scene.get_image` does.

    Used during evaluation, which opens no drawing resources of its own, so
    there is no target to draw into: only pixels are drawn.
    """

    def __init__(self, manager: Manager) -> None:
        self._manager = manager

    def pixels(self, frame: _Frame) -> RGBAPixelArray:
        manager = self._manager
        return np.asarray(manager.renderer._get_scene_image(manager.scene))


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
        mode: _RunMode,
        cache: _SegmentCache | None = None,
    ) -> None:
        self.manager = manager
        self.listeners = tuple(listeners)
        self.selection = selection
        self.drawer = drawer
        self.cache = cache
        """Where the partial movie files of earlier renders are, if the run
        reuses them."""
        self.mode = mode
        """What the run produces: a render or an evaluation."""
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
        if self.cache is not None:
            play = self.cache.look_up(play)
        # For code outside the loop that asks whether the current play draws,
        # such as the renderers' add_frame(); listeners read play.drawn.
        manager.skip_animations = not play.drawn
        started: list[_RunListener] = []
        try:
            for listener in self.listeners:
                started.append(listener)
                listener.play_started(play)
            scene.begin_animations()
            try:
                if play.frozen:
                    repeat = frame_count(
                        scene.duration, manager.session_spec.frame_rate
                    )
                    self._frame(play, state.frame, repeat, 0.0)
                    state.frame += repeat
                else:
                    self._step(play)
            finally:
                self.drawer.end_play(play)
        except BaseException:
            # Let every listener that saw the play start drop its state for it,
            # so that a caller that handles the error can play on.
            for listener in started:
                listener.play_failed(play)
            raise
        for listener in self.listeners:
            listener.play_finished(play)
        manager.num_plays += 1

    def _step(self, play: _Play) -> None:
        scene = self.manager.scene
        state = self.manager._execution
        assert scene.animations is not None
        scene.duration = scene.get_run_time(scene.animations)
        rate = self.manager.session_spec.frame_rate
        count = frame_count(scene.duration, rate)
        if count and self._unobserved(play, scene):
            # Only the end state matters: go to the last frame in one step, at
            # the same scene time as when stepping through every frame.
            state.frame += count - 1
            offset = (count - 1) / rate
            scene.update_to_time(offset)
            self._frame(play, state.frame, 1, offset)
            state.frame += 1
            count = 0
        for index in range(count):
            offset = index / rate
            scene.update_to_time(offset)
            self._frame(play, state.frame, 1, offset)
            # Stop conditions, finish() and cleanup see the end of the frame.
            state.frame += 1
            if scene.stop_condition is not None and scene.stop_condition():
                break
        for animation in scene.animations:
            animation.finish()
            animation.clean_up_from_scene(scene)
        scene.update_mobjects(0)

    def _unobserved(self, play: _Play, scene: Scene) -> bool:
        """Whether nothing can see the frames between a play's start and end.

        True when the frames are not drawn (the play is left out of the output
        or reused from the cache) and no code runs per frame: no updaters, no
        stop condition. The play then only needs its last frame.
        """
        if play.drawn or scene.stop_condition is not None:
            return False
        if scene.updaters or scene.always_update_mobjects or scene.meshes:
            return False
        assert scene.animations is not None
        roots: list[Mobject | OpenGLMobject] = [*scene.mobjects]
        roots += [a.mobject for a in scene.animations if a.mobject is not None]
        return not any(
            mobject.get_updaters() for root in roots for mobject in root.get_family()
        )

    def _frame(self, play: _Play, index: int, repeat: int, offset: float) -> None:
        frame = _Frame(self, play, index, repeat, offset)
        for listener in self.listeners:
            listener.frame(frame)

    def add_frames(self, pixels: RGBAPixelArray | None, repeat: int) -> None:
        """Show ``repeat`` frames outside of a play, as CairoRenderer.add_frame() does.

        ``pixels`` is the image to show, or None for the scene's current state,
        drawn only if a listener asks for it.
        """
        manager = self.manager
        state = manager._execution
        play = _Play(
            index=manager.num_plays,
            first_frame=state.frame,
            run_time=repeat / manager.session_spec.frame_rate,
            animations=(),
            frozen=True,
            in_output=self.in_output,
        )
        frame = _Frame(self, play, state.frame, repeat, 0.0)
        if pixels is not None:
            frame._pixels = pixels
            frame._drawn = True
        for listener in self.listeners:
            listener.frame(frame)
        state.frame += repeat

    @property
    def in_output(self) -> bool:
        """Whether the next play, which starts at the current time, is in the output."""
        return self.selection.includes(self.manager.num_plays)

    def marker(self, marker: _Section | _Caption | _Sound) -> None:
        if isinstance(marker, _Section):
            self.selection.start_section(marker.skip_animations)
        elif isinstance(marker, _Sound):
            # Heard if the play that starts at the sound's call is in the output.
            marker = replace(marker, in_output=self.in_output)
        for listener in self.listeners:
            listener.marker(marker)
