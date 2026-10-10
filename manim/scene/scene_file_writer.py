"""Scene output coordination and media-artifact assembly."""

from __future__ import annotations

__all__ = ["SceneFileWriter"]

import json
import math
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
from functools import cached_property
from io import BytesIO
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from threading import Thread
from typing import TYPE_CHECKING, Any

import av
import numpy as np
import srt
from PIL import Image

from manim import __version__

from .. import logger
from .._config.output_plan import OutputPlan
from .._config.video_encoder import VideoEncoderSpec
from ..utils.caching import prune_segment_cache
from ..utils.file_ops import modify_atime
from ..utils.sounds import get_full_sound_file_path
from .audio_mixer import LAYOUT, SAMPLE_RATE, _probe_duration, _Sound, _SoundMix
from .section import DefaultSectionType, Section
from .video_segment_encoder import VideoSegmentEncoder

if TYPE_CHECKING:
    from manim.typing import RGBAPixelArray, StrPath


def _audio_codec(container_extension: str) -> str:
    """Return the audio codec used for sound in a video container."""
    if container_extension == ".webm":
        return _webm_audio_codec()
    if container_extension == ".mov":
        return "pcm_s16le"
    return "aac"


def _webm_audio_codec() -> str:
    """Return the audio codec used for sound in a VP9 (webm) video.

    ``libvorbis`` is preferred, but PyAV wheels are not always built with it
    (for example some Windows builds only ship the experimental native
    ``vorbis`` encoder). Fall back to ``libopus``, which webm also supports.
    """
    try:
        av.codec.Codec("libvorbis", "w")
    except ValueError:
        return "libopus"
    return "libvorbis"


class _AudioTrack:
    """Encode mixed sample blocks into an audio stream of an output container."""

    def __init__(
        self,
        container: av.container.OutputContainer,
        codec: str,
        blocks: Iterator[np.ndarray],
    ) -> None:
        self._container = container
        self._stream = container.add_stream(codec, rate=SAMPLE_RATE, layout=LAYOUT)
        self._blocks = blocks
        self._written = 0
        self._finished = False

    def write_until(self, seconds: float | None) -> None:
        """Encode audio up to ``seconds``, or all remaining audio for ``None``."""
        while not self._finished and (
            seconds is None or self._written < seconds * SAMPLE_RATE
        ):
            block = next(self._blocks, None)
            if block is None:
                self._finished = True
                frame = None
            else:
                # PyAV converts the sample format and frame size for the codec.
                frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=LAYOUT)
                frame.sample_rate = SAMPLE_RATE
                frame.pts = self._written
                frame.time_base = Fraction(1, SAMPLE_RATE)
                self._written += block.shape[1]
            for packet in self._stream.encode(frame):
                self._container.mux(packet)


class _PartialMovieEncodeJob:
    """Run one segment encoder on a dedicated worker thread."""

    def __init__(
        self,
        *,
        animation_index: int,
        encoder: VideoSegmentEncoder,
        frame_queue_size: int,
    ) -> None:
        self.path = encoder.target
        self.animation_index = animation_index
        self.encoder = encoder
        # Bound the queue so rendering cannot run arbitrarily far ahead of the
        # encoder; at the default capacity, eight 1080p RGBA frames occupy about
        # 66 MB per job. The worker drains through the sentinel after an
        # exception, so a bounded queue cannot deadlock.
        self.queue: Queue[tuple[int, RGBAPixelArray | None]] = Queue(
            maxsize=frame_queue_size,
        )
        self._exception: BaseException | None = None
        self._sealed = False
        self._abort_requested = False
        self.thread = Thread(
            target=self._listen_and_write,
            name=f"partial-movie-encoder-{animation_index}",
        )
        self.thread.start()

    def _capture_exception(self, exception: BaseException) -> None:
        if self._exception is None:
            self._exception = exception

    @property
    def failed(self) -> bool:
        """Whether the worker has captured an exception."""
        return self._exception is not None

    def _abort_encoder(self) -> None:
        try:
            self.encoder.abort()
        except Exception as exception:
            logger.warning(
                "Failed to clean up incomplete segment %(path)s: %(error)s",
                {"path": f"'{self.path}'", "error": exception},
            )
            self._capture_exception(exception)

    def _listen_and_write(self) -> None:
        while True:
            repeat, frame_data = self.queue.get()
            if frame_data is None:
                break
            if self._exception is not None:
                continue

            try:
                self.encoder.write_frame(frame_data, repeat=repeat)
            except BaseException as exception:
                self._capture_exception(exception)

        if self._abort_requested or self._exception is not None:
            self._abort_encoder()
            return

        try:
            self.encoder.finish()
        except BaseException as exception:
            self._capture_exception(exception)
            self._abort_encoder()

    def put(self, repeat: int, frame: RGBAPixelArray) -> None:
        """Add a frame to the encoding queue."""
        self.queue.put((repeat, frame))

    def seal(self) -> None:
        """Signal that no more frames will be added."""
        if not self._sealed:
            self._sealed = True
            self.queue.put((-1, None))

    def abort(self) -> None:
        """Signal that the segment must be discarded."""
        self._abort_requested = True
        self.seal()

    def join(self) -> None:
        """Wait for encoding to finish and propagate worker failures."""
        self.thread.join()
        if self._exception is not None:
            raise self._exception
        if not self._abort_requested:
            logger.info(
                f"Animation {self.animation_index} : Partial movie file written in %(path)s",
                {"path": f"'{self.path}'"},
            )


@dataclass(frozen=True, slots=True)
class _SceneFileWriterSettings:
    """Immutable inputs consumed by one :class:`SceneFileWriter`.

    The settings contain resolved output paths and segment encoding, bounded
    encoder-pool limits, cache maintenance, and the sound-asset search root.
    """

    plan: OutputPlan
    video_encoder: VideoEncoderSpec | None
    max_inflight_encoders: int
    encoder_queue_size: int
    max_files_cached: int
    assets_dir: Path

    def __post_init__(self) -> None:
        output = self.plan.output
        if output.is_video != (self.video_encoder is not None):
            raise ValueError(
                "Video output and resolved video encoder settings must be provided together.",
            )
        expected_segment_extension = (
            output.segment_extension if output.is_video else None
        )
        if self.plan.segment_extension != expected_segment_extension:
            raise ValueError(
                "The output plan segment extension does not match its output specification.",
            )
        if (
            self.video_encoder is not None
            and f".{self.video_encoder.container_format}" != expected_segment_extension
        ):
            raise ValueError(
                "The video encoder container does not match the output plan.",
            )
        if self.max_inflight_encoders <= 0:
            raise ValueError("max_inflight_encoders must be positive.")
        if self.encoder_queue_size <= 0:
            raise ValueError("encoder_queue_size must be positive.")
        if self.max_files_cached < -1:
            raise ValueError("max_files_cached must be non-negative or -1.")
        if not self.assets_dir.is_absolute():
            raise ValueError("assets_dir must be absolute.")


class SceneFileWriter:
    """Coordinate segment jobs and assemble one scene's media artifacts.

    The writer receives immutable resolved settings and concrete top-left-origin
    RGBA arrays. Ownership of each array passed to
    :meth:`write_frame` transfers to the writer; callers must not mutate or reuse
    it afterward. For video output the writer coordinates queued
    :class:`.VideoSegmentEncoder` jobs, then assembles their silent cached
    segments with optional audio, sections, and subcaptions. It also writes
    still images and PNG sequences described by the output plan.

    Parameters
    ----------
    settings
        Resolved output, encoding, pool, cache, and asset-search settings.

    Attributes
    ----------
    sections
        Ordered section metadata for the scene.
    partial_movie_files
        Segment paths in animation order, including ``None`` for skipped plays.
    """

    def __init__(self, settings: _SceneFileWriterSettings) -> None:
        self.settings = settings
        self.output_spec = settings.plan.output
        self.output_plan = settings.plan
        self.video_encoder = settings.video_encoder
        self._inflight_encode_jobs: list[_PartialMovieEncodeJob] = []
        self._inflight_by_path: dict[str, _PartialMovieEncodeJob] = {}
        self._current_encode_job: _PartialMovieEncodeJob | None = None
        self._sounds: list[_Sound] = []
        # Scene-time intervals visible in the movie, in order; adjacent ones merged.
        self._intervals: list[tuple[float, float]] = []
        self.frame_count = 0
        self.partial_movie_files: list[str | None] = []
        self.subcaptions: list[srt.Subtitle] = []
        self.sections: list[Section] = []
        # first section gets automatically created for convenience
        # if you need the first section to be skipped, add a first section by hand, it will replace this one
        self.next_section(
            name="autocreated", type_=DefaultSectionType.NORMAL, skip_animations=False
        )

    @property
    def output_name(self) -> Path:
        """Return the planned logical output stem as a compatibility view."""
        return Path(self.output_plan.output_stem)

    @property
    def image_file_path(self) -> Path:
        """Return the planned still or video-fallback image path."""
        if self.output_spec.is_image_sequence:
            return self.image_sequence_directory.with_suffix(".png")
        path = (
            self.output_plan.primary_artifact
            if self.output_spec.is_still
            else self.output_plan.fallback_image
        )
        if path is None:
            raise AttributeError("This output plan does not contain an image path.")
        return path

    @property
    def image_sequence_directory(self) -> Path:
        """Return the planned PNG-sequence directory."""
        path = self.output_plan.image_sequence_dir
        if path is None:
            raise AttributeError("This output plan does not contain an image sequence.")
        return path

    @property
    def movie_file_path(self) -> Path:
        """Return the planned primary video artifact path."""
        if not self.output_spec.is_video or self.output_plan.primary_artifact is None:
            raise AttributeError("This output plan does not contain a video artifact.")
        return self.output_plan.primary_artifact

    @property
    def gif_file_path(self) -> Path:
        """Return the planned GIF artifact path."""
        if not self.output_spec.is_gif:
            raise AttributeError("This output plan does not contain a GIF artifact.")
        return self.movie_file_path

    @property
    def sections_output_dir(self) -> Path:
        """Return the planned sections directory, or the legacy empty path."""
        return self.output_plan.sections_dir or Path("")

    @property
    def partial_movie_directory(self) -> Path:
        """Return the planned silent-segment cache directory."""
        path = self.output_plan.segment_cache_dir
        if path is None:
            raise AttributeError("This output plan does not contain video segments.")
        return path

    def finish_last_section(self) -> None:
        """Delete current section if it is empty."""
        if len(self.sections) and self.sections[-1].is_empty():
            self.sections.pop()

    def next_section(self, name: str, type_: str, skip_animations: bool) -> None:
        """Create segmentation cut here."""
        self.finish_last_section()

        # images don't support sections
        section_video: str | None = None
        # don't save when None
        if self.output_spec.save_sections and not skip_animations:
            section_path = self.output_plan.section_path(len(self.sections), name)
            assert self.output_plan.sections_dir is not None
            # Section stores paths relative to its index file.
            section_video = section_path.relative_to(
                self.output_plan.sections_dir,
            ).as_posix()

        self.sections.append(
            Section(
                type_,
                section_video,
                name,
                skip_animations,
                output_start=self._output_time(math.inf),
            ),
        )

    def add_partial_movie_file(self, hash_animation: str | None) -> None:
        """Append a planned segment path to the writer and current section.

        The list retains one entry per animation so explicit animation indices
        select the corresponding segment.

        Parameters
        ----------
        hash_animation
            Hash of the animation.
        """
        if not self.output_spec.is_video:
            return

        # Skipped animations retain a placeholder to preserve index alignment.
        if hash_animation is None:
            self.partial_movie_files.append(None)
            self.sections[-1].partial_movie_files.append(None)
        else:
            new_partial_movie_file = str(self.output_plan.segment_path(hash_animation))
            self.partial_movie_files.append(new_partial_movie_file)
            self.sections[-1].partial_movie_files.append(new_partial_movie_file)

    # Sound
    def add_sound(
        self,
        sound_file: StrPath,
        time: float,
        gain: float | None = None,
        gain_to_background: float | None = None,
    ) -> None:
        """Add a sound at a scene time.

        The file is located and checked immediately, so a missing or unreadable
        file fails at the call site. It is decoded and mixed when the movie is
        assembled; see :mod:`~manim.scene.audio_mixer`.

        Parameters
        ----------
        sound_file
            The path to the sound file, absolute or relative to the assets
            directory.
        time
            The scene time in seconds at which the sound starts.
        gain
            Gain applied to the sound, in dB.
        gain_to_background
            Gain applied to all previously added sounds while this one plays,
            in dB.
        """
        file_path = get_full_sound_file_path(sound_file, self.settings.assets_dir)
        _probe_duration(file_path)
        self._sounds.append(_Sound(file_path, time, gain, gain_to_background))

    @cached_property
    def _sound_mix(self) -> _SoundMix | None:
        """The scene's mix through the visible intervals, created on first use."""
        if not self._sounds:
            return None
        return _SoundMix(self._sounds, self._intervals)

    # Writers
    def begin_animation(
        self,
        allow_write: bool = False,
        *,
        animation_index: int,
        file_path: StrPath | None = None,
    ) -> None:
        """Start a segment job for one animation when video writing is enabled.

        Parameters
        ----------
        allow_write
            Whether this animation needs a new segment.
        animation_index
            Scene-local animation index used to select and label the segment.
        file_path
            Explicit segment target, or ``None`` to use the planned cache path.
        """
        if self.output_spec.is_video and allow_write:
            self.open_partial_movie_stream(
                animation_index=animation_index,
                file_path=file_path,
            )

    def end_animation(
        self,
        allow_write: bool = False,
        *,
        scene_interval: tuple[float, float],
    ) -> None:
        """Seal the current segment job and record the scene time it shows.

        Parameters
        ----------
        allow_write
            Whether the current animation has an open segment job.
        scene_interval
            The scene times at which the animation started and ended. The
            interval is part of the movie unless the animation was excluded from
            the output; sounds and subcaptions are placed through these intervals.
        """
        if not self.output_spec.is_video:
            return
        if allow_write:
            self.close_partial_movie_stream()
        if self.partial_movie_files and self.partial_movie_files[-1] is not None:
            start, end = scene_interval
            if self._intervals and self._intervals[-1][1] == start:
                start = self._intervals.pop()[0]
            self._intervals.append((start, end))

    def write_frame(
        self,
        pixels: RGBAPixelArray,
        *,
        repeat: int = 1,
    ) -> None:
        """Take ownership of one top-left C-contiguous ``uint8`` RGBA frame.

        The caller must not mutate or reuse ``pixels`` after this method returns
        because video encoding can consume the array asynchronously.
        """
        if self.output_spec.is_video:
            job = self._current_encode_job
            if job is None:
                # Presentation rendering can emit frames outside an open
                # segment; such frames do not belong to file output.
                return
            if job.failed:
                # Surface the failure at the first write after it was captured;
                # the worker discards the partial before join() re-raises.
                job.seal()
                self._current_encode_job = None
                job.join()
            job.put(repeat, pixels)

        if self.output_spec.is_image_sequence:
            self.output_image(Image.fromarray(pixels))

    def output_image(self, image: Image.Image) -> None:
        file_path = self.output_plan.image_frame_path(self.frame_count)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(file_path)
        self.frame_count += 1

    def save_image(self, pixels: RGBAPixelArray) -> None:
        """Save one RGBA frame to the planned still-image path."""
        if not self.output_spec.enabled:
            return
        self.image_file_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(self.image_file_path)
        self.print_file_ready_message(self.image_file_path)

    def finish(self) -> None:
        """Drain segment jobs and assemble the configured time-based output."""
        if self.output_spec.is_video:
            self.join_all_encode_jobs()
            self.combine_to_movie()
            if self.output_spec.save_sections:
                self.combine_to_section_videos()
            # Cache cleanup runs after the in-flight encode jobs have been drained.
            prune_segment_cache(
                self.partial_movie_directory,
                self.settings.max_files_cached,
            )
        elif self.output_spec.is_image_sequence:
            target_dir = self.image_sequence_directory
            self.final_file_path = target_dir
            logger.info("\n%i images ready at %s\n", self.frame_count, str(target_dir))
        if self.subcaptions:
            self.write_subcaption_file()

    def _create_segment_encoder(self, target: Path) -> VideoSegmentEncoder:
        encoder = self.video_encoder
        if encoder is None:
            raise RuntimeError("Video segment encoding requires resolved settings.")
        return VideoSegmentEncoder(target=target, spec=encoder)

    def open_partial_movie_stream(
        self,
        *,
        animation_index: int,
        file_path: StrPath | None = None,
    ) -> None:
        """Create a queued encoder job for one planned video segment."""
        if self._current_encode_job is not None:
            raise RuntimeError(
                "Cannot open a video segment while another segment is still open.",
            )
        if file_path is None:
            file_path = self.partial_movie_files[animation_index]
            if file_path is None:
                raise RuntimeError(
                    "open_partial_movie_stream() called for a play that has no "
                    "partial movie file path.",
                )
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        path_key = str(file_path)
        if path_key in self._inflight_by_path:
            self._join_job_and_drain_on_failure(self._inflight_by_path[path_key])
        segment_encoder = self._create_segment_encoder(file_path)
        self._current_encode_job = _PartialMovieEncodeJob(
            animation_index=animation_index,
            encoder=segment_encoder,
            frame_queue_size=self.settings.encoder_queue_size,
        )

    def _join_job(self, job: _PartialMovieEncodeJob) -> None:
        """Remove and join an in-flight partial movie encode job."""
        if job in self._inflight_encode_jobs:
            self._inflight_encode_jobs.remove(job)
        self._inflight_by_path.pop(str(job.path), None)
        job.join()

    def _join_job_and_drain_on_failure(
        self,
        job: _PartialMovieEncodeJob,
    ) -> None:
        """Join one job, draining all remaining jobs if it fails."""
        try:
            self._join_job(job)
        except BaseException:
            # Preserve the failure which triggered the drain.
            with suppress(BaseException):
                self.join_all_encode_jobs()
            raise

    def join_all_encode_jobs(self) -> None:
        """Join every in-flight encode job, re-raising the first failure."""
        first_exception: BaseException | None = None
        for job in list(self._inflight_encode_jobs):
            try:
                self._join_job(job)
            except BaseException as exception:
                if first_exception is None:
                    first_exception = exception

        self._inflight_encode_jobs.clear()
        self._inflight_by_path.clear()
        if first_exception is not None:
            raise first_exception

    def abort_encode_jobs(self, reraise_encoder_failures: bool = False) -> None:
        """Discard the current segment and drain completed encode jobs.

        When ``reraise_encoder_failures`` is true, the first encoder failure is
        propagated. Otherwise failures are logged so an active render exception
        remains primary.
        """
        current_exception: BaseException | None = None
        job = self._current_encode_job
        if job is not None:
            # Request abort before clearing: an interrupt between these
            # statements must not orphan a worker blocked on its queue.
            job.abort()
            self._current_encode_job = None
            job.thread.join()
            current_exception = job._exception
            if current_exception is not None:
                logger.error(
                    "Encoder for aborted animation %d had also failed",
                    job.animation_index,
                    exc_info=current_exception,
                )
            else:
                logger.info(
                    "Discarded partial movie file of aborted animation %(index)d",
                    {"index": job.animation_index},
                )
        if reraise_encoder_failures:
            self.join_all_encode_jobs()
            if current_exception is not None:
                # The rerun path has no primary exception: a failed current
                # job must not be silently absorbed.
                raise current_exception
        else:
            try:
                self.join_all_encode_jobs()
            except BaseException:
                logger.exception("Encoder failure while aborting render")

    def close_partial_movie_stream(self) -> None:
        """Seal the current segment and enforce the in-flight job limit."""
        job = self._current_encode_job
        if job is None:
            raise RuntimeError(
                "close_partial_movie_stream() called without an open partial "
                "movie stream.",
            )
        job.seal()
        self._inflight_encode_jobs.append(job)
        self._inflight_by_path[str(job.path)] = job
        self._current_encode_job = None

        while len(self._inflight_encode_jobs) >= self.settings.max_inflight_encoders:
            self._join_job_and_drain_on_failure(self._inflight_encode_jobs[0])

    def is_already_cached(self, hash_invocation: str) -> bool:
        """Will check if a file named with `hash_invocation` exists.

        Parameters
        ----------
        hash_invocation
            The hash corresponding to an invocation to either `scene.play` or `scene.wait`.

        Returns
        -------
        :class:`bool`
            Whether the file exists.
        """
        if not self.output_spec.is_video:
            return False
        path = self.output_plan.segment_path(hash_invocation)
        path_key = str(path)
        if path_key in self._inflight_by_path:
            self._join_job_and_drain_on_failure(self._inflight_by_path[path_key])
        return path.exists()

    @staticmethod
    def _concat_manifest_bytes(input_files: list[str]) -> bytes:
        """Return a complete FFmpeg concat manifest for ``input_files``."""
        manifest_text = (
            "# This file records the segment order used by Manim.\n"
            + "".join(
                f"file 'file:{Path(file_path).as_posix()}'\n"
                for file_path in input_files
            )
        )
        return manifest_text.encode("utf-8")

    def _write_concat_manifest(self, input_files: list[str]) -> None:
        """Atomically persist the complete scene segment order for diagnostics."""
        manifest_path = self.output_plan.concat_manifest
        assert manifest_path is not None
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(self._concat_manifest_bytes(input_files))
            temporary_path.replace(manifest_path)
        except BaseException:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
            raise

    def combine_files(
        self,
        input_files: list[str],
        output_file: Path,
        create_gif: bool = False,
        audio: Iterator[np.ndarray] | None = None,
    ) -> None:
        """Concatenate segments into ``output_file``, optionally adding audio.

        Parameters
        ----------
        input_files
            Segment paths in playback order.
        output_file
            The artifact to write.
        create_gif
            Whether to encode a GIF instead of copying the video packets.
        audio
            Mixed float32 ``(2, n)`` sample blocks covering the artifact, as
            produced by the scene's sound mix. Ignored for GIFs.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Partial movie files to combine ({len(input_files)} files): %(p)s",
            {"p": input_files[:5]},
        )
        manifest = BytesIO(self._concat_manifest_bytes(input_files))

        av_options = {
            "safe": "0",  # needed to read files
        }

        partial_movies_input = av.open(
            manifest,
            options=av_options,
            format="concat",
        )
        partial_movies_stream = partial_movies_input.streams.video[0]
        output_container = av.open(str(output_file), mode="w")
        output_container.metadata["comment"] = (
            f"Rendered with Manim Community v{__version__}"
        )
        if create_gif:
            """The following solution was largely inspired from this comment
            https://github.com/imageio/imageio/issues/995#issuecomment-1580533018,
            and the following code
            https://github.com/imageio/imageio/blob/65d79140018bb7c64c0692ea72cb4093e8d632a0/imageio/plugins/pyav.py#L927-L996.
            """
            output_stream = output_container.add_stream(
                codec_name="gif",
            )
            output_stream.pix_fmt = "rgb8"
            if self.output_spec.transparent:
                output_stream.pix_fmt = "pal8"
            encoder = self.video_encoder
            assert encoder is not None
            output_stream.width = encoder.width
            output_stream.height = encoder.height
            output_stream.rate = encoder.frame_rate
            graph = av.filter.Graph()
            input_buffer = graph.add_buffer(template=partial_movies_stream)
            split = graph.add("split")
            palettegen = graph.add("palettegen", "stats_mode=diff")
            paletteuse = graph.add(
                "paletteuse", "dither=bayer:bayer_scale=5:diff_mode=rectangle"
            )
            output_sink = graph.add("buffersink")

            input_buffer.link_to(split)
            split.link_to(palettegen, 0, 0)  # 1st input of split -> input of palettegen
            split.link_to(paletteuse, 1, 0)  # 2nd output of split -> 1st input
            palettegen.link_to(paletteuse, 0, 1)  # output of palettegen -> 2nd input
            paletteuse.link_to(output_sink)

            graph.configure()

            for frame in partial_movies_input.decode(video=0):
                graph.push(frame)

            graph.push(None)  # EOF: https://github.com/PyAV-Org/PyAV/issues/886.

            frames_written = 0
            while True:
                try:
                    frame = graph.pull()
                    if output_stream.codec_context.time_base is not None:
                        frame.time_base = output_stream.codec_context.time_base
                    frame.pts = frames_written
                    frames_written += 1
                    output_container.mux(output_stream.encode(frame))
                except av.error.EOFError:
                    break

            for packet in output_stream.encode():
                output_container.mux(packet)

        else:
            output_stream = output_container.add_stream_from_template(
                template=partial_movies_stream,
            )
            if (
                self.output_spec.transparent
                and self.output_spec.segment_extension == ".webm"
            ):
                output_stream.pix_fmt = "yuva420p"
            audio_track = (
                None
                if audio is None
                else _AudioTrack(
                    output_container,
                    _audio_codec(self.output_spec.segment_extension),
                    audio,
                )
            )
            for packet in partial_movies_input.demux(partial_movies_stream):
                # We need to skip the "flushing" packets that `demux` generates.
                if packet.dts is None:
                    continue

                packet.dts = None  # This seems to be needed, as dts from consecutive
                # files may not be monotically increasing, so we let libav compute it.

                # Keep audio interleaved with the video it accompanies.
                if audio_track is not None and packet.pts is not None:
                    audio_track.write_until(float(packet.pts * packet.time_base))

                # We need to assign the packet to the new stream.
                packet.stream = output_stream
                output_container.mux(packet)
            if audio_track is not None:
                audio_track.write_until(None)

        partial_movies_input.close()
        output_container.close()
        manifest.close()

    def combine_to_movie(self) -> None:
        """Used internally by Manim to combine the separate
        partial movie files that make up a Scene into a single
        video file for that Scene.
        """
        partial_movie_files = [el for el in self.partial_movie_files if el is not None]
        # NOTE: Here we should do a check and raise an exception if partial
        # movie file is empty.  We can't, as a lot of stuff (in particular, in
        # tests) use scene initialization, and this error would be raised as
        # it's just an empty scene initialized.

        # determine output path
        movie_file_path = self.movie_file_path
        if self.output_spec.is_gif:
            movie_file_path = self.gif_file_path

        if len(partial_movie_files) == 0:  # Prevent calling concat on empty list
            logger.info("No animations are contained in this scene.")
            return

        logger.info("Combining to Movie file.")
        self._write_concat_manifest(partial_movie_files)
        mix = None if self.output_spec.is_gif else self._sound_mix
        if mix is not None and (cut := mix.cut_at_end()) > 0:
            logger.warning(
                "Sound runs up to %(cut).2f s past the end of the movie; that part "
                "is cut. Add a wait() at the end of the scene to hear all of it.",
                {"cut": cut},
            )
        self.combine_files(
            partial_movie_files,
            movie_file_path,
            self.output_spec.is_gif,
            audio=None if mix is None else mix.blocks(0, mix.duration),
        )

        self.print_file_ready_message(str(movie_file_path))
        if self.output_spec.is_video:
            for file_path in partial_movie_files:
                # We have to modify the accessed time so if we have to clean the cache we remove the one used the longest.
                modify_atime(file_path)

    def combine_to_section_videos(self) -> None:
        """Concatenate partial movie files for each section."""
        self.finish_last_section()
        mix = self._sound_mix
        sections_index: list[dict[str, Any]] = []
        for index, section in enumerate(self.sections):
            # only if section does want to be saved
            if section.video is not None:
                logger.info(f"Combining partial files for section '{section.name}'")
                section_path = self.sections_output_dir / section.video
                # A section shows its stretch of the scene's single mix.
                end = (
                    self.sections[index + 1].output_start
                    if index + 1 < len(self.sections)
                    else self._output_time(math.inf)
                )
                self.combine_files(
                    section.get_clean_partial_movie_files(),
                    section_path,
                    audio=None
                    if mix is None
                    else mix.blocks(section.output_start, end),
                )
                sections_index.append(section.get_dict(self.sections_output_dir))
        section_index = self.output_plan.section_index
        assert section_index is not None
        section_index.parent.mkdir(parents=True, exist_ok=True)
        with section_index.open("w") as file:
            json.dump(sections_index, file, indent=4)

    def write_subcaption_file(self) -> None:
        """Writes the subcaption file next to the primary video artifact."""
        if not self.output_spec.is_video:
            return
        subcaption_file = self.output_plan.subcaption_file
        assert subcaption_file is not None
        # Subcaptions are recorded at scene time; place them in the movie.
        output_end = self._intervals[-1][1] if self._intervals else 0.0
        subcaptions: list[srt.Subtitle] = []
        for subcaption in self.subcaptions:
            start = subcaption.start.total_seconds()
            end = subcaption.end.total_seconds()
            if start < output_end < end:
                logger.warning(
                    "Subcaption %(content)r runs %(cut).2f s past the end of the "
                    "movie; that part is cut.",
                    {"content": subcaption.content, "cut": end - output_end},
                )
            start, end = self._output_time(start), self._output_time(end)
            if start < end:
                subcaptions.append(
                    srt.Subtitle(
                        index=len(subcaptions),
                        content=subcaption.content,
                        start=timedelta(seconds=start),
                        end=timedelta(seconds=end),
                    )
                )
        subcaption_file.parent.mkdir(parents=True, exist_ok=True)
        subcaption_file.write_text(srt.compose(subcaptions), encoding="utf-8")
        logger.info(f"Subcaption file has been written as {subcaption_file}")

    def _output_time(self, scene_time: float) -> float:
        """Return the movie time at which ``scene_time`` is shown.

        Excluded parts of the scene take no movie time, so the result never
        decreases and a scene interval maps to one contiguous movie interval.
        """
        return sum(
            max(0.0, min(scene_time, end) - start) for start, end in self._intervals
        )

    def print_file_ready_message(self, file_path: StrPath) -> None:
        """Record and report a completed primary artifact."""
        self.final_file_path = Path(file_path)
        logger.info("\nFile ready at %(file_path)s\n", {"file_path": f"'{file_path}'"})
