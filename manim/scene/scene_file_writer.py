"""Scene output coordination and media-artifact assembly."""

from __future__ import annotations

__all__ = ["SceneFileWriter"]

import json
import shutil
import warnings
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from threading import Thread
from typing import TYPE_CHECKING, Any

import av
import numpy as np
import srt
from PIL import Image

# Manim handles audio conversion through PyAV directly. Importing pydub emits a
# RuntimeWarning if ffmpeg/avconv is not on PATH, even when only WAV code paths
# are used (which do not need ffmpeg). Silence this specific warning.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*ffmpeg or avconv.*",
        category=RuntimeWarning,
    )
    from pydub import AudioSegment

from manim import __version__

from .. import logger
from .._config.output_plan import OutputPlan
from .._config.video_encoder import VideoEncoderSpec
from ..utils.caching import prune_segment_cache
from ..utils.file_ops import modify_atime
from ..utils.sounds import get_full_sound_file_path
from .section import DefaultSectionType, Section
from .video_segment_encoder import VideoSegmentEncoder

if TYPE_CHECKING:
    from manim.typing import RGBAPixelArray, StrPath


def convert_audio(
    input_path: Path, output_path: Path | _TemporaryFileWrapper[bytes], codec_name: str
) -> None:
    with (
        av.open(input_path) as input_audio,
        av.open(output_path, "w") as output_audio,
    ):
        input_audio_stream = input_audio.streams.audio[0]
        output_audio_stream = output_audio.add_stream(codec_name)
        for frame in input_audio.decode(input_audio_stream):
            for packet in output_audio_stream.encode(frame):
                output_audio.mux(packet)

        for packet in output_audio_stream.encode():
            output_audio.mux(packet)


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
        # A size of 0 preserves the unbounded queue used by serial encoding.
        # Parallel encoding uses a bounded queue; at the default capacity, eight
        # 1080p RGBA frames occupy about 66 MB per job. The worker drains through
        # the sentinel after an exception, so a bounded queue cannot deadlock.
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
        except BaseException as exception:
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
        self.init_audio()
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
    def init_audio(self) -> None:
        """Preps the writer for adding audio to the movie."""
        self.includes_sound = False

    def create_audio_segment(self) -> None:
        """Creates an empty, silent, Audio Segment."""
        self.audio_segment = AudioSegment.silent()

    def add_audio_segment(
        self,
        new_segment: AudioSegment,
        time: float | None = None,
        gain_to_background: float | None = None,
    ) -> None:
        """This method adds an audio segment from an AudioSegment type object
        and suitable parameters.

        Parameters
        ----------
        new_segment
            The audio segment to add

        time
            the timestamp at which the sound should be added.

        gain_to_background
            The gain of the segment from the background.
        """
        if not self.includes_sound:
            self.includes_sound = True
            self.create_audio_segment()
        segment = self.audio_segment
        curr_end = segment.duration_seconds
        if time is None:
            time = curr_end
        if time < 0:
            raise ValueError("Adding sound at timestamp < 0")

        new_end = time + new_segment.duration_seconds
        diff = new_end - curr_end
        if diff > 0:
            segment = segment.append(
                AudioSegment.silent(int(np.ceil(diff * 1000))),
                crossfade=0,
            )
        self.audio_segment = segment.overlay(
            new_segment,
            position=int(1000 * time),
            gain_during_overlay=gain_to_background,
        )

    def add_sound(
        self,
        sound_file: StrPath,
        time: float | None = None,
        gain: float | None = None,
        **kwargs: Any,
    ) -> None:
        """This method adds an audio segment from a sound file.

        Parameters
        ----------
        sound_file
            The path to the sound file.

        time
            The timestamp at which the audio should be added.

        gain
            The gain of the given audio segment.

        **kwargs
            This method uses add_audio_segment, so any keyword arguments
            used there can be referenced here.

        """
        file_path = get_full_sound_file_path(sound_file, self.settings.assets_dir)
        # we assume files with .wav / .raw suffix are actually
        # .wav and .raw files, respectively.
        if file_path.suffix not in (".wav", ".raw"):
            # we need to pass delete=False to work on Windows
            # TODO: figure out a way to cache the wav file generated (benchmark needed)
            with NamedTemporaryFile(suffix=".wav", delete=False) as wav_file_path:
                convert_audio(file_path, wav_file_path, "pcm_s16le")
                new_segment = AudioSegment.from_file(wav_file_path.name)
                logger.info(f"Automatically converted {file_path} to .wav")
            Path(wav_file_path.name).unlink()
        else:
            new_segment = AudioSegment.from_file(file_path)

        if gain:
            new_segment = new_segment.apply_gain(gain)
        self.add_audio_segment(new_segment, time, **kwargs)

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

    def end_animation(self, allow_write: bool = False) -> None:
        """Seal the current segment job when video writing is enabled.

        Parameters
        ----------
        allow_write
            Whether the current animation has an open segment job.
        """
        if self.output_spec.is_video and allow_write:
            self.close_partial_movie_stream()

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
        frame_queue_size = (
            0
            if self.settings.max_inflight_encoders == 1
            else self.settings.encoder_queue_size
        )
        self._current_encode_job = _PartialMovieEncodeJob(
            animation_index=animation_index,
            encoder=segment_encoder,
            frame_queue_size=frame_queue_size,
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
        includes_sound: bool = False,
    ) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Partial movie files to combine ({len(input_files)} files): %(p)s",
            {"p": input_files[:5]},
        )
        manifest = BytesIO(self._concat_manifest_bytes(input_files))

        av_options = {
            "safe": "0",  # needed to read files
        }

        if not includes_sound:
            av_options["an"] = "1"

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
            for packet in partial_movies_input.demux(partial_movies_stream):
                # We need to skip the "flushing" packets that `demux` generates.
                if packet.dts is None:
                    continue

                packet.dts = None  # This seems to be needed, as dts from consecutive
                # files may not be monotically increasing, so we let libav compute it.

                # We need to assign the packet to the new stream.
                packet.stream = output_stream
                output_container.mux(packet)

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
        self.combine_files(
            partial_movie_files,
            movie_file_path,
            self.output_spec.is_gif,
            self.includes_sound,
        )

        # handle sound
        if self.includes_sound and not self.output_spec.is_gif:
            sound_file_path = movie_file_path.with_suffix(".wav")
            # Makes sure sound file length will match video file
            self.add_audio_segment(AudioSegment.silent(0))
            self.audio_segment.export(
                sound_file_path,
                format="wav",
                bitrate="312k",
            )
            # Audio added to a VP9 encoded (webm) video file needs
            # to be encoded as vorbis or opus. Directly exporting
            # self.audio_segment with such a codec works in principle,
            # but tries to call ffmpeg via its CLI -- which we want
            # to avoid. This is why we need to do the conversion
            # manually.
            if self.output_spec.segment_extension == ".webm":
                ogg_sound_file_path = sound_file_path.with_suffix(".ogg")
                convert_audio(sound_file_path, ogg_sound_file_path, "libvorbis")
                sound_file_path = ogg_sound_file_path
            elif self.output_spec.segment_extension == ".mp4":
                # Similarly, pyav may reject wav audio in an .mp4 file;
                # convert to AAC.
                aac_sound_file_path = sound_file_path.with_suffix(".aac")
                convert_audio(sound_file_path, aac_sound_file_path, "aac")
                sound_file_path = aac_sound_file_path

            temp_file_path = movie_file_path.with_name(
                f"{movie_file_path.stem}_temp{movie_file_path.suffix}"
            )
            av_options = {
                "shortest": "1",
                "metadata": f"comment=Rendered with Manim Community v{__version__}",
            }

            with (
                av.open(movie_file_path) as video_input,
                av.open(sound_file_path) as audio_input,
            ):
                video_stream = video_input.streams.video[0]
                audio_stream = audio_input.streams.audio[0]
                output_container = av.open(
                    str(temp_file_path), mode="w", options=av_options
                )
                output_video_stream = output_container.add_stream_from_template(
                    template=video_stream
                )
                output_audio_stream = output_container.add_stream_from_template(
                    template=audio_stream
                )

                for packet in video_input.demux(video_stream):
                    # We need to skip the "flushing" packets that `demux` generates.
                    if packet.dts is None:
                        continue

                    # We need to assign the packet to the new stream.
                    packet.stream = output_video_stream
                    output_container.mux(packet)

                for packet in audio_input.demux(audio_stream):
                    # We need to skip the "flushing" packets that `demux` generates.
                    if packet.dts is None:
                        continue

                    # We need to assign the packet to the new stream.
                    packet.stream = output_audio_stream
                    output_container.mux(packet)

                output_container.close()

            shutil.move(str(temp_file_path), str(movie_file_path))
            sound_file_path.unlink()

        self.print_file_ready_message(str(movie_file_path))
        if self.output_spec.is_video:
            for file_path in partial_movie_files:
                # We have to modify the accessed time so if we have to clean the cache we remove the one used the longest.
                modify_atime(file_path)

    def combine_to_section_videos(self) -> None:
        """Concatenate partial movie files for each section."""
        self.finish_last_section()
        sections_index: list[dict[str, Any]] = []
        for section in self.sections:
            # only if section does want to be saved
            if section.video is not None:
                logger.info(f"Combining partial files for section '{section.name}'")
                section_path = self.sections_output_dir / section.video
                self.combine_files(
                    section.get_clean_partial_movie_files(),
                    section_path,
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
        subcaption_file.parent.mkdir(parents=True, exist_ok=True)
        subcaption_file.write_text(srt.compose(self.subcaptions), encoding="utf-8")
        logger.info(f"Subcaption file has been written as {subcaption_file}")

    def print_file_ready_message(self, file_path: StrPath) -> None:
        """Record and report a completed primary artifact."""
        self.final_file_path = Path(file_path)
        logger.info("\nFile ready at %(file_path)s\n", {"file_path": f"'{file_path}'"})
