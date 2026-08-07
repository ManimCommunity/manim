"""The interface between scenes and ffmpeg."""

from __future__ import annotations

__all__ = ["SceneFileWriter"]

import json
import shutil
import warnings
from contextlib import suppress
from fractions import Fraction
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

from .. import config, logger
from .._config.logger_utils import set_file_logger
from ..constants import RendererType
from ..utils.file_ops import (
    add_extension_if_not_present,
    add_version_before_extension,
    guarantee_existence,
    is_gif_format,
    is_png_format,
    modify_atime,
    write_to_movie,
)
from ..utils.sounds import get_full_sound_file_path
from .section import DefaultSectionType, Section

if TYPE_CHECKING:
    from av.container.output import OutputContainer
    from av.stream import Stream

    from manim.renderer.cairo_renderer import CairoRenderer
    from manim.renderer.opengl_renderer import OpenGLRenderer
    from manim.typing import PixelArray, StrPath


def to_av_frame_rate(fps: float) -> Fraction:
    epsilon1 = 1e-4
    epsilon2 = 0.02

    if isinstance(fps, int):
        (num, denom) = (fps, 1)
    elif abs(fps - round(fps)) < epsilon1:
        (num, denom) = (round(fps), 1)
    else:
        denom = 1001
        num = round(fps * denom / 1000) * 1000
        if abs(fps - num / denom) >= epsilon2:
            raise ValueError("invalid frame rate")

    return Fraction(num, denom)


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
    """Encode and write the frames for one partial movie file."""

    def __init__(
        self,
        path: StrPath,
        animation_index: int,
        container: OutputContainer,
        stream: Stream,
        frame_queue_size: int,
    ) -> None:
        self.path = path
        self.animation_index = animation_index
        self.container = container
        self.stream = stream
        # A size of 0 preserves the unbounded queue used by serial encoding.
        # Parallel encoding uses a bounded queue; at the default capacity, eight
        # 1080p RGBA frames occupy about 66 MB per job. The worker drains through
        # the sentinel after an exception, so a bounded queue cannot deadlock.
        self.queue: Queue[tuple[int, PixelArray | None]] = Queue(
            maxsize=frame_queue_size,
        )
        self._exception: BaseException | None = None
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

    def _listen_and_write(self) -> None:
        while True:
            num_frames, frame_data = self.queue.get()
            if frame_data is None:
                break
            if self._exception is not None:
                continue

            try:
                self._encode_and_write_frame(frame_data, num_frames)
            except BaseException as exception:
                self._capture_exception(exception)

        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        except BaseException as exception:
            self._capture_exception(exception)
        finally:
            try:
                self.container.close()
            except BaseException as exception:
                self._capture_exception(exception)

    def _encode_and_write_frame(self, frame: PixelArray, num_frames: int) -> None:
        for _ in range(num_frames):
            # Notes: precomputing reusing packets does not work!
            # I.e., you cannot do `packets = encode(...)`
            # and reuse it, as it seems that `mux(...)`
            # consumes the packet.
            # The same issue applies for `av_frame`,
            # reusing it renders weird-looking frames.
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgba")
            for packet in self.stream.encode(av_frame):
                self.container.mux(packet)

    def put(self, num_frames: int, frame: PixelArray) -> None:
        """Add a frame to the encoding queue."""
        self.queue.put((num_frames, frame))

    def seal(self) -> None:
        """Signal that no more frames will be added."""
        self.queue.put((-1, None))

    def join(self) -> None:
        """Wait for encoding to finish and propagate worker failures."""
        self.thread.join()
        if self._exception is not None:
            # A failed encode may leave a structurally valid but truncated
            # file behind; remove it so a later run cannot cache-hit it.
            try:
                Path(self.path).unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.warning(
                    "Failed to remove incomplete partial movie file %(path)s: "
                    "%(error)s",
                    {"path": f"'{self.path}'", "error": cleanup_error},
                )
            raise self._exception

        logger.info(
            f"Animation {self.animation_index} : Partial movie file written in %(path)s",
            {"path": f"'{self.path}'"},
        )


class SceneFileWriter:
    """SceneFileWriter is the object that actually writes the animations
    played, into video files, using FFMPEG.
    This is mostly for Manim's internal use. You will rarely, if ever,
    have to use the methods for this class, unless tinkering with the very
    fabric of Manim's reality.

    Attributes
    ----------
        sections : list of :class:`.Section`
            used to segment scene

        sections_output_dir : :class:`pathlib.Path`
            where are section videos stored

        output_name : str
            name of movie without extension and basis for section video names

    Some useful attributes are:
        "write_to_movie" (bool=False)
            Whether or not to write the animations into a video file.
        "movie_file_extension" (str=".mp4")
            The file-type extension of the outputted video.
        "partial_movie_files"
            List of all the partial-movie files.

    """

    force_output_as_scene_name = False

    def __init__(
        self,
        renderer: CairoRenderer | OpenGLRenderer,
        scene_name: str,
        **kwargs: Any,
    ) -> None:
        self.renderer = renderer
        self._inflight_encode_jobs: list[_PartialMovieEncodeJob] = []
        self._inflight_by_path: dict[str, _PartialMovieEncodeJob] = {}
        self._current_encode_job: _PartialMovieEncodeJob | None = None
        self.init_output_directories(scene_name)
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

    def init_output_directories(self, scene_name: str) -> None:
        """Initialise output directories.

        Notes
        -----
        The directories are read from ``config``, for example
        ``config['media_dir']``.  If the target directories don't already
        exist, they will be created.

        """
        if config["dry_run"]:  # in dry-run mode there is no output
            return

        module_name = config.get_dir("input_file").stem if config["input_file"] else ""

        if SceneFileWriter.force_output_as_scene_name:
            self.output_name = Path(scene_name)
        elif config["output_file"] and not config["write_all"]:
            self.output_name = config.get_dir("output_file")
        else:
            self.output_name = Path(scene_name)

        if config["media_dir"]:
            image_dir = guarantee_existence(
                config.get_dir(
                    "images_dir", module_name=module_name, scene_name=scene_name
                ),
            )
            self.image_file_path = image_dir / add_extension_if_not_present(
                self.output_name, ".png"
            )

        if write_to_movie():
            movie_dir = guarantee_existence(
                config.get_dir(
                    "video_dir", module_name=module_name, scene_name=scene_name
                ),
            )
            self.movie_file_path = movie_dir / add_extension_if_not_present(
                self.output_name, config["movie_file_extension"]
            )

            # TODO: /dev/null would be good in case sections_output_dir is used without being set (doesn't work on Windows), everyone likes defensive programming, right?
            self.sections_output_dir = Path("")
            if config.save_sections:
                self.sections_output_dir = guarantee_existence(
                    config.get_dir(
                        "sections_dir", module_name=module_name, scene_name=scene_name
                    )
                )

            if is_gif_format():
                self.gif_file_path = add_extension_if_not_present(
                    self.output_name, ".gif"
                )

                if not config["output_file"]:
                    self.gif_file_path = add_version_before_extension(
                        self.gif_file_path
                    )

                self.gif_file_path = movie_dir / self.gif_file_path

            self.partial_movie_directory = guarantee_existence(
                config.get_dir(
                    "partial_movie_dir",
                    scene_name=scene_name,
                    module_name=module_name,
                ),
            )

            if config["log_to_file"]:
                log_dir = guarantee_existence(config.get_dir("log_dir"))
                set_file_logger(
                    scene_name=scene_name, module_name=module_name, log_dir=log_dir
                )

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
        if (
            not config.dry_run
            and write_to_movie()
            and config.save_sections
            and not skip_animations
        ):
            # relative to index file
            section_video = f"{self.output_name}_{len(self.sections):04}_{name}{config.movie_file_extension}"

        self.sections.append(
            Section(
                type_,
                section_video,
                name,
                skip_animations,
            ),
        )

    def add_partial_movie_file(self, hash_animation: str | None) -> None:
        """Adds a new partial movie file path to ``scene.partial_movie_files``
        and current section from a hash.

        This method will compute the path from the hash. In addition to that it
        adds the new animation to the current section.

        Parameters
        ----------
        hash_animation
            Hash of the animation.
        """
        if not hasattr(self, "partial_movie_directory") or not write_to_movie():
            return

        # None has to be added to partial_movie_files to keep the right index with scene.num_plays.
        # i.e if an animation is skipped, scene.num_plays is still incremented and we add an element to partial_movie_file be even with num_plays.
        if hash_animation is None:
            self.partial_movie_files.append(None)
            self.sections[-1].partial_movie_files.append(None)
        else:
            new_partial_movie_file = str(
                self.partial_movie_directory
                / f"{hash_animation}{config['movie_file_extension']}"
            )
            self.partial_movie_files.append(new_partial_movie_file)
            self.sections[-1].partial_movie_files.append(new_partial_movie_file)

    def get_resolution_directory(self) -> str:
        """Get the name of the resolution directory directly containing
        the video file.

        This method gets the name of the directory that immediately contains the
        video file. This name is ``<height_in_pixels_of_video>p<frame_rate>``.
        For example, if you are rendering an 854x480 px animation at 15fps,
        the name of the directory that immediately contains the video,  file
        will be ``480p15``.

        The file structure should look something like::

            MEDIA_DIR
                |--Tex
                |--texts
                |--videos
                    |--<name_of_file_containing_scene>
                        |--<height_in_pixels_of_video>p<frame_rate>
                            |--partial_movie_files
                            |--<scene_name>.mp4
                            |--<scene_name>.srt

        Returns
        -------
        :class:`str`
            The name of the directory.
        """
        pixel_height = config["pixel_height"]
        frame_rate = config["frame_rate"]
        return f"{pixel_height}p{frame_rate}"

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
        file_path = get_full_sound_file_path(sound_file)
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
        self, allow_write: bool = False, file_path: StrPath | None = None
    ) -> None:
        """Used internally by manim to stream the animation to FFMPEG for
        displaying or writing to a file.

        Parameters
        ----------
        allow_write
            Whether or not to write to a video file.
        """
        if write_to_movie() and allow_write:
            self.open_partial_movie_stream(file_path=file_path)

    def end_animation(self, allow_write: bool = False) -> None:
        """Internally used by Manim to stop streaming to FFMPEG gracefully.

        Parameters
        ----------
        allow_write
            Whether or not to write to a video file.
        """
        if write_to_movie() and allow_write:
            self.close_partial_movie_stream()

    def write_frame(
        self, frame_or_renderer: PixelArray | OpenGLRenderer, num_frames: int = 1
    ) -> None:
        """Used internally by Manim to write a frame to the FFMPEG input buffer.

        Parameters
        ----------
        frame_or_renderer
            Pixel array of the frame.
        num_frames
            The number of times to write frame.
        """
        if write_to_movie():
            if isinstance(frame_or_renderer, np.ndarray):
                frame = frame_or_renderer
            else:
                frame = (
                    frame_or_renderer.get_frame()
                    if config.renderer == RendererType.OPENGL
                    else frame_or_renderer
                )

            job = self._current_encode_job
            if job is None:
                # Interactive OpenGL rendering emits frames outside an open
                # partial movie stream; drop them silently.
                return
            if job.failed:
                # Surface the failure at the first write after it was captured;
                # join() unlinks the partial and re-raises.
                job.seal()
                self._current_encode_job = None
                job.join()
            job.put(num_frames, frame)

        if is_png_format() and not config["dry_run"]:
            if isinstance(frame_or_renderer, np.ndarray):
                image = Image.fromarray(frame_or_renderer)
            else:
                image = (
                    frame_or_renderer.get_image()
                    if config.renderer == RendererType.OPENGL
                    else Image.fromarray(frame_or_renderer)
                )
            target_dir = self.image_file_path.parent / self.image_file_path.stem
            extension = self.image_file_path.suffix
            self.output_image(
                image,
                target_dir,
                extension,
                config["zero_pad"],
            )

    def output_image(
        self, image: Image.Image, target_dir: StrPath, ext: str, zero_pad: int
    ) -> None:
        if zero_pad:
            image.save(f"{target_dir}{str(self.frame_count).zfill(zero_pad)}{ext}")
        else:
            image.save(f"{target_dir}{self.frame_count}{ext}")
        self.frame_count += 1

    def save_image(self, image: Image.Image) -> None:
        """This method saves the image passed to it in the default image directory.

        Parameters
        ----------
        image
            The pixel array of the image to save.
        """
        if config["dry_run"]:
            return
        if not config["output_file"]:
            self.image_file_path = add_version_before_extension(self.image_file_path)

        image.save(self.image_file_path)
        self.print_file_ready_message(self.image_file_path)

    def finish(self) -> None:
        """Finishes writing to the FFMPEG buffer or writing images to output directory.
        Combines the partial movie files into the whole scene.
        If save_last_frame is True, saves the last frame in the default image directory.
        """
        if write_to_movie():
            self.join_all_encode_jobs()
            self.combine_to_movie()
            if config.save_sections:
                self.combine_to_section_videos()
            # Cache cleanup runs after the in-flight encode jobs have been drained.
            if config["flush_cache"]:
                self.flush_cache_directory()
            else:
                self.clean_cache()
        elif is_png_format() and not config["dry_run"]:
            target_dir = self.image_file_path.parent / self.image_file_path.stem
            logger.info("\n%i images ready at %s\n", self.frame_count, str(target_dir))
        if self.subcaptions:
            self.write_subcaption_file()

    def open_partial_movie_stream(self, file_path: StrPath | None = None) -> None:
        """Open a container holding a video stream.

        This is used internally by Manim initialize the container holding
        the video stream of a partial movie file.
        """
        if file_path is None:
            file_path = self.partial_movie_files[self.renderer.num_plays]
            if file_path is None:
                raise RuntimeError(
                    "open_partial_movie_stream() called for a play that has no "
                    "partial movie file path.",
                )
        path_key = str(file_path)
        if path_key in self._inflight_by_path:
            self._join_job_and_drain_on_failure(self._inflight_by_path[path_key])
        self.partial_movie_file_path = file_path

        fps = to_av_frame_rate(config.frame_rate)

        partial_movie_file_codec = "libx264"
        partial_movie_file_pix_fmt = "yuv420p"
        av_options = {
            "an": "1",  # ffmpeg: -an, no audio
            "crf": "23",  # ffmpeg: -crf, constant rate factor (improved bitrate)
        }

        if config.movie_file_extension == ".webm":
            partial_movie_file_codec = "libvpx-vp9"
            av_options["-auto-alt-ref"] = "1"
            if config.transparent:
                partial_movie_file_pix_fmt = "yuva420p"

        elif config.transparent:
            partial_movie_file_codec = "qtrle"
            partial_movie_file_pix_fmt = "argb"

        video_container = av.open(file_path, mode="w")
        stream = video_container.add_stream(
            partial_movie_file_codec,
            rate=fps,
            options=av_options,
        )
        stream.pix_fmt = partial_movie_file_pix_fmt
        stream.width = config.pixel_width
        stream.height = config.pixel_height

        frame_queue_size = (
            0 if config.max_inflight_encoders == 1 else config.encoder_queue_size
        )
        self._current_encode_job = _PartialMovieEncodeJob(
            path=file_path,
            animation_index=self.renderer.num_plays,
            container=video_container,
            stream=stream,
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
        """Tear down encode jobs after an aborted or rerun render.

        Seals the current job so its worker can exit (a non-daemon thread
        blocked on the queue would hang the process at exit), then deletes its
        partial file unconditionally: an aborted partial is structurally valid
        but truncated, so leaving it behind produces an erroneous cache hit on
        a later run. Sealed in-flight jobs are then drained. With
        ``reraise_encoder_failures=False`` (a render exception is already
        propagating) drain failures are logged, not raised; with ``True``
        (rerun path -- no primary exception exists) the first drain failure
        propagates so corrupt completed partials cannot be silently ignored.
        """
        current_exception: BaseException | None = None
        job = self._current_encode_job
        if job is not None:
            # Seal before clearing: an interrupt landing between the two
            # statements must not orphan the worker.
            job.seal()
            self._current_encode_job = None
            job.thread.join()
            current_exception = job._exception
            if current_exception is not None:
                logger.error(
                    "Encoder for aborted animation %d had also failed",
                    job.animation_index,
                    exc_info=current_exception,
                )
            try:
                Path(job.path).unlink(missing_ok=True)
                logger.info(
                    "Discarded partial movie file of aborted animation %(index)d",
                    {"index": job.animation_index},
                )
            except OSError as cleanup_error:
                logger.warning(
                    "Failed to remove incomplete partial movie file %(path)s: "
                    "%(error)s",
                    {"path": f"'{job.path}'", "error": cleanup_error},
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
        """Close the currently opened video container.

        Used internally by Manim to first flush the remaining packages
        in the video stream holding a partial file, and then close
        the corresponding container.
        """
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

        while len(self._inflight_encode_jobs) >= config.max_inflight_encoders:
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
        if not hasattr(self, "partial_movie_directory") or not write_to_movie():
            return False
        path = (
            self.partial_movie_directory
            / f"{hash_invocation}{config['movie_file_extension']}"
        )
        path_key = str(path)
        if path_key in self._inflight_by_path:
            self._join_job_and_drain_on_failure(self._inflight_by_path[path_key])
        return path.exists()

    def combine_files(
        self,
        input_files: list[str],
        output_file: Path,
        create_gif: bool = False,
        includes_sound: bool = False,
    ) -> None:
        file_list = self.partial_movie_directory / "partial_movie_file_list.txt"
        logger.debug(
            f"Partial movie files to combine ({len(input_files)} files): %(p)s",
            {"p": input_files[:5]},
        )
        with file_list.open("w", encoding="utf-8") as fp:
            fp.write("# This file is used internally by FFMPEG.\n")
            for pf_path in input_files:
                pf_path = Path(pf_path).as_posix()
                fp.write(f"file 'file:{pf_path}'\n")

        av_options = {
            "safe": "0",  # needed to read files
        }

        if not includes_sound:
            av_options["an"] = "1"

        partial_movies_input = av.open(
            str(file_list), options=av_options, format="concat"
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
            if config.transparent:
                output_stream.pix_fmt = "pal8"
            output_stream.width = config.pixel_width
            output_stream.height = config.pixel_height
            output_stream.rate = to_av_frame_rate(config.frame_rate)
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
            if config.transparent and config.movie_file_extension == ".webm":
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
        if is_gif_format():
            movie_file_path = self.gif_file_path

        if len(partial_movie_files) == 0:  # Prevent calling concat on empty list
            logger.info("No animations are contained in this scene.")
            return

        logger.info("Combining to Movie file.")
        self.combine_files(
            partial_movie_files,
            movie_file_path,
            is_gif_format(),
            self.includes_sound,
        )

        # handle sound
        if self.includes_sound and config.format != "gif":
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
            if config.movie_file_extension == ".webm":
                ogg_sound_file_path = sound_file_path.with_suffix(".ogg")
                convert_audio(sound_file_path, ogg_sound_file_path, "libvorbis")
                sound_file_path = ogg_sound_file_path
            elif config.movie_file_extension == ".mp4":
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
        if write_to_movie():
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
                self.combine_files(
                    section.get_clean_partial_movie_files(),
                    self.sections_output_dir / section.video,
                )
                sections_index.append(section.get_dict(self.sections_output_dir))
        with (self.sections_output_dir / f"{self.output_name}.json").open("w") as file:
            json.dump(sections_index, file, indent=4)

    def clean_cache(self) -> None:
        """Will clean the cache by removing the oldest partial_movie_files."""
        cached_partial_movies = [
            (self.partial_movie_directory / file_name)
            for file_name in self.partial_movie_directory.iterdir()
            if file_name != "partial_movie_file_list.txt"
        ]
        if len(cached_partial_movies) > config["max_files_cached"]:
            number_files_to_delete = (
                len(cached_partial_movies) - config["max_files_cached"]
            )
            oldest_files_to_delete = sorted(
                cached_partial_movies,
                key=lambda path: path.stat().st_atime,
            )[:number_files_to_delete]
            for file_to_delete in oldest_files_to_delete:
                file_to_delete.unlink()
            logger.info(
                f"The partial movie directory is full (> {config['max_files_cached']} files). Therefore, manim has removed the {number_files_to_delete} oldest file(s)."
                " You can change this behaviour by changing max_files_cached in config.",
            )

    def flush_cache_directory(self) -> None:
        """Delete all the cached partial movie files"""
        cached_partial_movies = [
            self.partial_movie_directory / file_name
            for file_name in self.partial_movie_directory.iterdir()
            if file_name != "partial_movie_file_list.txt"
        ]
        for f in cached_partial_movies:
            f.unlink()
        logger.info(
            f"Cache flushed. {len(cached_partial_movies)} file(s) deleted in %(par_dir)s.",
            {"par_dir": self.partial_movie_directory},
        )

    def write_subcaption_file(self) -> None:
        """Writes the subcaption file."""
        if config.output_file is None:
            return
        subcaption_file = Path(config.output_file).with_suffix(".srt")
        subcaption_file.write_text(srt.compose(self.subcaptions), encoding="utf-8")
        logger.info(f"Subcaption file has been written as {subcaption_file}")

    def print_file_ready_message(self, file_path: StrPath) -> None:
        """Prints the "File Ready" message to STDOUT."""
        config["output_file"] = file_path
        logger.info("\nFile ready at %(file_path)s\n", {"file_path": f"'{file_path}'"})
