"""The interface between scenes and ffmpeg."""

from __future__ import annotations

__all__ = ["SceneFileWriter"]

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import srt
from PIL import Image
from pydub import AudioSegment

from manim import __version__

from .. import config, logger
from .._config.logger_utils import set_file_logger
from ..constants import RendererType
from ..utils.file_ops import (
    add_extension_if_not_present,
    add_version_before_extension,
    ensure_executable,
    guarantee_existence,
    is_gif_format,
    is_png_format,
    is_webm_format,
    modify_atime,
    write_to_movie,
)
from ..utils.sounds import get_full_sound_file_path
from .section import DefaultSectionType, Section

if TYPE_CHECKING:
    from manim.renderer.opengl_renderer import OpenGLRenderer


class SceneFileWriter:
    """
    SceneFileWriter is the object that actually writes the animations
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

    def __init__(self, renderer, scene_name, **kwargs):
        self.renderer = renderer
        self.init_output_directories(scene_name)
        self.init_audio()
        self.frame_count = 0
        self.partial_movie_files: list[str] = []
        self.subcaptions: list[srt.Subtitle] = []
        self.sections: list[Section] = []
        # first section gets automatically created for convenience
        # if you need the first section to be skipped, add a first section by hand, it will replace this one
        self.next_section(
            name="autocreated", type=DefaultSectionType.NORMAL, skip_animations=False
        )
        # fail fast if ffmpeg is not found
        if not ensure_executable(Path(config.ffmpeg_executable)):
            raise RuntimeError(
                "Manim could not find ffmpeg, which is required for generating video output.\n"
                "For installing ffmpeg please consult https://docs.manim.community/en/stable/installation.html\n"
                "Make sure to either add ffmpeg to the PATH environment variable\n"
                "or set path to the ffmpeg executable under the ffmpeg header in Manim's configuration."
            )

    def init_output_directories(self, scene_name):
        """Initialise output directories.

        Notes
        -----
        The directories are read from ``config``, for example
        ``config['media_dir']``.  If the target directories don't already
        exist, they will be created.

        """
        if config["dry_run"]:  # in dry-run mode there is no output
            return

        if config["input_file"]:
            module_name = config.get_dir("input_file").stem
        else:
            module_name = ""

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

            # TODO: /dev/null would be good in case sections_output_dir is used without bein set (doesn't work on Windows), everyone likes defensive programming, right?
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

    def next_section(self, name: str, type: str, skip_animations: bool) -> None:
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
                type,
                section_video,
                name,
                skip_animations,
            ),
        )

    def add_partial_movie_file(self, hash_animation: str):
        """Adds a new partial movie file path to `scene.partial_movie_files` and current section from a hash.
        This method will compute the path from the hash. In addition to that it adds the new animation to the current section.

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

    def get_resolution_directory(self):
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
                        |--<scene_name>.mp4

        Returns
        -------
        :class:`str`
            The name of the directory.
        """
        pixel_height = config["pixel_height"]
        frame_rate = config["frame_rate"]
        return f"{pixel_height}p{frame_rate}"

    # Sound
    def init_audio(self):
        """
        Preps the writer for adding audio to the movie.
        """
        self.includes_sound = False

    def create_audio_segment(self):
        """
        Creates an empty, silent, Audio Segment.
        """
        self.audio_segment = AudioSegment.silent()

    def add_audio_segment(
        self,
        new_segment: AudioSegment,
        time: float | None = None,
        gain_to_background: float | None = None,
    ):
        """
        This method adds an audio segment from an
        AudioSegment type object and suitable parameters.

        Parameters
        ----------
        new_segment
            The audio segment to add

        time
            the timestamp at which the
            sound should be added.

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
        sound_file: str,
        time: float | None = None,
        gain: float | None = None,
        **kwargs,
    ):
        """
        This method adds an audio segment from a sound file.

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
        new_segment = AudioSegment.from_file(file_path)
        if gain:
            new_segment = new_segment.apply_gain(gain)
        self.add_audio_segment(new_segment, time, **kwargs)

    # Writers
    def begin_animation(self, allow_write: bool = False, file_path=None):
        """
        Used internally by manim to stream the animation to FFMPEG for
        displaying or writing to a file.

        Parameters
        ----------
        allow_write
            Whether or not to write to a video file.
        """
        if write_to_movie() and allow_write:
            self.open_movie_pipe(file_path=file_path)

    def end_animation(self, allow_write: bool = False):
        """
        Internally used by Manim to stop streaming to
        FFMPEG gracefully.

        Parameters
        ----------
        allow_write
            Whether or not to write to a video file.
        """
        if write_to_movie() and allow_write:
            self.close_movie_pipe()

    def write_frame(self, frame_or_renderer: np.ndarray | OpenGLRenderer):
        """
        Used internally by Manim to write a frame to
        the FFMPEG input buffer.

        Parameters
        ----------
        frame_or_renderer
            Pixel array of the frame.
        """
        if config.renderer == RendererType.OPENGL:
            self.write_opengl_frame(frame_or_renderer)
        elif config.renderer == RendererType.CAIRO:
            frame = frame_or_renderer
            if write_to_movie():
                self.writing_process.stdin.write(frame.tobytes())
            if is_png_format() and not config["dry_run"]:
                self.output_image_from_array(frame)

    def write_opengl_frame(self, renderer: OpenGLRenderer):
        if write_to_movie():
            self.writing_process.stdin.write(
                renderer.get_raw_frame_buffer_object_data(),
            )
        elif is_png_format() and not config["dry_run"]:
            target_dir = self.image_file_path.parent / self.image_file_path.stem
            extension = self.image_file_path.suffix
            self.output_image(
                renderer.get_image(),
                target_dir,
                extension,
                config["zero_pad"],
            )

    def output_image_from_array(self, frame_data):
        target_dir = self.image_file_path.parent / self.image_file_path.stem
        extension = self.image_file_path.suffix
        self.output_image(
            Image.fromarray(frame_data),
            target_dir,
            extension,
            config["zero_pad"],
        )

    def output_image(self, image: Image.Image, target_dir, ext, zero_pad: bool):
        if zero_pad:
            image.save(f"{target_dir}{str(self.frame_count).zfill(zero_pad)}{ext}")
        else:
            image.save(f"{target_dir}{self.frame_count}{ext}")
        self.frame_count += 1

    def save_final_image(self, image: np.ndarray):
        """
        The name is a misnomer. This method saves the image
        passed to it as an in the default image directory.

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

    def finish(self):
        """
        Finishes writing to the FFMPEG buffer or writing images
        to output directory.
        Combines the partial movie files into the
        whole scene.
        If save_last_frame is True, saves the last
        frame in the default image directory.
        """
        if write_to_movie():
            if hasattr(self, "writing_process"):
                self.writing_process.terminate()
            self.combine_to_movie()
            if config.save_sections:
                self.combine_to_section_videos()
            if config["flush_cache"]:
                self.flush_cache_directory()
            else:
                self.clean_cache()
        elif is_png_format() and not config["dry_run"]:
            target_dir = self.image_file_path.parent / self.image_file_path.stem
            logger.info("\n%i images ready at %s\n", self.frame_count, str(target_dir))
        if self.subcaptions:
            self.write_subcaption_file()

    def open_movie_pipe(self, file_path=None):
        """
        Used internally by Manim to initialise
        FFMPEG and begin writing to FFMPEG's input
        buffer.
        """
        if file_path is None:
            file_path = self.partial_movie_files[self.renderer.num_plays]
        self.partial_movie_file_path = file_path

        fps = config["frame_rate"]
        if fps == int(fps):  # fps is integer
            fps = int(fps)
        if config.renderer == RendererType.OPENGL:
            width, height = self.renderer.get_pixel_shape()
        else:
            height = config["pixel_height"]
            width = config["pixel_width"]

        command = [
            config.ffmpeg_executable,
            "-y",  # overwrite output file if it exists
            "-f",
            "rawvideo",
            "-s",
            "%dx%d" % (width, height),  # size of one frame
            "-pix_fmt",
            "rgba",
            "-r",
            str(fps),  # frames per second
            "-i",
            "-",  # The input comes from a pipe
            "-an",  # Tells FFMPEG not to expect any audio
            "-loglevel",
            config["ffmpeg_loglevel"].lower(),
            "-metadata",
            f"comment=Rendered with Manim Community v{__version__}",
        ]
        if config.renderer == RendererType.OPENGL:
            command += ["-vf", "vflip"]
        if is_webm_format():
            command += ["-vcodec", "libvpx-vp9", "-auto-alt-ref", "0"]
        # .mov format
        elif config["transparent"]:
            command += ["-vcodec", "qtrle"]
        else:
            command += ["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
        command += [file_path]
        self.writing_process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def close_movie_pipe(self):
        """
        Used internally by Manim to gracefully stop writing to FFMPEG's input buffer
        """
        self.writing_process.stdin.close()
        self.writing_process.wait()

        logger.info(
            f"Animation {self.renderer.num_plays} : Partial movie file written in %(path)s",
            {"path": f"'{self.partial_movie_file_path}'"},
        )

    def is_already_cached(self, hash_invocation: str):
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
        return path.exists()

    def combine_files(
        self,
        input_files: list[str],
        output_file: Path,
        create_gif=False,
        includes_sound=False,
    ):
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
        commands = [
            config.ffmpeg_executable,
            "-y",  # overwrite output file if it exists
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list),
            "-loglevel",
            config.ffmpeg_loglevel.lower(),
            "-metadata",
            f"comment=Rendered with Manim Community v{__version__}",
            "-nostdin",
        ]

        if create_gif:
            commands += [
                "-vf",
                f"fps={np.clip(config['frame_rate'], 1, 50)},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            ]
        else:
            commands += ["-c", "copy"]

        if not includes_sound:
            commands += ["-an"]

        commands += [str(output_file)]

        combine_process = subprocess.Popen(commands)
        combine_process.wait()

    def combine_to_movie(self):
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
        logger.info("Combining to Movie file.")
        self.combine_files(
            partial_movie_files,
            movie_file_path,
            is_gif_format(),
            self.includes_sound,
        )

        # handle sound
        if self.includes_sound:
            sound_file_path = movie_file_path.with_suffix(".wav")
            # Makes sure sound file length will match video file
            self.add_audio_segment(AudioSegment.silent(0))
            self.audio_segment.export(
                sound_file_path,
                bitrate="312k",
            )
            temp_file_path = movie_file_path.with_name(
                f"{movie_file_path.stem}_temp{movie_file_path.suffix}"
            )
            commands = [
                config.ffmpeg_executable,
                "-i",
                str(movie_file_path),
                "-i",
                str(sound_file_path),
                "-y",  # overwrite output file if it exists
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "320k",
                # select video stream from first file
                "-map",
                "0:v:0",
                # select audio stream from second file
                "-map",
                "1:a:0",
                "-loglevel",
                config.ffmpeg_loglevel.lower(),
                "-metadata",
                f"comment=Rendered with Manim Community v{__version__}",
                # "-shortest",
                str(temp_file_path),
            ]
            subprocess.call(commands)
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

    def clean_cache(self):
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

    def flush_cache_directory(self):
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

    def write_subcaption_file(self):
        """Writes the subcaption file."""
        if config.output_file is None:
            return
        subcaption_file = Path(config.output_file).with_suffix(".srt")
        subcaption_file.write_text(srt.compose(self.subcaptions), encoding="utf-8")
        logger.info(f"Subcaption file has been written as {subcaption_file}")

    def print_file_ready_message(self, file_path):
        """Prints the "File Ready" message to STDOUT."""
        config["output_file"] = file_path
        logger.info("\nFile ready at %(file_path)s\n", {"file_path": f"'{file_path}'"})
