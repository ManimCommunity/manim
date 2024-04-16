from __future__ import annotations

import os
import pathlib as path
import platform
import shutil
import subprocess as sp
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm as ProgressDisplay

from manim import config
from manim._config import logger as log
from manim.mobject.mobject import Mobject
from manim.utils.file_ops import (
    add_extension_if_not_present,
    get_sorted_integer_files,
    guarantee_existence,
)
from manim.utils.sounds import get_full_sound_file_path

if TYPE_CHECKING:
    from PIL.Image import Image

    from manim.camera.camera import OpenGLCamera as Camera


class FileWriter:
    def __init__(
        self,
        file_name: str,
        write_to_movie: bool = False,
        break_into_partial_movies: bool = False,
        save_pngs: bool = False,  # TODO, this currently does nothing
        png_mode: str = "RGBA",
        save_last_frame: bool = False,
        movie_file_extension: str = ".mp4",
        # What python file is generating this scene
        input_file_path: str = "",
        # Where should this be written
        output_directory: str | None = None,
        open_file_upon_completion: bool = False,
        show_file_location_upon_completion: bool = False,
        quiet: bool = False,
        total_frames: int = 0,
        progress_description_len: int = 40,
    ):
        self.frames: list[Image] = []
        self.write_to_movie = write_to_movie
        self.break_into_partial_movies = break_into_partial_movies
        self.save_pngs = save_pngs
        self.png_mode = png_mode
        self.save_last_frame = save_last_frame
        self.movie_file_extension = movie_file_extension
        self.input_file_path = input_file_path
        self.output_directory = output_directory
        self.file_name = file_name
        self.open_file_upon_completion = open_file_upon_completion
        self.show_file_location_upon_completion = show_file_location_upon_completion
        self.quiet = quiet
        self.total_frames = total_frames
        self.progress_description_len = progress_description_len

        # State during file writing
        self.writing_process: sp.Popen | None = None
        self.progress_display: ProgressDisplay | None = None
        self.ended_with_interrupt: bool = False
        self.init_output_directories()
        self.init_audio()

    # Output directories and files
    def init_output_directories(self) -> None:
        out_dir = self.output_directory or ""
        scene_name = Path(self.file_name)
        if self.save_last_frame:
            image_dir = guarantee_existence(Path(out_dir) / "images")
            image_file = add_extension_if_not_present(scene_name, ".png")
            self.image_file_path = Path(image_dir) / image_file
        if self.write_to_movie:
            movie_dir = guarantee_existence(Path(out_dir) / "videos")
            movie_file = add_extension_if_not_present(
                scene_name, self.movie_file_extension
            )
            self.movie_file_path = Path(movie_dir) / movie_file
            if self.break_into_partial_movies:
                self.partial_movie_directory = guarantee_existence(
                    Path(movie_dir) / "partial_movie_files" / scene_name,
                )
        # A place to save mobjects
        self.saved_mobject_directory = Path(out_dir) / "mobjects" / scene_name

    def add_frames(self, *frames: Image) -> None:
        self.frames.extend(frames)

    def get_default_module_directory(self) -> str:
        path, _ = os.path.splitext(self.input_file_path)
        path = path.removeprefix("_")
        return path

    # Directory getters
    def get_image_file_path(self) -> str:
        return self.image_file_path

    # Sound
    def init_audio(self) -> None:
        self.includes_sound: bool = False

    def create_audio_segment(self) -> None:
        self.audio_segment = AudioSegment.silent()

    def add_audio_segment(
        self,
        new_segment: AudioSegment,
        time: float | None = None,
        gain_to_background: float | None = None,
    ) -> None:
        if not self.includes_sound:
            self.includes_sound = True
            self.create_audio_segment()
        segment = self.audio_segment
        curr_end = segment.duration_seconds
        if time is None:
            time = curr_end
        if time < 0:
            raise Exception("Adding sound at timestamp < 0")

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
        gain_to_background: float | None = None,
    ) -> None:
        file_path = get_full_sound_file_path(sound_file)
        new_segment = AudioSegment.from_file(file_path)
        if gain:
            new_segment = new_segment.apply_gain(gain)
        self.add_audio_segment(new_segment, time, gain_to_background)

    # Writers
    def begin(self) -> None:
        if not self.break_into_partial_movies and self.write_to_movie:
            self.open_movie_pipe(self.get_movie_file_path())

    def begin_animation(self) -> None:
        if self.break_into_partial_movies and self.write_to_movie:
            # self.open_movie_pipe(self.get_next_partial_movie_path())
            ...

    def end_animation(self) -> None:
        if self.break_into_partial_movies and self.write_to_movie:
            # self.close_movie_pipe()
            ...

    def finish(self) -> None:
        if self.write_to_movie:
            if self.break_into_partial_movies:
                self.combine_movie_files()
            else:
                self.close_movie_pipe()
            if self.includes_sound:
                self.add_sound_to_video()
            self.print_file_ready_message(self.get_movie_file_path())
        if self.save_last_frame:
            self.save_final_image(self.scene.get_image())
        if self.should_open_file():
            self.open_file()

    def open_movie_pipe(self, file_path: str) -> None:
        stem, ext = os.path.splitext(file_path)
        self.final_file_path = file_path
        self.temp_file_path = stem + "_temp" + ext

        fps = self.scene.camera.fps
        width, height = self.scene.camera.get_pixel_shape()

        command = [
            config.ffmpeg_executable,
            "-y",  # overwrite output file if it exists
            "-f",
            "rawvideo",
            "-s",
            f"{width}x{height}",  # size of one frame
            "-pix_fmt",
            "rgba",
            "-r",
            str(fps),  # frames per second
            "-i",
            "-",  # The input comes from a pipe
            "-vf",
            "vflip",
            "-an",  # Tells FFMPEG not to expect any audio
            "-loglevel",
            "error",
        ]
        if self.movie_file_extension == ".mov":
            # This is if the background of the exported
            # video should be transparent.
            command += [
                "-vcodec",
                "prores_ks",
            ]
        elif self.movie_file_extension != ".gif":
            command += [
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ]
        command += [self.temp_file_path]
        self.writing_process = sp.Popen(command, stdin=sp.PIPE)

        if self.total_frames > 0 and not self.quiet:
            self.progress_display = ProgressDisplay(
                range(self.total_frames),
                # bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}",
                leave=False,
                ascii=True if platform.system() == "Windows" else None,
                dynamic_ncols=True,
            )
            self.set_progress_display_description()

    def has_progress_display(self):
        return self.progress_display is not None

    def set_progress_display_description(
        self, file: str = "", sub_desc: str = ""
    ) -> None:
        if self.progress_display is None:
            return

        desc_len = self.progress_description_len
        if not file:
            file = os.path.split(self.get_movie_file_path())[1]
        full_desc = f"{file} {sub_desc}"
        if len(full_desc) > desc_len:
            full_desc = full_desc[: desc_len - 3] + "..."
        else:
            full_desc += " " * (desc_len - len(full_desc))
        self.progress_display.set_description(full_desc)

    def write_frame(self, frame: Image) -> None:
        if self.write_to_movie:
            self.writing_process.stdin.write(frame.tobytes("utf-8"))
            if self.progress_display is not None:
                self.progress_display.update()

    def close_movie_pipe(self) -> None:
        self.writing_process.stdin.close()
        self.writing_process.wait()
        self.writing_process.terminate()
        if self.progress_display is not None:
            self.progress_display.close()

        if not self.ended_with_interrupt:
            shutil.move(self.temp_file_path, self.final_file_path)
        else:
            self.movie_file_path = self.temp_file_path

    def combine_movie_files(self) -> None:
        kwargs = {
            "remove_non_integer_files": True,
            "extension": self.movie_file_extension,
        }
        if self.scene.start_at_animation_number is not None:
            kwargs["min_index"] = self.scene.start_at_animation_number
        if self.scene.end_at_animation_number is not None:
            kwargs["max_index"] = self.scene.end_at_animation_number
        else:
            kwargs["remove_indices_greater_than"] = self.scene.num_plays - 1
        partial_movie_files = get_sorted_integer_files(
            self.partial_movie_directory, **kwargs
        )
        if len(partial_movie_files) == 0:
            log.warning("No animations in this scene")
            return

        # Write a file partial_file_list.txt containing all
        # partial movie files
        file_list = Path(self.partial_movie_directory) / "partial_movie_file_list.txt"
        with open(file_list, "w") as fp:
            for pf_path in partial_movie_files:
                if os.name == "nt":
                    pf_path = pf_path.replace("\\", "/")
                fp.write(f"file '{pf_path}'\n")

        movie_file_path = self.get_movie_file_path()
        commands = [
            config.ffmpeg_executable,
            "-y",  # overwrite output file if it exists
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            file_list,
            "-loglevel",
            "error",
            "-c",
            "copy",
            movie_file_path,
        ]
        if not self.includes_sound:
            commands.insert(-1, "-an")

        combine_process = sp.Popen(commands)
        combine_process.wait()

    def add_sound_to_video(self) -> None:
        movie_file_path = self.get_movie_file_path()
        stem, ext = os.path.splitext(movie_file_path)
        sound_file_path = stem + ".wav"
        # Makes sure sound file length will match video file
        self.add_audio_segment(AudioSegment.silent(0))
        self.audio_segment.export(
            sound_file_path,
            bitrate="312k",
        )
        temp_file_path = stem + "_temp" + ext
        commands = [
            config.ffmpeg_executable,
            "-i",
            movie_file_path,
            "-i",
            sound_file_path,
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
            "error",
            # "-shortest",
            temp_file_path,
        ]
        sp.call(commands)
        shutil.move(temp_file_path, movie_file_path)
        os.remove(sound_file_path)

    def save_final_image(self, image: Image) -> None:
        file_path = self.get_image_file_path()
        image.save(file_path)
        self.print_file_ready_message(file_path)

    def print_file_ready_message(self, file_path: str) -> None:
        if not self.quiet:
            log.info(f"File ready at {file_path}")

    def should_open_file(self) -> bool:
        return any(
            (
                self.show_file_location_upon_completion,
                self.open_file_upon_completion,
            )
        )

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

    def open_file(self) -> None:
        if self.quiet:
            curr_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        current_os = platform.system()
        file_paths = []

        if self.save_last_frame:
            file_paths.append(self.get_image_file_path())
        if self.write_to_movie:
            file_paths.append(self.get_movie_file_path())

        for file_path in file_paths:
            if current_os == "Windows":
                os.startfile(file_path)
            else:
                commands = []
                if current_os == "Linux":
                    commands.append("xdg-open")
                elif current_os.startswith("CYGWIN"):
                    commands.append("cygstart")
                else:  # Assume macOS
                    commands.append("open")

                if self.show_file_location_upon_completion:
                    commands.append("-R")

                commands.append(file_path)

                FNULL = open(os.devnull, "w")
                sp.call(commands, stdout=FNULL, stderr=sp.STDOUT)
                FNULL.close()

        if self.quiet:
            sys.stdout.close()
            sys.stdout = curr_stdout
