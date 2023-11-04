"""Utilities to create and set the config.

The main class exported by this module is :class:`ManimConfig`.  This class
contains all configuration options, including frame geometry (e.g. frame
height/width, frame rate), output (e.g. directories, logging), styling
(e.g. background color, transparency), and general behavior (e.g. writing a
movie vs writing a single frame).

See :doc:`/guides/configuration` for an introduction to Manim's configuration system.

"""
from __future__ import annotations

import argparse
import configparser
import copy
import errno
import logging
import os
import re
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .. import constants
from ..constants import RendererType
from ..typing import StrPath
from ..utils.color import ManimColor
from ..utils.tex import TexTemplate, TexTemplateFromFile


def config_file_paths() -> list[Path]:
    """The paths where ``.cfg`` files will be searched for.

    When manim is first imported, it processes any ``.cfg`` files it finds.  This
    function returns the locations in which these files are searched for.  In
    ascending order of precedence, these are: the library-wide config file, the
    user-wide config file, and the folder-wide config file.

    The library-wide config file determines manim's default behavior.  The
    user-wide config file is stored in the user's home folder, and determines
    the behavior of manim whenever the user invokes it from anywhere in the
    system.  The folder-wide config file only affects scenes that are in the
    same folder.  The latter two files are optional.

    These files, if they exist, are meant to loaded into a single
    :class:`configparser.ConfigParser` object, and then processed by
    :class:`ManimConfig`.

    Returns
    -------
    List[:class:`Path`]
        List of paths which may contain ``.cfg`` files, in ascending order of
        precedence.

    See Also
    --------
    :func:`make_config_parser`, :meth:`ManimConfig.digest_file`,
    :meth:`ManimConfig.digest_parser`

    Notes
    -----
    The location of the user-wide config file is OS-specific.

    """
    library_wide = Path.resolve(Path(__file__).parent / "default.cfg")
    if sys.platform.startswith("win32"):
        user_wide = Path.home() / "AppData" / "Roaming" / "Manim" / "manim.cfg"
    else:
        user_wide = Path.home() / ".config" / "manim" / "manim.cfg"
    folder_wide = Path("manim.cfg")
    return [library_wide, user_wide, folder_wide]


def make_config_parser(
    custom_file: StrPath | None = None,
) -> configparser.ConfigParser:
    """Make a :class:`ConfigParser` object and load any ``.cfg`` files.

    The user-wide file, if it exists, overrides the library-wide file.  The
    folder-wide file, if it exists, overrides the other two.

    The folder-wide file can be ignored by passing ``custom_file``.  However,
    the user-wide and library-wide config files cannot be ignored.

    Parameters
    ----------
    custom_file
        Path to a custom config file.  If used, the folder-wide file in the
        relevant directory will be ignored, if it exists.  If None, the
        folder-wide file will be used, if it exists.

    Returns
    -------
    :class:`ConfigParser`
        A parser containing the config options found in the .cfg files that
        were found.  It is guaranteed to contain at least the config options
        found in the library-wide file.

    See Also
    --------
    :func:`config_file_paths`

    """
    library_wide, user_wide, folder_wide = config_file_paths()
    # From the documentation: "An application which requires initial values to
    # be loaded from a file should load the required file or files using
    # read_file() before calling read() for any optional files."
    # https://docs.python.org/3/library/configparser.html#configparser.ConfigParser.read
    parser = configparser.ConfigParser()
    with library_wide.open() as file:
        parser.read_file(file)  # necessary file

    other_files = [user_wide, Path(custom_file) if custom_file else folder_wide]
    parser.read(other_files)  # optional files

    return parser


def _determine_quality(qual: str) -> str:
    for quality, values in constants.QUALITIES.items():
        if values["flag"] is not None and values["flag"] == qual:
            return quality

    return qual


class ManimConfig(MutableMapping):
    """Dict-like class storing all config options.

    The global ``config`` object is an instance of this class, and acts as a
    single source of truth for all of the library's customizable behavior.

    The global ``config`` object is capable of digesting different types of
    sources and converting them into a uniform interface.  These sources are
    (in ascending order of precedence): configuration files, command line
    arguments, and programmatic changes.  Regardless of how the user chooses to
    set a config option, she can access its current value using
    :class:`ManimConfig`'s attributes and properties.

    Notes
    -----
    Each config option is implemented as a property of this class.

    Each config option can be set via a config file, using the full name of the
    property.  If a config option has an associated CLI flag, then the flag is
    equal to the full name of the property.  Those that admit an alternative
    flag or no flag at all are documented in the individual property's
    docstring.

    Examples
    --------
    We use a copy of the global configuration object in the following
    examples for the sake of demonstration; you can skip these lines
    and just import ``config`` directly if you actually want to modify
    the configuration:

    .. code-block:: pycon

        >>> from manim import config as global_config
        >>> config = global_config.copy()

    Each config option allows for dict syntax and attribute syntax.  For
    example, the following two lines are equivalent,

    .. code-block:: pycon

        >>> from manim import WHITE
        >>> config.background_color = WHITE
        >>> config["background_color"] = WHITE

    The former is preferred; the latter is provided mostly for backwards
    compatibility.

    The config options are designed to keep internal consistency.  For example,
    setting ``frame_y_radius`` will affect ``frame_height``:

    .. code-block:: pycon

        >>> config.frame_height
        8.0
        >>> config.frame_y_radius = 5.0
        >>> config.frame_height
        10.0

    There are many ways of interacting with config options.  Take for example
    the config option ``background_color``.  There are three ways to change it:
    via a config file, via CLI flags, or programmatically.

    To set the background color via a config file, save the following
    ``manim.cfg`` file with the following contents.

    .. code-block::

       [CLI]
       background_color = WHITE

    In order to have this ``.cfg`` file apply to a manim scene, it needs to be
    placed in the same directory as the script,

    .. code-block:: bash

          project/
          ├─scene.py
          └─manim.cfg

    Now, when the user executes

    .. code-block:: bash

        manim scene.py

    the background of the scene will be set to ``WHITE``.  This applies regardless
    of where the manim command is invoked from.

    Command line arguments override ``.cfg`` files.  In the previous example,
    executing

    .. code-block:: bash

        manim scene.py -c BLUE

    will set the background color to BLUE, regardless of the contents of
    ``manim.cfg``.

    Finally, any programmatic changes made within the scene script itself will
    override the command line arguments.  For example, if ``scene.py`` contains
    the following

    .. code-block:: python

        from manim import *

        config.background_color = RED


        class MyScene(Scene):
            ...

    the background color will be set to RED, regardless of the contents of
    ``manim.cfg`` or the CLI arguments used when invoking manim.

    """

    _OPTS = {
        "assets_dir",
        "background_color",
        "background_opacity",
        "custom_folders",
        "disable_caching",
        "disable_caching_warning",
        "dry_run",
        "enable_wireframe",
        "ffmpeg_loglevel",
        "ffmpeg_executable",
        "format",
        "flush_cache",
        "frame_height",
        "frame_rate",
        "frame_width",
        "frame_x_radius",
        "frame_y_radius",
        "from_animation_number",
        "images_dir",
        "input_file",
        "media_embed",
        "media_width",
        "log_dir",
        "log_to_file",
        "max_files_cached",
        "media_dir",
        "movie_file_extension",
        "notify_outdated_version",
        "output_file",
        "partial_movie_dir",
        "pixel_height",
        "pixel_width",
        "plugins",
        "preview",
        "progress_bar",
        "quality",
        "save_as_gif",
        "save_sections",
        "save_last_frame",
        "save_pngs",
        "scene_names",
        "show_in_file_browser",
        "tex_dir",
        "tex_template",
        "tex_template_file",
        "text_dir",
        "upto_animation_number",
        "renderer",
        "enable_gui",
        "gui_location",
        "use_projection_fill_shaders",
        "use_projection_stroke_shaders",
        "verbosity",
        "video_dir",
        "sections_dir",
        "fullscreen",
        "window_position",
        "window_size",
        "window_monitor",
        "write_all",
        "write_to_movie",
        "zero_pad",
        "force_window",
        "no_latex_cleanup",
    }

    def __init__(self) -> None:
        self._d = {k: None for k in self._OPTS}

    # behave like a dict
    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key) -> bool:
        try:
            self.__getitem__(key)
            return True
        except AttributeError:
            return False

    def __getitem__(self, key) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, val: Any) -> None:
        getattr(ManimConfig, key).fset(self, val)  # fset is the property's setter

    def update(self, obj: ManimConfig | dict) -> None:
        """Digest the options found in another :class:`ManimConfig` or in a dict.

        Similar to :meth:`dict.update`, replaces the values of this object with
        those of ``obj``.

        Parameters
        ----------
        obj
            The object to copy values from.

        Returns
        -------
        None

        Raises
        -----
        :class:`AttributeError`
            If ``obj`` is a dict but contains keys that do not belong to any
            config options.

        See Also
        --------
        :meth:`~ManimConfig.digest_file`, :meth:`~ManimConfig.digest_args`,
        :meth:`~ManimConfig.digest_parser`

        """

        if isinstance(obj, ManimConfig):
            self._d.update(obj._d)
            if obj.tex_template:
                self.tex_template = obj.tex_template

        elif isinstance(obj, dict):
            # First update the underlying _d, then update other properties
            _dict = {k: v for k, v in obj.items() if k in self._d}
            for k, v in _dict.items():
                self[k] = v

            _dict = {k: v for k, v in obj.items() if k not in self._d}
            for k, v in _dict.items():
                self[k] = v

    # don't allow to delete anything
    def __delitem__(self, key: str):
        raise AttributeError("'ManimConfig' object does not support item deletion")

    def __delattr__(self, key: str):
        raise AttributeError("'ManimConfig' object does not support item deletion")

    # copy functions
    def copy(self) -> ManimConfig:
        """Deepcopy the contents of this ManimConfig.

        Returns
        -------
        :class:`ManimConfig`
            A copy of this object containing no shared references.

        See Also
        --------
        :func:`tempconfig`

        Notes
        -----
        This is the main mechanism behind :func:`tempconfig`.

        """
        return copy.deepcopy(self)

    def __copy__(self) -> ManimConfig:
        """See ManimConfig.copy()."""
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: dict[str, Any]) -> ManimConfig:
        """See ManimConfig.copy()."""
        c = ManimConfig()
        # Deepcopying the underlying dict is enough because all properties
        # either read directly from it or compute their value on the fly from
        # values read directly from it.
        c._d = copy.deepcopy(self._d, memo)
        return c

    # helper type-checking methods
    def _set_from_list(self, key: str, val: Any, values: list) -> None:
        """Set ``key`` to ``val`` if ``val`` is contained in ``values``."""
        if val in values:
            self._d[key] = val
        else:
            raise ValueError(f"attempted to set {key} to {val}; must be in {values}")

    def _set_from_enum(self, key: str, enum_value: Any, enum_class: EnumMeta) -> None:
        """Set ``key`` to the enum object with value ``enum_value`` in the given
        ``enum_class``.

        Tests::

            >>> from enum import Enum
            >>> class Fruit(Enum):
            ...     APPLE = 1
            ...     BANANA = 2
            ...     CANTALOUPE = 3
            >>> test_config = ManimConfig()
            >>> test_config._set_from_enum("fruit", 1, Fruit)
            >>> test_config._d['fruit']
            <Fruit.APPLE: 1>
            >>> test_config._set_from_enum("fruit", Fruit.BANANA, Fruit)
            >>> test_config._d['fruit']
            <Fruit.BANANA: 2>
            >>> test_config._set_from_enum("fruit", 42, Fruit)
            Traceback (most recent call last):
            ...
            ValueError: 42 is not a valid Fruit
        """
        self._d[key] = enum_class(enum_value)

    def _set_boolean(self, key: str | int, val: Any) -> None:
        """Set ``key`` to ``val`` if ``val`` is Boolean."""
        if val in [True, False]:
            self._d[key] = val
        else:
            raise ValueError(f"{key} must be boolean")

    def _set_tuple(self, key: str, val: tuple) -> None:
        if isinstance(val, tuple):
            self._d[key] = val
        else:
            raise ValueError(f"{key} must be tuple")

    def _set_str(self, key: str, val: Any) -> None:
        """Set ``key`` to ``val`` if ``val`` is a string."""
        if isinstance(val, str):
            self._d[key] = val
        elif not val:
            self._d[key] = ""
        else:
            raise ValueError(f"{key} must be str or falsy value")

    def _set_between(self, key: str, val: float, lo: float, hi: float) -> None:
        """Set ``key`` to ``val`` if lo <= val <= hi."""
        if lo <= val <= hi:
            self._d[key] = val
        else:
            raise ValueError(f"{key} must be {lo} <= {key} <= {hi}")

    def _set_int_between(self, key: str, val: int, lo: int, hi: int) -> None:
        """Set ``key`` to ``val`` if lo <= val <= hi."""
        if lo <= val <= hi:
            self._d[key] = val
        else:
            raise ValueError(
                f"{key} must be an integer such that {lo} <= {key} <= {hi}",
            )

    def _set_pos_number(self, key: str, val: int, allow_inf: bool) -> None:
        """Set ``key`` to ``val`` if ``val`` is a positive integer."""
        if isinstance(val, int) and val > -1:
            self._d[key] = val
        elif allow_inf and val in [-1, float("inf")]:
            self._d[key] = float("inf")
        else:
            raise ValueError(
                f"{key} must be a non-negative integer (use -1 for infinity)",
            )

    def __repr__(self) -> str:
        rep = ""
        for k, v in sorted(self._d.items(), key=lambda x: x[0]):
            rep += f"{k}: {v}, "
        return rep

    # builders
    def digest_parser(self, parser: configparser.ConfigParser) -> ManimConfig:
        """Process the config options present in a :class:`ConfigParser` object.

        This method processes arbitrary parsers, not only those read from a
        single file, whereas :meth:`~ManimConfig.digest_file` can only process one
        file at a time.

        Parameters
        ----------
        parser
            An object reflecting the contents of one or many ``.cfg`` files.  In
            particular, it may reflect the contents of multiple files that have
            been parsed in a cascading fashion.

        Returns
        -------
        self : :class:`ManimConfig`
            This object, after processing the contents of ``parser``.

        See Also
        --------
        :func:`make_config_parser`, :meth:`~.ManimConfig.digest_file`,
        :meth:`~.ManimConfig.digest_args`,

        Notes
        -----
        If there are multiple ``.cfg`` files to process, it is always more
        efficient to parse them into a single :class:`ConfigParser` object
        first, and then call this function once (instead of calling
        :meth:`~.ManimConfig.digest_file` multiple times).

        Examples
        --------
        To digest the config options set in two files, first create a
        ConfigParser and parse both files and then digest the parser:

        .. code-block:: python

            parser = configparser.ConfigParser()
            parser.read([file1, file2])
            config = ManimConfig().digest_parser(parser)

        In fact, the global ``config`` object is initialized like so:

        .. code-block:: python

            parser = make_config_parser()
            config = ManimConfig().digest_parser(parser)

        """
        self._parser = parser

        # boolean keys
        for key in [
            "notify_outdated_version",
            "write_to_movie",
            "save_last_frame",
            "write_all",
            "save_pngs",
            "save_as_gif",
            "save_sections",
            "preview",
            "show_in_file_browser",
            "log_to_file",
            "disable_caching",
            "disable_caching_warning",
            "flush_cache",
            "custom_folders",
            "enable_gui",
            "fullscreen",
            "use_projection_fill_shaders",
            "use_projection_stroke_shaders",
            "enable_wireframe",
            "force_window",
            "no_latex_cleanup",
        ]:
            setattr(self, key, parser["CLI"].getboolean(key, fallback=False))

        # int keys
        for key in [
            "from_animation_number",
            "upto_animation_number",
            "max_files_cached",
            # the next two must be set BEFORE digesting frame_width and frame_height
            "pixel_height",
            "pixel_width",
            "window_monitor",
            "zero_pad",
        ]:
            setattr(self, key, parser["CLI"].getint(key))

        # str keys
        for key in [
            "assets_dir",
            "verbosity",
            "media_dir",
            "log_dir",
            "video_dir",
            "sections_dir",
            "images_dir",
            "text_dir",
            "tex_dir",
            "partial_movie_dir",
            "input_file",
            "output_file",
            "movie_file_extension",
            "background_color",
            "renderer",
            "window_position",
        ]:
            setattr(self, key, parser["CLI"].get(key, fallback="", raw=True))

        # float keys
        for key in [
            "background_opacity",
            "frame_rate",
            # the next two are floats but have their own logic, applied later
            # "frame_width",
            # "frame_height",
        ]:
            setattr(self, key, parser["CLI"].getfloat(key))

        # tuple keys
        gui_location = tuple(
            map(int, re.split(r"[;,\-]", parser["CLI"]["gui_location"])),
        )
        setattr(self, "gui_location", gui_location)

        window_size = parser["CLI"][
            "window_size"
        ]  # if not "default", get a tuple of the position
        if window_size != "default":
            window_size = tuple(map(int, re.split(r"[;,\-]", window_size)))
        setattr(self, "window_size", window_size)

        # plugins
        self.plugins = parser["CLI"].get("plugins", fallback="", raw=True).split(",")
        # the next two must be set AFTER digesting pixel_width and pixel_height
        self["frame_height"] = parser["CLI"].getfloat("frame_height", 8.0)
        width = parser["CLI"].getfloat("frame_width", None)
        if width is None:
            self["frame_width"] = self["frame_height"] * self["aspect_ratio"]
        else:
            self["frame_width"] = width

        # other logic
        val = parser["CLI"].get("tex_template_file")
        if val:
            self.tex_template_file = val

        val = parser["CLI"].get("progress_bar")
        if val:
            setattr(self, "progress_bar", val)

        val = parser["ffmpeg"].get("loglevel")
        if val:
            self.ffmpeg_loglevel = val

        # TODO: Fix the mess above and below
        val = parser["ffmpeg"].get("ffmpeg_executable")
        setattr(self, "ffmpeg_executable", val)

        try:
            val = parser["jupyter"].getboolean("media_embed")
        except ValueError:
            val = None
        setattr(self, "media_embed", val)

        val = parser["jupyter"].get("media_width")
        if val:
            setattr(self, "media_width", val)

        val = parser["CLI"].get("quality", fallback="", raw=True)
        if val:
            self.quality = _determine_quality(val)

        return self

    def digest_args(self, args: argparse.Namespace) -> ManimConfig:
        """Process the config options present in CLI arguments.

        Parameters
        ----------
        args
            An object returned by :func:`.main_utils.parse_args()`.

        Returns
        -------
        self : :class:`ManimConfig`
            This object, after processing the contents of ``parser``.

        See Also
        --------
        :func:`.main_utils.parse_args()`, :meth:`~.ManimConfig.digest_parser`,
        :meth:`~.ManimConfig.digest_file`

        Notes
        -----
        If ``args.config_file`` is a non-empty string, ``ManimConfig`` tries to digest the
        contents of said file with :meth:`~ManimConfig.digest_file` before
        digesting any other CLI arguments.

        """
        # if the input file is a config file, parse it properly
        if args.file.suffix == ".cfg":
            args.config_file = args.file

        # if args.file is `-`, the animation code has to be taken from STDIN, so the
        # input file path shouldn't be absolute, since that file won't be read.
        if str(args.file) == "-":
            self.input_file = args.file

        # if a config file has been passed, digest it first so that other CLI
        # flags supersede it
        if args.config_file:
            self.digest_file(args.config_file)

        # read input_file from the args if it wasn't set by the config file
        if not self.input_file:
            self.input_file = Path(args.file).absolute()

        self.scene_names = args.scene_names if args.scene_names is not None else []
        self.output_file = args.output_file

        for key in [
            "notify_outdated_version",
            "preview",
            "show_in_file_browser",
            "write_to_movie",
            "save_last_frame",
            "save_pngs",
            "save_as_gif",
            "save_sections",
            "write_all",
            "disable_caching",
            "format",
            "flush_cache",
            "progress_bar",
            "transparent",
            "scene_names",
            "verbosity",
            "renderer",
            "background_color",
            "enable_gui",
            "fullscreen",
            "use_projection_fill_shaders",
            "use_projection_stroke_shaders",
            "zero_pad",
            "enable_wireframe",
            "force_window",
            "dry_run",
            "no_latex_cleanup",
        ]:
            if hasattr(args, key):
                attr = getattr(args, key)
                # if attr is None, then no argument was passed and we should
                # not change the current config
                if attr is not None:
                    self[key] = attr

        for key in [
            "media_dir",  # always set this one first
            "log_dir",
            "log_to_file",  # always set this one last
        ]:
            if hasattr(args, key):
                attr = getattr(args, key)
                # if attr is None, then no argument was passed and we should
                # not change the current config
                if attr is not None:
                    self[key] = attr

        if self["save_last_frame"]:
            self["write_to_movie"] = False

        # Handle the -n flag.
        nflag = args.from_animation_number
        if nflag:
            self.from_animation_number = nflag[0]
            try:
                self.upto_animation_number = nflag[1]
            except Exception:
                logging.getLogger("manim").info(
                    f"No end scene number specified in -n option. Rendering from {nflag[0]} onwards...",
                )

        # Handle the quality flags
        self.quality = _determine_quality(getattr(args, "quality", None))

        # Handle the -r flag.
        rflag = args.resolution
        if rflag:
            self.pixel_width = int(rflag[0])
            self.pixel_height = int(rflag[1])

        fps = args.frame_rate
        if fps:
            self.frame_rate = float(fps)

        # Handle --custom_folders
        if args.custom_folders:
            for opt in [
                "media_dir",
                "video_dir",
                "sections_dir",
                "images_dir",
                "text_dir",
                "tex_dir",
                "log_dir",
                "partial_movie_dir",
            ]:
                self[opt] = self._parser["custom_folders"].get(opt, raw=True)
            # --media_dir overrides the default.cfg file
            if hasattr(args, "media_dir") and args.media_dir:
                self.media_dir = args.media_dir

        # Handle --tex_template
        if args.tex_template:
            self.tex_template = TexTemplateFromFile(tex_filename=args.tex_template)

        if (
            self.renderer == RendererType.OPENGL
            and getattr(args, "write_to_movie") is None
        ):
            # --write_to_movie was not passed on the command line, so don't generate video.
            self["write_to_movie"] = False

        # Handle --gui_location flag.
        if getattr(args, "gui_location") is not None:
            self.gui_location = args.gui_location

        return self

    def digest_file(self, filename: str | os.PathLike) -> ManimConfig:
        """Process the config options present in a ``.cfg`` file.

        This method processes a single ``.cfg`` file, whereas
        :meth:`~ManimConfig.digest_parser` can process arbitrary parsers, built
        perhaps from multiple ``.cfg`` files.

        Parameters
        ----------
        filename
            Path to the ``.cfg`` file.

        Returns
        -------
        self : :class:`ManimConfig`
            This object, after processing the contents of ``filename``.

        See Also
        --------
        :meth:`~ManimConfig.digest_file`, :meth:`~ManimConfig.digest_args`,
        :func:`make_config_parser`

        Notes
        -----
        If there are multiple ``.cfg`` files to process, it is always more
        efficient to parse them into a single :class:`ConfigParser` object
        first and digesting them with one call to
        :meth:`~ManimConfig.digest_parser`, instead of calling this method
        multiple times.

        """
        if not Path(filename).is_file():
            raise FileNotFoundError(
                errno.ENOENT,
                "Error: --config_file could not find a valid config file.",
                str(filename),
            )

        return self.digest_parser(make_config_parser(filename))

    # config options are properties
    preview = property(
        lambda self: self._d["preview"] or self._d["enable_gui"],
        lambda self, val: self._set_boolean("preview", val),
        doc="Whether to play the rendered movie (-p).",
    )

    show_in_file_browser = property(
        lambda self: self._d["show_in_file_browser"],
        lambda self, val: self._set_boolean("show_in_file_browser", val),
        doc="Whether to show the output file in the file browser (-f).",
    )

    progress_bar = property(
        lambda self: self._d["progress_bar"],
        lambda self, val: self._set_from_list(
            "progress_bar",
            val,
            ["none", "display", "leave"],
        ),
        doc="Whether to show progress bars while rendering animations.",
    )

    log_to_file = property(
        lambda self: self._d["log_to_file"],
        lambda self, val: self._set_boolean("log_to_file", val),
        doc="Whether to save logs to a file.",
    )

    notify_outdated_version = property(
        lambda self: self._d["notify_outdated_version"],
        lambda self, val: self._set_boolean("notify_outdated_version", val),
        doc="Whether to notify if there is a version update available.",
    )

    write_to_movie = property(
        lambda self: self._d["write_to_movie"],
        lambda self, val: self._set_boolean("write_to_movie", val),
        doc="Whether to render the scene to a movie file (-w).",
    )

    save_last_frame = property(
        lambda self: self._d["save_last_frame"],
        lambda self, val: self._set_boolean("save_last_frame", val),
        doc="Whether to save the last frame of the scene as an image file (-s).",
    )

    write_all = property(
        lambda self: self._d["write_all"],
        lambda self, val: self._set_boolean("write_all", val),
        doc="Whether to render all scenes in the input file (-a).",
    )

    save_pngs = property(
        lambda self: self._d["save_pngs"],
        lambda self, val: self._set_boolean("save_pngs", val),
        doc="Whether to save all frames in the scene as images files (-g).",
    )

    save_as_gif = property(
        lambda self: self._d["save_as_gif"],
        lambda self, val: self._set_boolean("save_as_gif", val),
        doc="Whether to save the rendered scene in .gif format (-i).",
    )

    save_sections = property(
        lambda self: self._d["save_sections"],
        lambda self, val: self._set_boolean("save_sections", val),
        doc="Whether to save single videos for each section in addition to the movie file.",
    )

    enable_wireframe = property(
        lambda self: self._d["enable_wireframe"],
        lambda self, val: self._set_boolean("enable_wireframe", val),
        doc="Enable wireframe debugging mode in opengl.",
    )

    force_window = property(
        lambda self: self._d["force_window"],
        lambda self, val: self._set_boolean("force_window", val),
        doc="Set to force window when using the opengl renderer",
    )

    no_latex_cleanup = property(
        lambda self: self._d["no_latex_cleanup"],
        lambda self, val: self._set_boolean("no_latex_cleanup", val),
        doc="Prevents deletion of .aux, .dvi, and .log files produced by Tex and MathTex.",
    )

    @property
    def verbosity(self):
        """Logger verbosity; "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL" (-v)."""
        return self._d["verbosity"]

    @verbosity.setter
    def verbosity(self, val: str) -> None:
        """Verbosity level of the logger."""
        self._set_from_list(
            "verbosity",
            val,
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )
        logging.getLogger("manim").setLevel(val)

    @property
    def format(self):
        """File format; "png", "gif", "mp4", "webm" or "mov"."""
        return self._d["format"]

    @format.setter
    def format(self, val: str) -> None:
        """File format the renderer will output."""
        self._set_from_list(
            "format",
            val,
            [None, "png", "gif", "mp4", "mov", "webm"],
        )
        if self.format == "webm":
            logging.getLogger("manim").warning(
                "Output format set as webm, this can be slower than other formats",
            )

    ffmpeg_loglevel = property(
        lambda self: self._d["ffmpeg_loglevel"],
        lambda self, val: self._set_from_list(
            "ffmpeg_loglevel",
            val,
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ),
        doc="Verbosity level of ffmpeg (no flag).",
    )

    ffmpeg_executable = property(
        lambda self: self._d["ffmpeg_executable"],
        lambda self, val: self._set_str("ffmpeg_executable", val),
        doc="Manually specify the path to the ffmpeg executable",
    )

    media_embed = property(
        lambda self: self._d["media_embed"],
        lambda self, val: self._d.__setitem__("media_embed", val),
        doc="Embed videos in Jupyter notebook",
    )

    media_width = property(
        lambda self: self._d["media_width"],
        lambda self, val: self._d.__setitem__("media_width", val),
        doc="Media width in Jupyter notebook",
    )

    pixel_width = property(
        lambda self: self._d["pixel_width"],
        lambda self, val: self._set_pos_number("pixel_width", val, False),
        doc="Frame width in pixels (--resolution, -r).",
    )

    pixel_height = property(
        lambda self: self._d["pixel_height"],
        lambda self, val: self._set_pos_number("pixel_height", val, False),
        doc="Frame height in pixels (--resolution, -r).",
    )

    aspect_ratio = property(
        lambda self: self._d["pixel_width"] / self._d["pixel_height"],
        doc="Aspect ratio (width / height) in pixels (--resolution, -r).",
    )

    frame_height = property(
        lambda self: self._d["frame_height"],
        lambda self, val: self._d.__setitem__("frame_height", val),
        doc="Frame height in logical units (no flag).",
    )

    frame_width = property(
        lambda self: self._d["frame_width"],
        lambda self, val: self._d.__setitem__("frame_width", val),
        doc="Frame width in logical units (no flag).",
    )

    frame_y_radius = property(
        lambda self: self._d["frame_height"] / 2,
        lambda self, val: (
            self._d.__setitem__("frame_y_radius", val)
            or self._d.__setitem__("frame_height", 2 * val)
        ),
        doc="Half the frame height (no flag).",
    )

    frame_x_radius = property(
        lambda self: self._d["frame_width"] / 2,
        lambda self, val: (
            self._d.__setitem__("frame_x_radius", val)
            or self._d.__setitem__("frame_width", 2 * val)
        ),
        doc="Half the frame width (no flag).",
    )

    top = property(
        lambda self: self.frame_y_radius * constants.UP,
        doc="Coordinate at the center top of the frame.",
    )

    bottom = property(
        lambda self: self.frame_y_radius * constants.DOWN,
        doc="Coordinate at the center bottom of the frame.",
    )

    left_side = property(
        lambda self: self.frame_x_radius * constants.LEFT,
        doc="Coordinate at the middle left of the frame.",
    )

    right_side = property(
        lambda self: self.frame_x_radius * constants.RIGHT,
        doc="Coordinate at the middle right of the frame.",
    )

    frame_rate = property(
        lambda self: self._d["frame_rate"],
        lambda self, val: self._d.__setitem__("frame_rate", val),
        doc="Frame rate in frames per second.",
    )

    background_color = property(
        lambda self: self._d["background_color"],
        lambda self, val: self._d.__setitem__("background_color", ManimColor(val)),
        doc="Background color of the scene (-c).",
    )

    from_animation_number = property(
        lambda self: self._d["from_animation_number"],
        lambda self, val: self._d.__setitem__("from_animation_number", val),
        doc="Start rendering animations at this number (-n).",
    )

    upto_animation_number = property(
        lambda self: self._d["upto_animation_number"],
        lambda self, val: self._set_pos_number("upto_animation_number", val, True),
        doc="Stop rendering animations at this nmber.  Use -1 to avoid skipping (-n).",
    )

    max_files_cached = property(
        lambda self: self._d["max_files_cached"],
        lambda self, val: self._set_pos_number("max_files_cached", val, True),
        doc="Maximum number of files cached.  Use -1 for infinity (no flag).",
    )

    window_monitor = property(
        lambda self: self._d["window_monitor"],
        lambda self, val: self._set_pos_number("window_monitor", val, True),
        doc="The monitor on which the scene will be rendered",
    )
    flush_cache = property(
        lambda self: self._d["flush_cache"],
        lambda self, val: self._set_boolean("flush_cache", val),
        doc="Whether to delete all the cached partial movie files.",
    )

    disable_caching = property(
        lambda self: self._d["disable_caching"],
        lambda self, val: self._set_boolean("disable_caching", val),
        doc="Whether to use scene caching.",
    )

    disable_caching_warning = property(
        lambda self: self._d["disable_caching_warning"],
        lambda self, val: self._set_boolean("disable_caching_warning", val),
        doc="Whether a warning is raised if there are too much submobjects to hash.",
    )

    movie_file_extension = property(
        lambda self: self._d["movie_file_extension"],
        lambda self, val: self._set_from_list(
            "movie_file_extension",
            val,
            [".mp4", ".mov", ".webm"],
        ),
        doc="Either .mp4, .webm or .mov.",
    )

    background_opacity = property(
        lambda self: self._d["background_opacity"],
        lambda self, val: self._set_between("background_opacity", val, 0, 1),
        doc="A number between 0.0 (fully transparent) and 1.0 (fully opaque).",
    )

    frame_size = property(
        lambda self: (self._d["pixel_width"], self._d["pixel_height"]),
        lambda self, tup: (
            self._d.__setitem__("pixel_width", tup[0])
            or self._d.__setitem__("pixel_height", tup[1])
        ),
        doc="Tuple with (pixel width, pixel height) (no flag).",
    )

    @property
    def quality(self):
        """Video quality (-q)."""
        keys = ["pixel_width", "pixel_height", "frame_rate"]
        q = {k: self[k] for k in keys}
        for qual in constants.QUALITIES:
            if all(q[k] == constants.QUALITIES[qual][k] for k in keys):
                return qual
        return None

    @quality.setter
    def quality(self, qual: str) -> None:
        if qual is None:
            return
        if qual not in constants.QUALITIES:
            raise KeyError(f"quality must be one of {list(constants.QUALITIES.keys())}")
        q = constants.QUALITIES[qual]
        self.frame_size = q["pixel_width"], q["pixel_height"]
        self.frame_rate = q["frame_rate"]

    @property
    def transparent(self):
        """Whether the background opacity is 0.0 (-t)."""
        return self._d["background_opacity"] == 0.0

    @transparent.setter
    def transparent(self, val: bool) -> None:
        self._d["background_opacity"] = float(not val)
        self.resolve_movie_file_extension(val)

    @property
    def dry_run(self):
        """Whether dry run is enabled."""
        return self._d["dry_run"]

    @dry_run.setter
    def dry_run(self, val: bool) -> None:
        self._d["dry_run"] = val
        if val:
            self.write_to_movie = False
            self.write_all = False
            self.save_last_frame = False
            self.format = None

    @property
    def renderer(self):
        """The currently active renderer.

        Populated with one of the available renderers in :class:`.RendererType`.

        Tests::

            >>> test_config = ManimConfig()
            >>> test_config.renderer is None  # a new ManimConfig is unpopulated
            True
            >>> test_config.renderer = 'opengl'
            >>> test_config.renderer
            <RendererType.OPENGL: 'opengl'>
            >>> test_config.renderer = 42
            Traceback (most recent call last):
            ...
            ValueError: 42 is not a valid RendererType

        Check that capitalization of renderer types is irrelevant::

            >>> test_config.renderer = 'OpenGL'
            >>> test_config.renderer = 'cAirO'
        """
        return self._d["renderer"]

    @renderer.setter
    def renderer(self, val: str | RendererType) -> None:
        """The setter of the renderer property.

        Takes care of switching inheritance bases using the
        :class:`.ConvertToOpenGL` metaclass.
        """
        if isinstance(val, str):
            val = val.lower()
        renderer = RendererType(val)
        try:
            from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
            from manim.mobject.opengl.opengl_mobject import OpenGLMobject
            from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject

            from ..mobject.mobject import Mobject
            from ..mobject.types.vectorized_mobject import VMobject

            for cls in ConvertToOpenGL._converted_classes:
                if renderer == RendererType.OPENGL:
                    conversion_dict = {
                        Mobject: OpenGLMobject,
                        VMobject: OpenGLVMobject,
                    }
                else:
                    conversion_dict = {
                        OpenGLMobject: Mobject,
                        OpenGLVMobject: VMobject,
                    }

                cls.__bases__ = tuple(
                    conversion_dict.get(base, base) for base in cls.__bases__
                )
        except ImportError:
            # The renderer is set during the initial import of the
            # library for the first time. The imports above cause an
            # ImportError due to circular imports. However, the
            # metaclass sets stuff up correctly in this case, so we
            # can just do nothing.
            pass

        self._set_from_enum("renderer", renderer, RendererType)

    media_dir = property(
        lambda self: self._d["media_dir"],
        lambda self, val: self._set_dir("media_dir", val),
        doc="Main output directory.  See :meth:`ManimConfig.get_dir`.",
    )

    window_position = property(
        lambda self: self._d["window_position"],
        lambda self, val: self._d.__setitem__("window_position", val),
        doc="Set the position of preview window. You can use directions, e.g. UL/DR/ORIGIN/LEFT...or the position(pixel) of the upper left corner of the window, e.g. '960,540'",
    )

    window_size = property(
        lambda self: self._d["window_size"],
        lambda self, val: self._d.__setitem__("window_size", val),
        doc="The size of the opengl window. 'default' to automatically scale the window based on the display monitor.",
    )

    def resolve_movie_file_extension(self, is_transparent):
        if is_transparent:
            self.movie_file_extension = ".webm" if self.format == "webm" else ".mov"
        elif self.format == "webm":
            self.movie_file_extension = ".webm"
        elif self.format == "mov":
            self.movie_file_extension = ".mov"
        else:
            self.movie_file_extension = ".mp4"

    enable_gui = property(
        lambda self: self._d["enable_gui"],
        lambda self, val: self._set_boolean("enable_gui", val),
        doc="Enable GUI interaction.",
    )

    gui_location = property(
        lambda self: self._d["gui_location"],
        lambda self, val: self._set_tuple("gui_location", val),
        doc="Enable GUI interaction.",
    )

    fullscreen = property(
        lambda self: self._d["fullscreen"],
        lambda self, val: self._set_boolean("fullscreen", val),
        doc="Expand the window to its maximum possible size.",
    )

    use_projection_fill_shaders = property(
        lambda self: self._d["use_projection_fill_shaders"],
        lambda self, val: self._set_boolean("use_projection_fill_shaders", val),
        doc="Use shaders for OpenGLVMobject fill which are compatible with transformation matrices.",
    )

    use_projection_stroke_shaders = property(
        lambda self: self._d["use_projection_stroke_shaders"],
        lambda self, val: self._set_boolean("use_projection_stroke_shaders", val),
        doc="Use shaders for OpenGLVMobject stroke which are compatible with transformation matrices.",
    )

    zero_pad = property(
        lambda self: self._d["zero_pad"],
        lambda self, val: self._set_int_between("zero_pad", val, 0, 9),
        doc="PNG zero padding. A number between 0 (no zero padding) and 9 (9 columns minimum).",
    )

    def get_dir(self, key: str, **kwargs: str) -> Path:
        """Resolve a config option that stores a directory.

        Config options that store directories may depend on one another.  This
        method is used to provide the actual directory to the end user.

        Parameters
        ----------
        key
            The config option to be resolved.  Must be an option ending in
            ``'_dir'``, for example ``'media_dir'`` or ``'video_dir'``.

        kwargs
            Any strings to be used when resolving the directory.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the requested directory.  If the path resolves to the empty
            string, return ``None`` instead.

        Raises
        ------
        :class:`KeyError`
            When ``key`` is not a config option that stores a directory and
            thus :meth:`~ManimConfig.get_dir` is not appropriate; or when
            ``key`` is appropriate but there is not enough information to
            resolve the directory.

        Notes
        -----
        Standard :meth:`str.format` syntax is used to resolve the paths so the
        paths may contain arbitrary placeholders using f-string notation.
        However, these will require ``kwargs`` to contain the required values.

        Examples
        --------

        The value of ``config.tex_dir`` is ``'{media_dir}/Tex'`` by default,
        i.e. it is a subfolder of wherever ``config.media_dir`` is located.  In
        order to get the *actual* directory, use :meth:`~ManimConfig.get_dir`.

        .. code-block:: pycon

            >>> from manim import config as globalconfig
            >>> config = globalconfig.copy()
            >>> config.tex_dir
            '{media_dir}/Tex'
            >>> config.media_dir
            './media'
            >>> config.get_dir("tex_dir").as_posix()
            'media/Tex'

        Resolving directories is done in a lazy way, at the last possible
        moment, to reflect any changes in other config options:

        .. code-block:: pycon

            >>> config.media_dir = "my_media_dir"
            >>> config.get_dir("tex_dir").as_posix()
            'my_media_dir/Tex'

        Some directories depend on information that is not available to
        :class:`ManimConfig`. For example, the default value of `video_dir`
        includes the name of the input file and the video quality
        (e.g. 480p15). This informamtion has to be supplied via ``kwargs``:

        .. code-block:: pycon

            >>> config.video_dir
            '{media_dir}/videos/{module_name}/{quality}'
            >>> config.get_dir("video_dir")
            Traceback (most recent call last):
            KeyError: 'video_dir {media_dir}/videos/{module_name}/{quality} requires the following keyword arguments: module_name'
            >>> config.get_dir("video_dir", module_name="myfile").as_posix()
            'my_media_dir/videos/myfile/1080p60'

        Note the quality does not need to be passed as keyword argument since
        :class:`ManimConfig` does store information about quality.

        Directories may be recursively defined.  For example, the config option
        ``partial_movie_dir`` depends on ``video_dir``, which in turn depends
        on ``media_dir``:

        .. code-block:: pycon

            >>> config.partial_movie_dir
            '{video_dir}/partial_movie_files/{scene_name}'
            >>> config.get_dir("partial_movie_dir")
            Traceback (most recent call last):
            KeyError: 'partial_movie_dir {video_dir}/partial_movie_files/{scene_name} requires the following keyword arguments: scene_name'
            >>> config.get_dir(
            ...     "partial_movie_dir", module_name="myfile", scene_name="myscene"
            ... ).as_posix()
            'my_media_dir/videos/myfile/1080p60/partial_movie_files/myscene'

        Standard f-string syntax is used.  Arbitrary names can be used when
        defining directories, as long as the corresponding values are passed to
        :meth:`ManimConfig.get_dir` via ``kwargs``.

        .. code-block:: pycon

            >>> config.media_dir = "{dir1}/{dir2}"
            >>> config.get_dir("media_dir")
            Traceback (most recent call last):
            KeyError: 'media_dir {dir1}/{dir2} requires the following keyword arguments: dir1'
            >>> config.get_dir("media_dir", dir1="foo", dir2="bar").as_posix()
            'foo/bar'
            >>> config.media_dir = "./media"
            >>> config.get_dir("media_dir").as_posix()
            'media'

        """
        dirs = [
            "assets_dir",
            "media_dir",
            "video_dir",
            "sections_dir",
            "images_dir",
            "text_dir",
            "tex_dir",
            "log_dir",
            "input_file",
            "output_file",
            "partial_movie_dir",
        ]
        if key not in dirs:
            raise KeyError(
                "must pass one of "
                "{media,video,images,text,tex,log}_dir "
                "or {input,output}_file",
            )

        dirs.remove(key)  # a path cannot contain itself

        all_args = {k: self._d[k] for k in dirs}
        all_args.update(kwargs)
        all_args["quality"] = f"{self.pixel_height}p{self.frame_rate:g}"

        path = self._d[key]
        while "{" in path:
            try:
                path = path.format(**all_args)
            except KeyError as exc:
                raise KeyError(
                    f"{key} {self._d[key]} requires the following "
                    + "keyword arguments: "
                    + " ".join(exc.args),
                ) from exc
        return Path(path) if path else None

    def _set_dir(self, key: str, val: str | Path):
        if isinstance(val, Path):
            self._d.__setitem__(key, str(val))
        else:
            self._d.__setitem__(key, val)

    assets_dir = property(
        lambda self: self._d["assets_dir"],
        lambda self, val: self._set_dir("assets_dir", val),
        doc="Directory to locate video assets (no flag).",
    )

    log_dir = property(
        lambda self: self._d["log_dir"],
        lambda self, val: self._set_dir("log_dir", val),
        doc="Directory to place logs.  See :meth:`ManimConfig.get_dir`.",
    )

    video_dir = property(
        lambda self: self._d["video_dir"],
        lambda self, val: self._set_dir("video_dir", val),
        doc="Directory to place videos (no flag).  See :meth:`ManimConfig.get_dir`.",
    )

    sections_dir = property(
        lambda self: self._d["sections_dir"],
        lambda self, val: self._set_dir("sections_dir", val),
        doc="Directory to place section videos (no flag).  See :meth:`ManimConfig.get_dir`.",
    )

    images_dir = property(
        lambda self: self._d["images_dir"],
        lambda self, val: self._set_dir("images_dir", val),
        doc="Directory to place images (no flag).  See :meth:`ManimConfig.get_dir`.",
    )

    text_dir = property(
        lambda self: self._d["text_dir"],
        lambda self, val: self._set_dir("text_dir", val),
        doc="Directory to place text (no flag).  See :meth:`ManimConfig.get_dir`.",
    )

    tex_dir = property(
        lambda self: self._d["tex_dir"],
        lambda self, val: self._set_dir("tex_dir", val),
        doc="Directory to place tex (no flag).  See :meth:`ManimConfig.get_dir`.",
    )

    partial_movie_dir = property(
        lambda self: self._d["partial_movie_dir"],
        lambda self, val: self._set_dir("partial_movie_dir", val),
        doc="Directory to place partial movie files (no flag).  See :meth:`ManimConfig.get_dir`.",
    )

    custom_folders = property(
        lambda self: self._d["custom_folders"],
        lambda self, val: self._set_boolean("custom_folders", val),
        doc="Whether to use custom folder output.",
    )

    input_file = property(
        lambda self: self._d["input_file"],
        lambda self, val: self._set_dir("input_file", val),
        doc="Input file name.",
    )

    output_file = property(
        lambda self: self._d["output_file"],
        lambda self, val: self._set_dir("output_file", val),
        doc="Output file name (-o).",
    )

    scene_names = property(
        lambda self: self._d["scene_names"],
        lambda self, val: self._d.__setitem__("scene_names", val),
        doc="Scenes to play from file.",
    )

    @property
    def tex_template(self):
        """Template used when rendering Tex.  See :class:`.TexTemplate`."""
        if not hasattr(self, "_tex_template") or not self._tex_template:
            fn = self._d["tex_template_file"]
            if fn:
                self._tex_template = TexTemplateFromFile(tex_filename=fn)
            else:
                self._tex_template = TexTemplate()
        return self._tex_template

    @tex_template.setter
    def tex_template(self, val: TexTemplateFromFile | TexTemplate) -> None:
        if isinstance(val, (TexTemplateFromFile, TexTemplate)):
            self._tex_template = val

    @property
    def tex_template_file(self):
        """File to read Tex template from (no flag).  See :class:`.TexTemplateFromFile`."""
        return self._d["tex_template_file"]

    @tex_template_file.setter
    def tex_template_file(self, val: str) -> None:
        if val:
            if not os.access(val, os.R_OK):
                logging.getLogger("manim").warning(
                    f"Custom TeX template {val} not found or not readable.",
                )
            else:
                self._d["tex_template_file"] = Path(val)
        else:
            self._d["tex_template_file"] = val  # actually set the falsy value

    @property
    def plugins(self):
        """List of plugins to enable."""
        return self._d["plugins"]

    @plugins.setter
    def plugins(self, value):
        self._d["plugins"] = value


class ManimFrame(Mapping):
    _OPTS: set[str] = {
        "pixel_width",
        "pixel_height",
        "aspect_ratio",
        "frame_height",
        "frame_width",
        "frame_y_radius",
        "frame_x_radius",
        "top",
        "bottom",
        "left_side",
        "right_side",
    }
    _CONSTANTS: dict[str, np.ndarray] = {
        "UP": np.array((0.0, 1.0, 0.0)),
        "DOWN": np.array((0.0, -1.0, 0.0)),
        "RIGHT": np.array((1.0, 0.0, 0.0)),
        "LEFT": np.array((-1.0, 0.0, 0.0)),
        "IN": np.array((0.0, 0.0, -1.0)),
        "OUT": np.array((0.0, 0.0, 1.0)),
        "ORIGIN": np.array((0.0, 0.0, 0.0)),
        "X_AXIS": np.array((1.0, 0.0, 0.0)),
        "Y_AXIS": np.array((0.0, 1.0, 0.0)),
        "Z_AXIS": np.array((0.0, 0.0, 1.0)),
        "UL": np.array((-1.0, 1.0, 0.0)),
        "UR": np.array((1.0, 1.0, 0.0)),
        "DL": np.array((-1.0, -1.0, 0.0)),
        "DR": np.array((1.0, -1.0, 0.0)),
    }

    def __init__(self, c: ManimConfig) -> None:
        if not isinstance(c, ManimConfig):
            raise TypeError("argument must be instance of 'ManimConfig'")
        # need to use __dict__ directly because setting attributes is not
        # allowed (see __setattr__)
        self.__dict__["_c"] = c

    # there are required by parent class Mapping to behave like a dict
    def __getitem__(self, key: str | int) -> Any:
        if key in self._OPTS:
            return self._c[key]
        elif key in self._CONSTANTS:
            return self._CONSTANTS[key]
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterable:
        return iter(list(self._OPTS) + list(self._CONSTANTS))

    def __len__(self) -> int:
        return len(self._OPTS)

    # make this truly immutable
    def __setattr__(self, attr, val) -> None:
        raise TypeError("'ManimFrame' object does not support item assignment")

    def __setitem__(self, key, val) -> None:
        raise TypeError("'ManimFrame' object does not support item assignment")

    def __delitem__(self, key) -> None:
        raise TypeError("'ManimFrame' object does not support item deletion")


for opt in list(ManimFrame._OPTS) + list(ManimFrame._CONSTANTS):
    setattr(ManimFrame, opt, property(lambda self, o=opt: self[o]))
