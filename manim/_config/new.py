import logging
import re
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, get_args

from cloup import Context, HelpFormatter, HelpTheme, Style
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    JsonValue,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    model_validator,
)
from pydantic_core import core_schema
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from typing_extensions import Annotated, Self, TypeAlias

from manim import constants
from manim.typing import QualityLiteral, Vector3D
from manim.utils.color import BLACK, ManimColor
from manim.utils.tex import TexTemplate

WindowPosition: TypeAlias = Literal[
    "ORIGIN",
    "UP",
    "DOWN",
    "RIGHT",
    "LEFT",
    "IN",
    "OUT",
    "UL",
    "UR",
    "DL",
    "DR",
]

Dirs: TypeAlias = Literal[
    "media_dir",
    "assets_dir",
    "log_dir",
    "video_dir",
    "sections_dir",
    "images_dir",
    "text_dir",
    "tex_dir",
    "partial_movie_dir",
    "input_file",
    "output_file",
]

Verbosity: TypeAlias = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Logging related constants:

HIGHLIGHTED_KEYWORDS = [  # these keywords are highlighted specially
    "Played",
    "animations",
    "scene",
    "Reading",
    "Writing",
    "script",
    "arguments",
    "Invalid",
    "Aborting",
    "module",
    "File",
    "Rendering",
    "Rendered",
]

# Field validators:


def _from_frame_height(
    v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> float:
    try:
        return handler(v)
    except ValidationError:
        frame_height = info.data["frame_height"]
        # TODO can we use the aspect_ratio property instead?
        return frame_height * (info.data["pixel_width"] / info.data["pixel_height"])


def _tuple_from_string(v: Any) -> Any:
    if isinstance(v, str):
        return tuple(map(int, re.split(r"[;,\-]")))
    return v


def _from_comma_string(v: Any) -> list[str] | Any:
    if isinstance(v, str):
        return v.split(",")
    return v


class _ManimColorPydanticAnnotation:
    """A class meant to be used as metadata to an annotated Pydantic field.

    It provides the Pydantic core schema (to handle validation and serialization) alongside
    JSON schema (variable depending on the current mode).
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # `ManimColor` takes care of validating input
        class_validator = core_schema.no_info_plain_validator_function(ManimColor)

        return core_schema.json_or_python_schema(
            json_schema=class_validator,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ManimColor),
                    class_validator,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.to_rgb(),  # TODO Choose serialization
                when_used="json",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonValue:
        if handler.mode == "serialization":  # TODO adapt from serialization above
            return handler(
                core_schema.tuple_variable_schema(core_schema.float_schema())
            )
        else:
            return handler(
                core_schema.union_schema(
                    [
                        core_schema.int_schema(),
                        core_schema.str_schema(),
                        core_schema.tuple_variable_schema(
                            core_schema.float_schema(),
                            min_length=3,
                            max_length=4,
                        ),
                    ]
                )
            )


class JupyterConfig(BaseModel):
    """Jupyter related config."""

    media_embed: bool = False
    """Whether to embed videos in Jupyter notebook."""

    media_width: str = "60%"
    """Media width in Jupyter notebook."""

    model_config = {"validate_assignment": True}


class FFmpegConfig(BaseModel):
    """FFmpeg related config."""

    loglevel: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "ERROR"
    """Verbosity of FFmpeg."""

    executable: str = "ffmpeg"  # TODO use pathlib.Path?
    """Custom path to the FFmpeg executable."""

    model_config = {"validate_assignment": True}


class CliFormatting(BaseModel):
    """CLI related formatting config."""

    # Context settings
    align_option_groups: bool = True
    """Whether definition lists of all option groups of a command should be aligned."""

    align_sections: bool = True
    """Whether definition lists of all subcommands of a group should be aligned."""

    show_constraints: bool = True
    """Whether to include a "Constraint" section in the command help."""

    # Formatter settings
    indent_increment: int = 2
    """Width of each identation increment."""

    width: int = 80
    """Content line width, excluding the newline character."""

    col1_max_width: int = 30
    """The maximum width of the first column of a definition list."""

    col2_min_width: int = 35
    """The minimum width for the second column of a definition list."""

    col_spacing: int = 2
    """The number of spaces between the column boundaries of a definition list."""

    row_sep: str | None = None
    """An "extra" separator to insert between the rows of a definition list."""

    theme: Literal["dark", "light"] | None = None
    """The theme to be used."""

    # Theme settings
    invoked_command: str | None = None
    """Style of the invoked command name (in Usage)."""

    command_help: str | None = None
    """Style of the invoked command description (below Usage)."""

    heading: str | None = None
    """Style of help section headings."""

    constraint: str | None = None
    """Style of an option group constraint description."""

    section_help: str | None = None
    """Style of the help text of a section (the optional paragraph below the heading)."""

    col1: str | None = None
    """Style of the first column of a definition list (options and command names)."""

    col2: str | None = None
    """Style of the second column of a definition list (help text)."""

    epilog: str | None = None
    """Style of the epilog."""

    @property
    def context_settings(self) -> dict[str, Any]:
        """A dictionary to be used in cloup commands when providing context settings."""

        formatter_keys = {
            "indent_increment",
            "width",
            "col1_max_width",
            "col2_min_width",
            "col_spacing",
            "row_sep",
        }
        formatter_settings = {
            k: getattr(self, v, None)
            for k in formatter_keys
            if getattr(self, v, None) is not None
        }

        theme_settings: dict[str, Style] = {}
        theme_keys = {
            "command_help",
            "invoked_command",
            "heading",
            "constraint",
            "section_help",
            "col1",
            "col2",
            "epilog",
        }
        for k in theme_keys:
            v = getattr(self, k, None)
            if v is not None:
                theme_settings[k] = Style(v)

        if self.theme is None:
            theme = HelpTheme(**theme_settings)
        elif self.theme == "dark":
            theme = HelpTheme.dark().with_(**theme_settings)
        else:
            theme = HelpTheme.light().with_(**theme_settings)

        formatter = HelpFormatter.settings(
            theme=theme,
            **formatter_settings,
        )
        return Context.settings(
            align_option_groups=self.align_option_groups,
            align_sections=self.align_sections,
            show_constraints=self.show_constraints,
            formatter_settings=formatter,
        )

    model_config = {"validate_assignment": True}


class LoggingConfig(BaseModel):
    """Rich logging related config."""

    verbosity: Verbosity = "INFO"
    """The logger verbosity.

    Modifying this attribute will set the ``manim`` logger level accordingly.

    CLI switch: ``-v, --verbosity``.
    """

    log_timestamps: bool = True
    """Whether to log timestamps."""

    theme_config: dict[str, Any] = {}  # TODO validate by calling Theme(...)?
    """The Rich theme config."""

    @property
    def rich_theme(self) -> Theme:
        """The rich Theme, created from ``theme_config``."""
        theme_kwargs = {k.replace("_", "."): v for k, v in self.theme_config.items()}

        return Theme(**theme_kwargs)

    @property
    def consoles(self) -> tuple[Console, Console]:
        """The rich standard and error console."""
        return Console(theme=self.rich_theme), Console(
            theme=self.rich_theme, stderr=True
        )

    def make_logger(self) -> logging.Logger:
        """Make the Manim logger, by using the ``rich`` handler.

        Returns
        -------

        :class:`logging.Logger`
            The Manim logger.
        """
        console, _ = self.consoles
        rich_handler = RichHandler(
            console=console,
            show_time=self.log_timestamps,
            keywords=HIGHLIGHTED_KEYWORDS,
        )
        logger = logging.getLogger("manim")
        logger.addHandler(rich_handler)
        logger.setLevel(self.verbosity)

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        logging.getLogger("manim").setLevel(self.verbosity)
        return self

    model_config = {"validate_assignment": True}


class ManimConfig(BaseModel):
    """Manim config."""

    input_file: str = ""  # TODO default value?
    """Input file name."""

    output_file: str = ""  # TODO default value?
    """Output file name."""

    # Boolean switches:

    preview: bool = False
    """Whether to play the rendered movie.

    CLI switch: ``-p, --preview``.
    """

    show_in_file_browser: bool = False
    """
    Whether to show the output file in the file browser.

    CLI switch: ``-f, --show_in_file_browser``.
    """

    log_to_file: bool = False
    """Whether to save logs to a file.

    CLI switch: ``--log_to_file``.
    """

    notify_outdated_version: bool = True
    """Whether to notify if there is a version update available.

    CLI switch: ``--notify_outdated_version/--silent``.
    """

    save_last_frame: bool = False
    """Whether to save the last frame of the scene as an image file. If set to ``True``,
    forces ``write_to_movie`` to be ``False``.

    CLI switch: ``-s, --save_last_frame``.
    """

    write_to_movie: bool = True
    """Whether to render the scene to a movie file.

    CLI switch: ``--write_to_movie``.
    """

    write_all: bool = False
    """Whether to render all scenes in the input file.

    CLI switch: ``-a, --write_all``.
    """

    # TODO emit deprecation warning if set to True, via validator? Via future impl. of deprecation in Pydantic?
    save_pngs: bool = False
    """Whether to save all frames in the scene as images files. Deprecated.

    CLI switch: ``-g, --save_pngs``.
    """

    save_as_gif: bool = False
    """Whether to save the rendered scene in .gif format. Deprecated.

    CLI switch: ``-i, --save_as_gif``.
    """

    save_sections: bool = False
    """Whether to save single videos for each section in addition to the movie file.

    CLI switch: ``save_sections``.
    """

    enable_wireframe: bool = False
    """Whether to enable wireframe debugging mode in OpenGL.

    CLI switch: ``--enable_wireframe``.
    """

    force_window: bool = False
    """Whether to force window when in OpenGL.

    CLI switch: ``--force_window``.
    """

    no_latex_cleanup: bool = False
    """Prevents deletion of .aux, .dvi, and .log files produced by Tex and MathTex.

    CLI switch: ``--no_latex_cleanup``.
    """

    progress_bar: Literal["display", "leave", "none"] = "display"
    """The ``tqdm`` progress bar configuration:
    - ``display``: will be displayed (default).
    - ``leave``: whether to keep progress bars once finished.
    - ``none``: disable progress bars.

    CLI switch: ``--progress_bar``.
    """

    logging: LoggingConfig = LoggingConfig()
    """The logging configuration."""

    @property
    def verbosity(self) -> Verbosity:
        """The logger verbosity.

        CLI switch: ``-v, --verbosity``.
        """

        warnings.warn(
            DeprecationWarning,
            "Accessing 'ManimConfig.verbosity' is deprecated, instead "
            "use 'ManimConfig.logging.verbosity'",
            stacklevel=2,
        )
        return self.logging.verbosity

    format: Literal["png", "gif", "mp4", "webm", "mov"] = "mp4"
    """File format.

    CLI switch: ``--format``.
    """

    ffmpeg: FFmpegConfig = FFmpegConfig()

    @property
    def ffmpeg_loglevel(self) -> bool:
        """Whether to embed videos in Jupyter notebook."""

        warnings.warn(
            DeprecationWarning,
            "Accessing 'ManimConfig.ffmpeg_loglevel' is deprecated, instead "
            "use 'ManimConfig.ffmpeg.ffmpeg_loglevel'",
            stacklevel=2,
        )
        return self.ffmpeg.loglevel

    @property
    def ffmpeg_executable(self) -> bool:
        """Media width in Jupyter notebook."""

        warnings.warn(
            DeprecationWarning,
            "Accessing 'ManimConfig.ffmpeg_executable' is deprecated, instead "
            "use 'ManimConfig.ffmpeg.ffmpeg_executable'",
            stacklevel=2,
        )
        return self.ffmpeg.executable

    jupyter: JupyterConfig = JupyterConfig()
    """Jupyter notebook related config."""

    @property
    def media_embed(self) -> bool:
        """Whether to embed videos in Jupyter notebook."""

        warnings.warn(
            DeprecationWarning,
            "Accessing 'ManimConfig.media_embed' is deprecated, instead "
            "use 'ManimConfig.jupyter.media_embed'",
            stacklevel=2,
        )
        return self.jupyter.media_embed

    @property
    def media_width(self) -> bool:
        """Media width in Jupyter notebook."""

        warnings.warn(
            DeprecationWarning,
            "Accessing 'ManimConfig.media_width' is deprecated, instead "
            "use 'ManimConfig.jupyter.media_width'",
            stacklevel=2,
        )
        return self.jupyter.media_width

    cli_formatting: CliFormatting = CliFormatting()
    """CLI related formatting config."""

    pixel_width: Annotated[int, Field(gte=0)] = 1920
    """Frame width in pixels.

    CLI switch: ``-r, --resolution``.
    """

    pixel_height: Annotated[int, Field(gte=0)] = 1080
    """Frame height in pixels.

    CLI switch: ``-r, --resolution``.
    """

    @property
    def aspect_ratio(self) -> int:
        """Aspect ratio (``pixel_width / pixel_height``)."""
        return self.pixel_width / self.pixel_height

    frame_height: float = 8.0  # TODO extra validation?
    """Frame height in logical units."""

    # If not provided, the wrap validator would use `None` as a value,
    # fail when calling handler, thus deriving it from `frame_height`.
    frame_width: Annotated[
        float, WrapValidator(_from_frame_height), Field(validate_default=True)
    ] = None
    """Frame width in logical units."""

    @property
    def frame_y_radius(self) -> float:
        """Half the frame height."""
        return self.frame_height / 2

    @frame_y_radius.setter
    def frame_y_radius(self, value: float) -> None:
        self.frame_height = 2 * value

    @property
    def frame_x_radius(self) -> float:
        """Half the frame width."""
        return self.frame_width / 2

    @frame_x_radius.setter
    def frame_x_radius(self, value: float) -> None:
        self.frame_width = 2 * value

    @property
    def top(self) -> Vector3D:
        """Coordinate at the center top of the frame."""
        return self.frame_y_radius * constants.UP

    @property
    def bottom(self) -> Vector3D:
        """Coordinate at the center bottom of the frame."""
        return self.frame_y_radius * constants.DOWN

    @property
    def left_side(self) -> Vector3D:
        """Coordinate at the middle left of the frame."""
        return self.frame_x_radius * constants.LEFT

    @property
    def right_side(self) -> Vector3D:
        """Coordinate at the middle right of the frame."""
        return self.frame_x_radius * constants.RIGHT

    frame_rate: float = 60  # TODO validation?
    """Frame rate in frames per second.

    CLI switch: ``--fps/--frame_rate``.
    """

    background_color: Annotated[ManimColor, _ManimColorPydanticAnnotation] = BLACK
    """Background color of the scene.

    CLI Switch: ``-c``.
    """

    from_animation_number: int = 0  # TODO validation
    """Start rendering animations at this number.

    CLI switch: ``-n, --from_animation_number``.
    """

    upto_animation_number: int = -1  # TODO validation TODO `None` instead of `-1`?
    """Stop rendering animations at this number. Use ``-1`` to avoid skipping.

    CLI switch: ``TBD``
    """

    max_files_cached: int = 100  # TODO validation TODO `None` instead of `-1`?
    """Maximum number of files cached. Use ``-1`` for infinity."""

    window_monitor: int = 0
    """The monitor on which the scene will be rendered"""

    flush_cache: bool = False
    """Whether to delete all the cached partial movie files."""

    disable_caching: bool = False
    """Whether to disable scene caching."""

    disable_caching_warning: bool = False
    """Whether a warning should be raised if there are too much submobjects to hash."""

    # why is this and format needed?
    movie_file_extension: Literal[".mp4", ".mov", ".webm"] = ".mp4"
    """Either .mp4, .webm or .mov."""

    background_opacity: Annotated[float, Field(ge=0.0, le=1.0)] = 1
    """The background opacity as a number between 0.0 (fully transparent) and 1.0 (fully opaque)."""

    # TODO should we allow the setter? It currently calls resolve_movie_file_extension
    # but this should probably be called when setting background_opacity directy?
    @property
    def transparent(self) -> bool:
        """Whether the background opacity is 0.0."""
        return self.background_opacity == 0.0

    # TODO make this a field? extra handling in model_validator
    @property
    def frame_size(self) -> tuple[int, int]:
        """A tuple representing the frame size (``(pixel_width, pixel_height)``)."""
        return (self.pixel_width, self.pixel_height)

    @frame_size.setter
    def frame_size(self, value: tuple[int, int]) -> None:
        self.frame_width = value[0]
        self.frame_height = value[1]

    # TODO make this a field? extra handling in model_validator
    @property
    def quality(self) -> Literal[QualityLiteral, "custom"]:
        """Video quality."""
        keys = ["pixel_width", "pixel_height", "frame_rate"]
        q = {k: self[k] for k in keys}
        for qual in constants.QUALITIES:
            if all(q[k] == constants.QUALITIES[qual][k] for k in keys):
                return qual
        return "custom"

    @quality.setter
    def quality(self, value: QualityLiteral) -> None:
        if value not in constants.QUALITIES:
            raise ValueError(
                f"'quality' must be one of {list(constants.QUALITIES)}, got {value!r}"
            )
        qual = constants.QUALITIES[value]
        self.frame_size = qual["pixel_width"], qual["pixel_height"]
        self.frame_rate = qual["frame_rate"]

    dry_run: bool = False
    """Whether dry run is enabled.

    If enabled, the following fields will be set to ``False``:
    - ``write_to_movie``
    - ``write_all``
    - ``save_last_frame``

    CLI switch: ``--dry_run``.
    """

    renderer: constants.RendererType = constants.RendererType.CAIRO
    """The currently active renderer."""  # TODO complete docstring

    window_position: WindowPosition | str = "UR"
    """The position of the preview window.

    Either a direction (e.g. ``UL``, ``DR``, ``ORIGIN``) or the pixel position
    of the upper left corner of the window (e.g. ``'950,540'``).
    """

    # TODO set correct JSON schema for "validation" mode, waiting for pydantic#8208
    window_size: (
        Literal["default"]
        | Annotated[tuple[int, int], BeforeValidator(_tuple_from_string)]
    ) = "default"
    """The size of the OpenGL window. If set to ``default``, window will be scaled
    based on the display monitor.
    """

    enable_gui: bool = False
    """Enable GUI interaction.

    CLI switch: ``--enable_gui``.
    """

    # TODO set correct JSON schema for "validation" mode, waiting for pydantic#8208
    gui_location: Annotated[tuple[int, int], BeforeValidator(_tuple_from_string)] = (
        0,
        0,
    )
    """Location of the GUI, if enabled.

    CLI switch: ``--gui_location``.
    """

    fullscreen: bool = False
    """Expand the window to its maximum possible size.

    CLI switch: ``--fullscreen``.
    """

    use_projection_fill_shaders: bool = False
    """Use shaders for OpenGLVMobject fill which are compatible with transformation matrices.

    CLI switch: ``--use_projection_fill_shaders``.
    """

    use_projection_stroke_shaders: bool = False
    """Use shaders for OpenGLVMobject stroke which are compatible with transformation matrices.

    CLI switch: ``--use_projection_stroke_shaders``.
    """

    zero_pad: Annotated[int, Field(ge=0, le=9)] = 4
    """PNG zero padding. A number between 0 (no zero padding) and 9 (9 columns minimum).

    CLI switch: ``--zero_pad``.
    """

    # TODO set correct JSON schema for "validation" mode, waiting for pydantic#8208
    plugins: Annotated[list[str], BeforeValidator(_from_comma_string)] = []
    """List of plugins to enable."""

    media_dir: str = "./media"
    """Main output directory.

    CLI switch: ``--media_dir``.
    """

    assets_dir: str = "./"
    """Video assets directory."""

    log_dir: str = "{media_dir}/logs"
    """Logs output directory."""

    video_dir: str = "{media_dir}/videos/{module_name}/{quality}"
    """Videos output directory."""

    sections_dir: str = "{video_dir}/sections"
    """Video sections output directory."""

    images_dir: str = "{media_dir}/images/{module_name}"
    """Images output directory."""

    text_dir: str = "{media_dir}/texts"
    """Texts output directory."""

    tex_dir: str = "{media_dir}/Tex"
    """Tex files output directory."""

    partial_movie_dir: str = "{video_dir}/partial_movie_files/{scene_name}"
    """Partial movie files output directory."""

    custom_folders: bool = False
    """Whether to use custom folder output.

    CLI switch: ``--custom_folders``.
    """

    scene_names: list[str] = []
    """Scenes to play from file."""

    tex_template_file: Path | None = None
    """File to read Tex template from. See :class:`.TexTemplate`.

    CLI switch: ``--tex_template``.
    """

    @cached_property
    def tex_template(self) -> TexTemplate:
        """Template used when rendering Tex. See :class:`.TexTemplate`."""
        if self.tex_template_file is not None:
            return TexTemplate.from_file(self.tex_template_file)
        return TexTemplate()

    def get_dir(self, key: Dirs, **kwargs: Any) -> Path:
        dirs = list(get_args(Dirs))
        if key not in dirs:
            raise KeyError(
                "must pass one of "
                "{media,video,images,text,tex,log}_dir "
                "or {input,output}_file",
            )

        dirs.remove(key)

        all_args = {k: getattr(self, k) for k in dirs}
        all_args.update(kwargs)
        all_args["quality"] = f"{self.pixel_height}p{self.frame_rate:g}"

        path: str = getattr(self, key)
        while "{" in path:
            try:
                path = path.format_map(all_args)
            except KeyError as exc:
                raise KeyError(
                    f"{key} requires the following "
                    "keyword arguments: "
                    " ".join(exc.args),
                ) from exc

        return Path(path)

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.enable_gui:
            self.preview = True

        if self.format == "webm":
            warnings.warn(
                UserWarning,
                "Output format is set as 'webm', which can be slower in some cases.",
                stacklevel=2,
            )

        if self.save_last_frame:
            self._set_skip_validation("write_to_movie", False)
        if self.dry_run:
            self._set_skip_validation("write_to_movie", False)
            self._set_skip_validation("write_all", False)
            self._set_skip_validation("save_last_frame", False)
            # Is this desirable?
            # self._set_skip_validation("format", None)

        try:
            from manim.mobject.opengl.swap_classes import swap_converted_classes
        except ImportError:
            # The renderer is set during the initial import of the
            # library for the first time. The imports above cause an
            # ImportError due to circular imports. However, the
            # metaclass sets stuff up correctly in this case, so we
            # can just do nothing.
            pass
        else:
            swap_converted_classes(self.renderer)

        return self

    def _set_skip_validation(self, name: str, value: Any) -> None:
        """Workaround to be able to set fields without validation.

        In the context of ``ManimConfig``, this is useful to avoid recursion errors
        when setting values based on other field values in the after model validator
        (caused by ``validate_assignment``).

        WARNING: This is a private method that should only be used in model validation.
        Not all the checks implemented in ``BaseModel.__setattr__`` are present.
        """
        attr = getattr(self.__class__, name, None)
        if isinstance(attr, property):
            attr.__set__(self, value)
        else:
            self.__dict__[name] = value
            self.__pydantic_fields_set__.add(name)

    model_config = {"validate_assignment": True}
