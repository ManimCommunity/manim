import sys
from abc import ABC, abstractmethod
from configparser import ConfigParser
from pathlib import Path
from typing import ClassVar

from typing_extensions import override

from manim.typing import StrPath

from .new import ManimConfig


class ConfigProvider(ABC):
    """A class responsible for loading Manim config files.

    When Manim is first imported, it processes any supported config file it finds.
    Each subclass of this class is responsible for handling a specific configuration file format,
    and should implement the abstract ``get_config`` method to return an instance of ``ManimConfig``.
    """

    file_extension: ClassVar[str]

    @property
    def config_file_paths(self) -> tuple[Path, Path]:
        """The location of the user and folder configuration files.

        The user-wide configuration file is stored in the user's home directory, and determines
        the behavior of manim whenever the user invokes it from anywhere in the system.
        The folder-wide config file only affects scenes that are in the same folder.

        Notes
        -----
        The location of the user-wide config file is OS-specific.
        """
        filename = f"manim{self.file_extension}"
        if sys.platform == "win32":
            user_wide = Path.home() / "AppData" / "Roaming" / "Manim" / filename
        else:
            user_wide = Path.home() / ".config" / "manim" / filename
        folder_wide = Path(filename)
        return [user_wide, folder_wide]

    @property
    def available(self) -> bool:
        """Whether one of the user or folder config files exists."""
        return any(path.is_file() for path in self.config_file_paths)

    @abstractmethod
    def get_config(self, custom_path: StrPath | None = None) -> ManimConfig:
        """Make a :class:`ManimConfig` object by loading the configuration files.

        The folder-wide file, if it exists, overrides the user-wide.
        The folder-wide file can be ignored by passing ``custom_path``.

        Parameters
        ----------
        custom_path
            Path to a custom config file.  If used, the folder-wide file in the
            relevant directory will be ignored, if it exists.  If ``None``, the
            folder-wide file will be used, if it exists.

        Returns
        -------
        :class:`ManimConfig`
            A Manim config containing the config options found in the config files that
            were found.
        """
        pass


class CfgProvider(ConfigProvider):
    """A config provider responsible for loading ``.cfg`` files, using
    :class:`ConfigParser`.
    """

    file_extension: ClassVar[str] = "cfg"

    @override
    def get_config(self, custom_path: StrPath | None = None) -> ManimConfig:
        data = {}

        parser = ConfigParser()
        user_wide, folder_wide = self.config_file_paths
        parser.read([user_wide, Path(custom_path) if custom_path else folder_wide])

        # CLI section
        cli_section = parser["CLI"]
        verbosity = cli_section.pop("verbosity", None)
        data.update(cli_section)

        # CLI_CTX section
        data["cli_formatting"] = parser["CLI_CTX"]

        logger_section = parser["logger"]
        theme_config = {
            k: v
            for k, v in logger_section.items()
            if k not in {"log_timestamps", "verbosity"}
        }
        logging_config = {"theme_config": theme_config}
        if "log_timestamps" in logger_section:
            logging_config["log_timestamps"] = logger_section["log_timestamps"]
        if verbosity is not None:
            # TODO raise deprecation, 'verbosity' should be in 'logger'
            logger_section.setdefault("verbosity", verbosity)

        data["logging"] = logging_config

        data["ffmpeg"] = parser["ffmpeg"]
        ffmeg_executable = data["ffmpeg"].pop("ffmeg_executable", None)
        if ffmeg_executable is not None:
            # TODO raise deprecation, 'ffmeg_executable' should be 'executable'
            data["ffmpeg"].setdefault("executable", ffmeg_executable)

        data["jupyter"] = parser["jupyter"]

        return ManimConfig.model_validate(data)


class TOMLProvider(ConfigProvider):
    file_extension: ClassVar[str] = "toml"

    @override
    def get_config(self, custom_path: StrPath | None = None) -> ManimConfig:
        raise NotImplementedError("TOML config support is not supported yet.")
